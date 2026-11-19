"""Capture at most 3 KB calls once; replay saved responses with no AWS/web access.

Raw KB passages stay in an ignored private cache, never in unit-test fixtures.
Capture is explicit; replay never falls back to a live request on a cache miss.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent.config import Settings
from agent.rag import INSUFFICIENT, LocalAnswerWorkflow


def save(path, value):
    with path.open("w", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2)


def key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


class CaptureClient:
    def __init__(self, client, directory):
        self.client, self.directory = client, directory

    def retrieve(self, **request):
        path = self.directory / (key(request) + ".json")
        if path.exists():
            entry = json.loads(path.read_text())
            if "response" not in entry:
                raise RuntimeError("A previous attempt failed or was interrupted; refusing an automatic retry")
            return deepcopy(entry["response"])
        if len(list(self.directory.glob("*.json"))) >= 3:
            raise RuntimeError("Three-request cache budget exhausted")
        # Reserve an attempt before calling AWS, even if it fails or is interrupted.
        with path.open("x", encoding="utf-8") as handle:
            os.chmod(path, 0o600)
            json.dump({"request": request, "timestamp": datetime.now(timezone.utc).isoformat()}, handle)
        response = self.client.retrieve(**request)
        response = {"retrievalResults": response.get("retrievalResults", []),
                    "guardrailAction": response.get("guardrailAction")}
        save(path, {"request": request, "response": response,
                    "timestamp": datetime.now(timezone.utc).isoformat()})
        return deepcopy(response)


class ReplayClient:
    def __init__(self, entry):
        self.entry, self.calls = entry, 0

    def retrieve(self, **request):
        if request != self.entry["request"]:
            raise AssertionError("Replay request differs from the cached request")
        self.calls += 1
        return deepcopy(self.entry["response"])


def blocked_network(sock, address):
    # Ollama HTTP only. Block any attempted AWS/web connection during replay.
    if isinstance(address, tuple) and str(address[0]) in {"127.0.0.1", "::1"}:
        return ORIGINAL_CONNECT(sock, address)
    raise AssertionError("Replay attempted a non-local network connection")


ORIGINAL_CONNECT = socket.socket.connect


def capture(args, settings):
    import boto3
    from botocore.config import Config
    if not args.question or not args.limitation:
        raise ValueError("capture requires --question and --limitation")
    if not os.getenv("BEDROCK_KNOWLEDGE_BASE_ID") or not os.getenv("AWS_REGION"):
        raise ValueError("Configure BEDROCK_KNOWLEDGE_BASE_ID and AWS_REGION first")
    client = boto3.client("bedrock-agent-runtime", region_name=os.environ["AWS_REGION"],
                          config=Config(connect_timeout=5, read_timeout=30,
                                        retries={"total_max_attempts": 1}))
    recorded = CaptureClient(client, args.cache / "retrievals")
    cases = [(args.question, "NONE"), (args.limitation, "NONE")]
    if args.managed:
        cases.insert(1, (args.question, "MANAGED"))
    try:
        for question, reranking in cases:
            flow = LocalAnswerWorkflow(settings, reranking=reranking, client=recorded)
            results = flow._retrieve(question)  # Exercise the application's exact request builder.
            print(json.dumps({"reranking": reranking, "hits": len(results),
                              "cached_attempts": len(list((args.cache / "retrievals").glob("*.json")))}), flush=True)
    finally:
        client.close()


def replay(args, settings):
    if settings.ollama_host not in {"http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"}:
        raise ValueError("This replay harness requires local Ollama on port 11434")
    paths = sorted((args.cache / "retrievals").glob("*.json"))
    fingerprints = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    entries = [json.loads(path.read_text()) for path in paths]
    if not entries or any("response" not in entry for entry in entries):
        raise ValueError("Replay requires completed cached responses; it never fetches missing results")
    summaries, baseline_questions = [], set()
    with patch.object(socket.socket, "connect", blocked_network):
        for index, entry in enumerate(entries, 1):
            request = entry["request"]
            question = request["retrievalQuery"]["text"]
            reranking = request["retrievalConfiguration"]["managedSearchConfiguration"]["rerankingModelType"]
            # Only injected replay client sees this ID. No AWS SDK is used here.
            os.environ["BEDROCK_KNOWLEDGE_BASE_ID"] = request["knowledgeBaseId"]
            os.environ.setdefault("AWS_REGION", "eu-west-2")
            client = ReplayClient(entry)
            flow = LocalAnswerWorkflow(settings, client=client, reranking=reranking,
                                       output_dir=args.cache / "answers")
            generate = flow._generate

            async def observed_generate(question, context):
                raw = await generate(question, context)
                save(args.cache / f"raw-answer-{index}.json", {"question": question, "answer": raw})
                return raw

            flow._generate = observed_generate
            result = flow.run(question)
            assert client.calls == 1
            assert result.answer == INSUFFICIENT or any(f"[{s.source_id}]" in result.answer for s in result.sources)
            summaries.append({"case": index, "question": question, "reranking": reranking,
                              "retrieved_hits": len(entry["response"]["retrievalResults"]),
                              "context_sources": len(result.sources), "abstained": result.answer == INSUFFICIENT,
                              "warnings": result.warnings, "run_directory": result.run_directory,
                              "answer": result.answer})
            print(json.dumps({k: summaries[-1][k] for k in ("case", "reranking", "retrieved_hits", "context_sources", "abstained")}), flush=True)
            if question not in baseline_questions:
                baseline_questions.add(question)
                baseline = LocalAnswerWorkflow(settings, mode="local", client=client,
                                               output_dir=args.cache / "answers").run(question)
                assert client.calls == 1  # Local baseline must not retrieve.
                summaries.append({"mode": "local", "question": question, "answer": baseline.answer,
                                  "run_directory": baseline.run_directory})
    assert fingerprints == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    save(args.cache / "replay-summary.json", {"model": settings.ollama_model, "cache_sha256": fingerprints,
         "aws_calls_during_replay": 0, "results": summaries,
         "limitation": "Citation syntax checks do not establish factual support; compare passages and answers."})


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "replay"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--question")
    parser.add_argument("--limitation")
    parser.add_argument("--managed", action="store_true", help="One optional same-question MANAGED retrieval")
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    # Keep confidential response caches inside the already ignored run tree.
    if not args.cache.is_relative_to(ROOT / "src/agent/memory/runs"):
        parser.error("--cache must be under src/agent/memory/runs (ignored by git)")
    args.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    (args.cache / "retrievals").mkdir(exist_ok=True, mode=0o700)
    try:
        settings = Settings.from_env()
        (capture if args.action == "capture" else replay)(args, settings)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
