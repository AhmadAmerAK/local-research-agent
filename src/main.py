"""CLI entry point; the PyQt worker can call ResearchWorkflow.run() directly."""
from pathlib import Path
import argparse
import logging
import sys

from dotenv import load_dotenv

# Set process environment before any harness/telemetry imports.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

from agent.config import Settings  # noqa: E402
from agent.orchestrator import ResearchWorkflow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Local research agent")
    parser.add_argument("question", nargs="*", help="A research question")
    parser.add_argument("--rag", choices=("on", "off"), help="on: KB only; off: local model only; omitted: existing research")
    parser.add_argument("--reranking", choices=("NONE", "MANAGED"), default=None, help="KB reranking (default NONE; requires --rag on)")
    args = parser.parse_args()
    if args.reranking and args.rag != "on":
        parser.error("--reranking requires --rag on")
    question = " ".join(args.question).strip() or input("Research question: ").strip()
    if not question:
        print("No research question supplied.", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.WARNING)
    workflow = None
    try:
        settings = Settings.from_env()
        status = lambda s: print(f"[status] {s}", file=sys.stderr)
        if args.rag is None:
            workflow = ResearchWorkflow(settings, on_status=status)
        else:
            from agent.rag import LocalAnswerWorkflow
            workflow = LocalAnswerWorkflow(settings, mode="rag" if args.rag == "on" else "local",
                                           reranking=args.reranking or "NONE", on_status=status)
        output = workflow.run(question)
    except Exception as exc:
        print(f"Research failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if workflow is not None:
            workflow.close()
    print(output.answer)
    for warning in output.warnings:
        print(f"[warning] {warning}", file=sys.stderr)
    if output.run_directory:
        print(f"[saved] {output.run_directory}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
