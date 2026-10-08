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
    args = parser.parse_args()
    question = " ".join(args.question).strip() or input("Research question: ").strip()
    if not question:
        print("No research question supplied.", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.WARNING)
    try:
        settings = Settings.from_env()
        workflow = ResearchWorkflow(settings, on_status=lambda s: print(f"[status] {s}", file=sys.stderr))
        output = workflow.run(question)
    except Exception as exc:
        print(f"Research failed: {exc}", file=sys.stderr)
        return 1
    print(output.answer)
    for warning in output.warnings:
        print(f"[warning] {warning}", file=sys.stderr)
    if output.run_directory:
        print(f"[saved] {output.run_directory}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
