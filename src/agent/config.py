"""Typed configuration; loaded once at application startup."""
from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = Path(__file__).resolve().parent


def _integer(name: str, default: int | None) -> int | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Settings:
    openalex_api_key: str
    tavily_api_key: str
    ollama_host: str
    ollama_model: str
    max_search_queries: int
    max_sources: int
    max_research_rounds: int
    min_required_sources: int
    context_window: int
    enable_skills: bool
    output_tokens: int = 3072
    agent_max_turns: int | None = None
    max_query_chars: int = 1500
    search_results_per_query: int | None = None
    search_snippet_chars: int = 700
    evidence_page_chars: int = 2000
    evidence_catalogue_page_size: int = 20

    def __post_init__(self):
        # Validate direct construction/dataclass replacement as well as .env.
        for name in ("max_search_queries", "max_research_rounds", "max_sources",
                     "min_required_sources", "context_window", "output_tokens",
                     "max_query_chars", "search_snippet_chars", "evidence_page_chars",
                     "evidence_catalogue_page_size", "agent_max_turns", "search_results_per_query"):
            value = getattr(self, name)
            if value is None and name in {"agent_max_turns", "search_results_per_query"}:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name.upper()} must be a positive integer")
        if self.min_required_sources > self.max_sources:
            raise ValueError("MIN_REQUIRED_SOURCES cannot exceed MAX_SOURCES")
        if self.output_tokens >= self.context_window:
            raise ValueError("OUTPUT_TOKENS must be less than CONTEXT_WINDOW to leave room for input")

    @property
    def search_budget(self) -> int:
        return self.max_search_queries * self.max_research_rounds

    @property
    def agent_turn_budget(self) -> int:
        # Preserve the previous allowance for routing, evidence reads and answers.
        return self.agent_max_turns if self.agent_max_turns is not None else self.search_budget + 12

    @property
    def results_per_query(self) -> int:
        return self.search_results_per_query if self.search_results_per_query is not None else self.max_sources

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openalex_api_key=os.getenv("OPENALEX_API_KEY", "").strip(),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            ollama_host=(os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip().rstrip("/"),
            ollama_model=(os.getenv("OLLAMA_MODEL") or "qwen3:4b-instruct").strip().removeprefix("ollama/"),
            max_search_queries=_integer("MAX_SEARCH_QUERIES", 2),
            max_sources=_integer("MAX_SOURCES", 5),
            max_research_rounds=_integer("MAX_RESEARCH_ROUNDS", 1),
            min_required_sources=_integer("MIN_REQUIRED_SOURCES", 3),
            context_window=_integer("CONTEXT_WINDOW", 8192),
            enable_skills=(os.getenv("ENABLE_SKILLS") or "false").lower() in {"1", "true", "yes"},
            output_tokens=_integer("OUTPUT_TOKENS", 3072),
            agent_max_turns=_integer("AGENT_MAX_TURNS", None),
            max_query_chars=_integer("MAX_QUERY_CHARS", 1500),
            search_results_per_query=_integer("SEARCH_RESULTS_PER_QUERY", None),
            search_snippet_chars=_integer("SEARCH_SNIPPET_CHARS", 700),
            evidence_page_chars=_integer("EVIDENCE_PAGE_CHARS", 2000),
            evidence_catalogue_page_size=_integer("EVIDENCE_CATALOGUE_PAGE_SIZE", 20),
        )
