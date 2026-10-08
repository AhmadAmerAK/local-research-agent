"""Typed configuration; loaded once at application startup."""
from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = Path(__file__).resolve().parent


def _integer(name: str, default: int, lowest: int, highest: int) -> int:
    value = int(os.getenv(name) or default)
    if not lowest <= value <= highest:
        raise ValueError(f"{name} must be between {lowest} and {highest}")
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

    @classmethod
    def from_env(cls) -> "Settings":
        max_sources = _integer("MAX_SOURCES", 5, 1, 20)
        minimum = _integer("MIN_REQUIRED_SOURCES", 3, 1, 20)
        if minimum > max_sources:
            raise ValueError("MIN_REQUIRED_SOURCES cannot exceed MAX_SOURCES")
        return cls(
            openalex_api_key=os.getenv("OPENALEX_API_KEY", "").strip(),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            ollama_host=(os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip().rstrip("/"),
            ollama_model=(os.getenv("OLLAMA_MODEL") or "qwen3:4b-instruct").strip().removeprefix("ollama/"),
            max_search_queries=_integer("MAX_SEARCH_QUERIES", 2, 1, 5),
            max_sources=max_sources,
            max_research_rounds=_integer("MAX_RESEARCH_ROUNDS", 1, 1, 3),
            min_required_sources=minimum,
            context_window=_integer("CONTEXT_WINDOW", 8192, 2048, 32768),
            enable_skills=(os.getenv("ENABLE_SKILLS") or "false").lower() in {"1", "true", "yes"},
        )
