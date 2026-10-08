"""Provider-agnostic evidence records and local, inspectable run storage."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit


@dataclass
class Source:
    source_id: str
    provider: str
    title: str
    url: str
    snippet: str
    authors: list[str]
    year: int | None = None
    published_date: str | None = None
    doi: str | None = None
    score: float | None = None
    query: str = ""


def abstract_from_inverted_index(index: object) -> str:
    """Decode OpenAlex abstract_inverted_index into readable plain text."""
    if not isinstance(index, dict):
        return ""
    positions: dict[int, str] = {}
    for word, offsets in index.items():
        if not isinstance(word, str) or not isinstance(offsets, list):
            continue
        for offset in offsets:
            if isinstance(offset, int) and 0 <= offset < 10000:
                positions[offset] = word
    return " ".join(positions[p] for p in sorted(positions))


def _clean(value: object, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def normalise_openalex(raw: dict) -> Source | None:
    title = _clean(raw.get("display_name") or raw.get("title"), 300)
    url = _clean(raw.get("id") or raw.get("doi"), 600)
    if not title or not url:
        return None
    authors = [
        _clean((item.get("author") or {}).get("display_name"), 90)
        for item in (raw.get("authorships") or []) if isinstance(item, dict)
    ][:4]
    abstract = abstract_from_inverted_index(raw.get("abstract_inverted_index"))
    publication_year = raw.get("publication_year")
    year = publication_year if isinstance(publication_year, int) else None
    doi = raw.get("doi")
    return Source("", "openalex", title, url, _clean(abstract, 2200),
                  [a for a in authors if a], year=year,
                  doi=doi if isinstance(doi, str) else None)


def normalise_tavily(raw: dict) -> Source | None:
    title = _clean(raw.get("title"), 300)
    url = _clean(raw.get("url"), 600)
    if not title or not url or not url.lower().startswith(("https://", "http://")):
        return None
    score = raw.get("score")
    date = raw.get("published_date")
    return Source("", "tavily", title, url, _clean(raw.get("content"), 2200), [],
                  published_date=date if isinstance(date, str) else None,
                  score=float(score) if isinstance(score, (float, int)) else None)


def _dedup_key(source: Source) -> str:
    if source.doi:
        return source.doi.lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    parsed = urlsplit(source.url.lower())
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


class EvidenceStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sources: list[Source] = []
        self._keys: set[str] = set()

    def add(self, incoming: list[Source]) -> list[Source]:
        added = []
        for item in incoming:
            key = _dedup_key(item)
            if key in self._keys:
                continue
            self._keys.add(key)
            item.source_id = f"S{len(self.sources) + 1}"
            self.sources.append(item)
            added.append(item)
        self.save()
        return added

    def save_raw(self, provider: str, query: str, results: list[dict]) -> None:
        # Raw provider responses stay on disk, not in the model's prompt.
        with (self.directory / "raw_results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"provider": provider, "query": query,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "results": results}, ensure_ascii=False) + "\n")

    def save(self) -> None:
        output = self.directory / "sources.json"
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps([asdict(x) for x in self.sources], indent=2,
                                   ensure_ascii=False), encoding="utf-8")
        temp.replace(output)

    def select(self, limit: int) -> list[Source]:
        """Prefer usable evidence and diversify across search queries."""
        groups: dict[str, list[Source]] = {}
        for source in self.sources:
            groups.setdefault(source.query or "default", []).append(source)
        for group in groups.values():
            # Prefer sources with actual evidence rather than just a title.
            # Respect original provider order when evidence availability ties.
            group.sort(key=lambda x: bool(x.snippet), reverse=True)
        chosen: list[Source] = []
        while len(chosen) < limit and any(groups.values()):
            for group in groups.values():
                if group:
                    chosen.append(group.pop(0))
                    if len(chosen) == limit:
                        break
        return chosen

    def validate_citations(self, report: str) -> tuple[list[str], list[str]]:
        cited = list(dict.fromkeys(re.findall(r"\[S\d+\]", report)))
        known = {f"[{x.source_id}]" for x in self.sources}
        invalid = [cid for cid in cited if cid not in known]
        return cited, invalid
