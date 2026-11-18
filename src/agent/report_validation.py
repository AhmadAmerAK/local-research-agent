"""Deterministic report checks; these do not establish factual claim support."""
import re

from .evidence import CITATION_PATTERN, normalize_citations, parse_citations


_REFERENCES = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*|__)?References[ \t]*:?"
    r"(?:\*\*|__)?[ \t]*(?:#+[ \t]*)?$"
)
_SECTIONS = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*|__)?"
    r"(?:(Perspective[ \t]+\d+\b)[^\n]*|"
    r"(?:Executive[ \t]+summary|Overall[ \t]+synthesis|Final[ \t]+assessment|"
    r"Limitations(?:[ \t]+and[ \t]+uncertainties)?|References)"
    r"[ \t]*:?(?:\*\*|__)?[ \t]*(?:#+[ \t]*)?)$"
)


def is_citation_metadata(line: str) -> bool:
    """Recognise source-ID-only labels, never discard a factual assertion."""
    value = line.strip().strip("*_").strip().rstrip(".")
    if re.match(r"\[(?:Citations?|Sources?)\s*:", value, re.I) and value.endswith("]"):
        value = value[1:-1]
    value = value.replace("**", "").replace("__", "")
    label = re.fullmatch(r"(?:Citations?|Sources?)\s*:\s*(.+)", value, re.I)
    if label is None or not re.search(r"\bS\d+\b", label[1]):
        return False
    remainder = re.sub(r"\bS\d+\b", "", label[1])
    return re.fullmatch(r"[\s,;\[\]]*", remainder) is not None


def prepare_report(text: str) -> str:
    """Discard untrusted bibliographies and canonicalize inline citations."""
    heading = _REFERENCES.search(text)
    narrative = text[:heading.start()] if heading else text
    narrative = "".join(line for line in narrative.splitlines(keepends=True) if not is_citation_metadata(line))
    return normalize_citations(narrative).strip()


def validate_report(text: str, source_ids: set[str], min_required_sources: int = 3) -> list[str]:
    """Check citation coverage and three nonempty, individually cited sections."""
    if type(min_required_sources) is not int or min_required_sources < 1:
        raise ValueError("MIN_REQUIRED_SOURCES must be a positive integer")
    text = prepare_report(text)
    cited = parse_citations(text)
    known = {f"[{source_id}]" for source_id in source_ids}
    valid = set(cited) & known
    issues = []
    invalid = [citation for citation in cited if citation not in known]
    if invalid:
        issues.append("Citations not present in the supplied evidence: " + ", ".join(invalid))
    required = min(min_required_sources, len(source_ids))
    if len(valid) < required:
        issues.append(f"The report needs {required} distinct valid source citations; found {len(valid)}.")
    if len(source_ids) < min_required_sources:
        issues.append(f"Only {len(source_ids)} sources are available; the configured minimum is "
                      f"{min_required_sources}. The report has insufficient evidence for full validation.")

    headings = list(_SECTIONS.finditer(text))
    perspectives: dict[int, list[str]] = {}
    for index, heading in enumerate(headings):
        number = re.fullmatch(r"Perspective\s+(\d+)", heading[1] or "", re.IGNORECASE)
        if number:
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            perspectives.setdefault(int(number[1]), []).append(text[heading.end():end].strip())
    for number in (1, 2, 3):
        sections = perspectives.get(number, [])
        if not sections:
            issues.append(f"Missing Perspective {number} section.")
        elif len(sections) != 1:
            issues.append(f"Perspective {number} appears more than once.")
        else:
            body = sections[0]
            if not re.search(r"\w", CITATION_PATTERN.sub("", body)):
                issues.append(f"Perspective {number} has no substantive text.")
            if not set(parse_citations(body)) & known:
                issues.append(f"Perspective {number} has no valid supporting citation.")
    return issues
