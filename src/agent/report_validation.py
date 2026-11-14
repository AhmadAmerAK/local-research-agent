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


def prepare_report(text: str) -> str:
    """Discard untrusted bibliographies and canonicalize inline citations."""
    heading = _REFERENCES.search(text)
    return normalize_citations(text[:heading.start()] if heading else text).strip()


def validate_report(text: str, source_ids: set[str]) -> list[str]:
    """Check citation coverage and three nonempty, individually cited sections."""
    text = prepare_report(text)
    cited = parse_citations(text)
    known = {f"[{source_id}]" for source_id in source_ids}
    valid = set(cited) & known
    issues = []
    invalid = [citation for citation in cited if citation not in known]
    if invalid:
        issues.append("Citations not present in the supplied evidence: " + ", ".join(invalid))
    required = min(3, len(source_ids))
    if len(valid) < required:
        issues.append(f"The report needs {required} distinct valid source citations; found {len(valid)}.")
    if len(source_ids) < 3:
        issues.append("Fewer than three sources are available; the report has insufficient evidence for full validation.")

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
