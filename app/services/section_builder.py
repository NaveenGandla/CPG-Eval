"""Build structured sections from Document Intelligence output."""

import re
import uuid

import structlog

from app.models.requests import ReportSection
from app.utils.keyword_extraction import extract_keywords

logger = structlog.get_logger()

# Patterns for heading detection
_NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z\s\-:,&]{4,}$")
_SHORT_LINE_MAX_CHARS = 120


def build_sections(layout_output: dict) -> list[ReportSection]:
    """Build structured sections from Document Intelligence layout output.

    Args:
        layout_output: Dict with 'paragraphs' (list of {content, role}) and 'tables' (list of str).

    Returns:
        List of normalized ReportSection objects.
    """
    paragraphs = layout_output.get("paragraphs", [])
    tables = layout_output.get("tables", [])

    if not paragraphs and not tables:
        logger.warning("section_builder_empty_input")
        return []

    # Merge table content into paragraphs at the end (they'll get grouped with preceding section)
    all_items = list(paragraphs)
    for table_text in tables:
        all_items.append({"content": table_text, "role": "table"})

    # Phase 1: Identify headings and group content
    raw_sections = _split_into_sections(all_items)

    # Phase 2: Normalize into ReportSection objects
    sections = []
    for i, (title, content_parts) in enumerate(raw_sections):
        content = "\n\n".join(content_parts).strip()
        if not content:
            continue

        cleaned_title = _clean_title(title)
        section_type = infer_section_type(cleaned_title)
        keywords = extract_keywords(content, top_n=10)

        sections.append(
            ReportSection(
                id=str(uuid.uuid4()),
                title=cleaned_title,
                content=content,
                section_type=section_type,
                order=i,
                keywords=keywords,
            )
        )

    logger.info("section_builder_done", num_sections=len(sections))
    return sections


def _split_into_sections(
    items: list[dict],
) -> list[tuple[str, list[str]]]:
    """Split items into (title, content_parts) tuples based on heading detection.

    Uses Document Intelligence role metadata when available, with heuristic fallbacks.
    """
    sections: list[tuple[str, list[str]]] = []
    current_title = "Untitled"
    current_content: list[str] = []

    for item in items:
        content = item.get("content", "").strip()
        role = item.get("role")

        if not content:
            continue

        is_heading = _is_heading(content, role)

        if is_heading:
            # Save previous section if it has content
            if current_content:
                sections.append((current_title, current_content))
            current_title = content
            current_content = []
        else:
            current_content.append(content)

    # Don't forget the last section
    if current_content:
        sections.append((current_title, current_content))

    # Fallback: if no headings detected, create pseudo-sections by chunking
    if len(sections) <= 1 and sections:
        title, all_content = sections[0]
        if len(all_content) > 5:
            sections = _chunk_into_pseudo_sections(all_content)

    return sections


def _is_heading(text: str, role: str | None) -> bool:
    """Determine if a text item is a heading."""
    # Trust Document Intelligence role metadata first
    if role in ("title", "sectionHeading"):
        return True

    # Heuristic: numbered header (1., 1.1, etc.)
    if _NUMBERED_HEADING_RE.match(text):
        return True

    # Heuristic: ALL CAPS line (short enough to be a title)
    stripped = text.strip()
    if _ALL_CAPS_RE.match(stripped) and len(stripped) <= _SHORT_LINE_MAX_CHARS:
        return True

    # Heuristic: short line (likely a title) — only if very short
    if len(stripped) <= 60 and not stripped.endswith(".") and "\n" not in stripped:
        # Avoid false positives: must look like a title (starts with uppercase, no period)
        if stripped and stripped[0].isupper():
            return True

    return False


def _chunk_into_pseudo_sections(
    content_parts: list[str], chunk_size: int = 5
) -> list[tuple[str, list[str]]]:
    """Create pseudo-sections by grouping paragraphs when no headings found."""
    sections = []
    for i in range(0, len(content_parts), chunk_size):
        chunk = content_parts[i : i + chunk_size]
        title = f"Section {i // chunk_size + 1}"
        sections.append((title, chunk))
    return sections


def _clean_title(title: str) -> str:
    """Clean and normalize a section title."""
    # Remove leading numbers like "1.", "1.1", "1.1.1"
    title = re.sub(r"^\d+(\.\d+)*\.?\s*", "", title).strip()
    # Title case if all caps
    if title.isupper():
        title = title.title()
    return title if title else "Untitled"


def infer_section_type(title: str) -> str:
    """Infer section type from title using keyword rules (no LLM).

    Returns one of: definitions, abbreviations, guideline, general.
    """
    lower = title.lower()

    if "definition" in lower:
        return "definitions"
    if "abbreviation" in lower or "acronym" in lower or "glossary" in lower:
        return "abbreviations"
    if any(
        kw in lower
        for kw in [
            "guideline",
            "recommendation",
            "treatment",
            "therapy",
            "diagnosis",
            "management",
            "clinical",
            "protocol",
            "dosing",
            "dosage",
            "regimen",
            "safety",
            "adverse",
            "efficacy",
            "outcome",
        ]
    ):
        return "guideline"

    return "general"
