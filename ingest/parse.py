import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}


@dataclass(frozen=True)
class Section:
    doc_id: str
    section_path: str
    heading_level: int
    anchor: str
    text: str
    source_url: str


def parse_html(html: str, doc_id: str, source_url: str, title: str) -> list[Section]:
    """Split an HTML document into sections along its own heading structure
    (h1-h4). Text between one heading and the next becomes that heading's
    section body; section_path is the heading breadcrumb from the document
    title down to that heading."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("main") or soup.find("article") or soup.body or soup

    sections: list[Section] = []
    stack: list[str] = [title]
    current_level = 0
    current_anchor = _slugify(title)
    buffer: list[str] = []

    def flush() -> None:
        text = _clean_text(" ".join(buffer))
        if text:
            sections.append(
                Section(
                    doc_id=doc_id,
                    section_path=" > ".join(stack),
                    heading_level=current_level,
                    anchor=current_anchor,
                    text=text,
                    source_url=source_url,
                )
            )
        buffer.clear()

    for node in body.descendants:
        if isinstance(node, Tag) and node.name in HEADING_TAGS:
            flush()
            level = HEADING_TAGS[node.name]
            heading_text = _clean_text(node.get_text(" "))
            if not heading_text:
                continue
            stack = _rebuild_stack(stack, title, heading_text, level)
            current_level = level
            heading_id = node.get("id")
            current_anchor = (
                heading_id if isinstance(heading_id, str) and heading_id else _slugify(heading_text)
            )
        elif isinstance(node, NavigableString):
            parent = node.parent
            if parent is not None and parent.name in ("script", "style", "nav", "footer"):
                continue
            if node.find_parent(list(HEADING_TAGS)) is not None:
                continue  # text inside a heading tag was already captured as heading_text
            text = str(node).strip()
            if text:
                buffer.append(text)

    flush()
    return sections


def _rebuild_stack(stack: list[str], title: str, heading_text: str, level: int) -> list[str]:
    # Maintain a breadcrumb of at most `level` headings under the title, trimming
    # deeper levels when we go back up and replacing the entry at this level.
    padded = stack if len(stack) >= level else stack + [heading_text] * (level - len(stack))
    trimmed = padded[: level - 1] + [heading_text]
    if trimmed[0] != title:
        trimmed[0] = title
    return trimmed


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"
