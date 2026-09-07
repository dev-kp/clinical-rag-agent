import hashlib
from dataclasses import dataclass

import tiktoken

from ingest.parse import Section

DEFAULT_MAX_TOKENS = 700
DEFAULT_OVERLAP_TOKENS = 120
ENCODING_NAME = "cl100k_base"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    section_path: str
    anchor: str
    source_url: str
    chunk_index: int
    text: str
    token_count: int


def chunk_sections(
    sections: list[Section],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split each section into one or more chunks of at most max_tokens tokens,
    with overlap_tokens of overlap between consecutive windows of an oversized
    section. Every chunk is prefixed with its section's breadcrumb path so an
    isolated chunk still carries its own context. Pure and deterministic: the
    same sections always produce the same chunks."""
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    chunks: list[Chunk] = []

    for section in sections:
        body_tokens = encoding.encode(section.text)
        windows = _windows(body_tokens, max_tokens, overlap_tokens, section.section_path, encoding)

        for index, body_window in enumerate(windows):
            text = f"{section.section_path}\n\n{encoding.decode(body_window)}"
            token_count = len(encoding.encode(text))
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(section.doc_id, section.section_path, index),
                    doc_id=section.doc_id,
                    section_path=section.section_path,
                    anchor=section.anchor,
                    source_url=section.source_url,
                    chunk_index=index,
                    text=text,
                    token_count=token_count,
                )
            )

    return chunks


def _windows(
    body_tokens: list[int],
    max_tokens: int,
    overlap_tokens: int,
    section_path: str,
    encoding: tiktoken.Encoding,
) -> list[list[int]]:
    prefix_tokens = len(encoding.encode(f"{section_path}\n\n"))
    budget = max(max_tokens - prefix_tokens, 1)

    if len(body_tokens) <= budget:
        return [body_tokens]

    step = max(budget - overlap_tokens, 1)
    windows: list[list[int]] = []
    start = 0
    while start < len(body_tokens):
        windows.append(body_tokens[start : start + budget])
        if start + budget >= len(body_tokens):
            break
        start += step
    return windows


def _chunk_id(doc_id: str, section_path: str, index: int) -> str:
    digest = hashlib.sha1(f"{doc_id}|{section_path}".encode()).hexdigest()[:10]
    return f"{doc_id}::{digest}::{index}"
