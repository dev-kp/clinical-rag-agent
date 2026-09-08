from dataclasses import dataclass
from typing import Protocol

from ingest.chunk import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    doc_id: str
    section_path: str
    anchor: str
    source_url: str
    text: str
    score: float


class VectorStore(Protocol):
    """Anything that can store chunks with their embeddings and search over them.
    PgVectorStore (local, hand-rolled hybrid search) and, later, an Azure AI
    Search implementation both satisfy this without sharing a base class."""

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search(
        self, query_text: str, query_embedding: list[float], k: int
    ) -> list[SearchResult]: ...
