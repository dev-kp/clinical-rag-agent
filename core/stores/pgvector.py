import psycopg
from pgvector.psycopg import register_vector

from core.retrieval import reciprocal_rank_fusion
from core.stores.base import SearchResult
from ingest.chunk import Chunk


class PgVectorStore:
    """Hybrid search over Postgres + pgvector: cosine similarity for semantic
    matches, full-text search (tsvector/ts_rank) for exact keyword matches,
    fused with Reciprocal Rank Fusion. Azure AI Search will do this fusion
    natively later, this is the hand-rolled version behind the same
    VectorStore interface."""

    def __init__(
        self, dsn: str, dimension: int, table: str = "chunks", connect_timeout: int = 5
    ):
        self.dimension = dimension
        self.table = table
        self._conn = psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                section_path TEXT NOT NULL,
                anchor TEXT NOT NULL,
                source_url TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding VECTOR({self.dimension}) NOT NULL,
                tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            )
            """
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_tsv_idx ON {self.table} USING gin(tsv)"
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        self._conn.cursor().executemany(
            f"""
            INSERT INTO {self.table}
                (chunk_id, doc_id, section_path, anchor, source_url, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                section_path = EXCLUDED.section_path,
                anchor = EXCLUDED.anchor,
                source_url = EXCLUDED.source_url,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding
            """,
            [
                (c.chunk_id, c.doc_id, c.section_path, c.anchor, c.source_url, c.text, e)
                for c, e in zip(chunks, embeddings, strict=True)
            ],
        )

    def search(self, query_text: str, query_embedding: list[float], k: int) -> list[SearchResult]:
        """Hybrid search: cosine similarity + keyword match, fused with RRF."""
        candidate_pool = max(k * 4, 20)

        vector_ranking = self._vector_ranking(query_embedding, candidate_pool)
        keyword_ranking = [
            row[0]
            for row in self._conn.execute(
                f"""
                SELECT chunk_id FROM {self.table}
                WHERE tsv @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC
                LIMIT %s
                """,
                (query_text, query_text, candidate_pool),
            ).fetchall()
        ]

        fused = reciprocal_rank_fusion([vector_ranking, keyword_ranking])[:k]
        if not fused:
            return []

        score_by_id = {f.item_id: f.score for f in fused}
        return self._fetch_results([f.item_id for f in fused], score_by_id)

    def search_vector_only(self, query_embedding: list[float], k: int) -> list[SearchResult]:
        """Pure cosine-similarity search, no keyword fusion.

        Used as the ablation baseline (config 1: single-shot, vector-only)
        to isolate the value hybrid retrieval adds on top of vector search
        alone.
        """
        ranked_ids = self._vector_ranking(query_embedding, k)
        if not ranked_ids:
            return []

        # Distance-based score: closer rank = higher score, same shape as RRF
        # output so downstream code doesn't need to special-case this path.
        score_by_id = {chunk_id: 1.0 / (rank + 1) for rank, chunk_id in enumerate(ranked_ids)}
        return self._fetch_results(ranked_ids, score_by_id)

    def _vector_ranking(self, query_embedding: list[float], limit: int) -> list[str]:
        return [
            row[0]
            for row in self._conn.execute(
                f"SELECT chunk_id FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (query_embedding, limit),
            ).fetchall()
        ]

    def _fetch_results(
        self, chunk_ids: list[str], score_by_id: dict[str, float]
    ) -> list[SearchResult]:
        rows = {
            row[0]: row
            for row in self._conn.execute(
                f"""
                SELECT chunk_id, doc_id, section_path, anchor, source_url, text
                FROM {self.table} WHERE chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            ).fetchall()
        }

        results = []
        for chunk_id in chunk_ids:
            row = rows[chunk_id]
            results.append(
                SearchResult(
                    chunk_id=row[0],
                    doc_id=row[1],
                    section_path=row[2],
                    anchor=row[3],
                    source_url=row[4],
                    text=row[5],
                    score=score_by_id[chunk_id],
                )
            )
        return results
