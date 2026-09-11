import os

import psycopg
import pytest

from core.providers.fake import FakeEmbeddingProvider
from core.stores.pgvector import PgVectorStore
from ingest.chunk import Chunk

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/clinical_rag_test"
)


@pytest.fixture
def pg_store():
    try:
        store = PgVectorStore(TEST_DSN, dimension=16, table="chunks_test")
    except psycopg.OperationalError as e:
        pytest.skip(f"no test Postgres available at {TEST_DSN}: {e}")

    store._conn.execute("TRUNCATE chunks_test")
    yield store
    store.close()


def make_chunk(chunk_id: str, text: str, section_path: str = "Test > Section") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="test-doc",
        section_path=section_path,
        anchor="test-anchor",
        source_url="https://example.com/test",
        chunk_index=0,
        text=text,
        token_count=10,
    )


def test_upsert_and_search_by_vector_similarity(pg_store):
    provider = FakeEmbeddingProvider(dimension=16)

    chunks = [
        make_chunk("chunk-1", "The primary treatment for syphilis is penicillin"),
        make_chunk("chunk-2", "Dosing for syphilis in adults is 2.4 million units IM"),
        make_chunk("chunk-3", "COVID-19 vaccines are widely available"),
    ]
    embeddings = provider.embed([c.text for c in chunks])

    pg_store.upsert(chunks, embeddings)

    query_embedding = provider.embed(["syphilis treatment"])[0]
    results = pg_store.search("syphilis treatment", query_embedding, k=2)

    assert len(results) <= 2
    assert any("syphilis" in r.text.lower() for r in results)


def test_search_by_keyword_match(pg_store):
    provider = FakeEmbeddingProvider(dimension=16)

    chunks = [
        make_chunk("chunk-1", "Penicillin G is the preferred treatment"),
        make_chunk("chunk-2", "Dosing recommendations for various infections"),
    ]
    embeddings = provider.embed([c.text for c in chunks])

    pg_store.upsert(chunks, embeddings)

    query_embedding = provider.embed(["dosing"])[0]
    results = pg_store.search("dosing", query_embedding, k=2)

    assert any("dosing" in r.text.lower() for r in results)


def test_search_returns_all_fields_intact(pg_store):
    provider = FakeEmbeddingProvider(dimension=16)

    chunk = make_chunk(
        "chunk-1",
        "Test content",
        section_path="Doc > Section > Subsection"
    )
    embeddings = provider.embed([chunk.text])

    pg_store.upsert([chunk], embeddings)

    query_embedding = provider.embed(["test"])[0]
    results = pg_store.search("test", query_embedding, k=1)

    assert len(results) == 1
    r = results[0]
    assert r.chunk_id == "chunk-1"
    assert r.doc_id == "test-doc"
    assert r.section_path == "Doc > Section > Subsection"
    assert r.anchor == "test-anchor"
    assert r.source_url == "https://example.com/test"
    assert r.text == "Test content"
    assert r.score > 0


def test_search_against_empty_table_returns_empty_list(pg_store):
    provider = FakeEmbeddingProvider(dimension=16)

    query_embedding = provider.embed(["syphilis treatment"])[0]
    results = pg_store.search("syphilis treatment", query_embedding, k=1)

    assert results == []
