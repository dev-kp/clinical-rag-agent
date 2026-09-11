"""Run the full ingestion pipeline against the real corpus in SOURCES.md:
fetch -> parse -> chunk -> embed -> upsert into the vector store.

Usage: python -m ingest.run
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from core.providers.fake import FakeEmbeddingProvider
from core.providers.huggingface import HuggingFaceEmbeddingProvider
from core.stores.pgvector import PgVectorStore
from ingest.chunk import Chunk, chunk_sections
from ingest.embed import embed_and_store
from ingest.fetch import fetch_all
from ingest.parse import parse_html
from ingest.sources import MANIFEST

load_dotenv()

RAW_DIR = Path("ingest/raw")
DSN = "postgresql://postgres:postgres@localhost:5433/clinical_rag_test"


def get_embedding_provider():
    hf_api_key = os.environ.get("HF_API_KEY")
    if hf_api_key:
        hf_model = os.environ.get(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        print(f"Using HuggingFaceEmbeddingProvider (model={hf_model})")
        return HuggingFaceEmbeddingProvider(api_key=hf_api_key, model=hf_model)

    print("HF_API_KEY not set, using FakeEmbeddingProvider")
    return FakeEmbeddingProvider(dimension=16)


def main() -> None:
    print(f"Fetching {len(MANIFEST)} documents...")
    with httpx.Client(timeout=30.0) as client:
        records = fetch_all(RAW_DIR, client)
    print(f"Fetched/cached {len(records)} documents into {RAW_DIR}/")

    all_chunks: list[Chunk] = []
    for doc, record in zip(MANIFEST, records, strict=True):
        html = Path(record.path).read_text(encoding="utf-8")
        sections = parse_html(html, doc_id=doc.doc_id, source_url=doc.url, title=doc.title)
        chunks = chunk_sections(sections)
        all_chunks.extend(chunks)
        print(f"  {doc.doc_id}: {len(sections)} sections -> {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    provider = get_embedding_provider()
    store = PgVectorStore(DSN, dimension=provider.dimension)

    print(f"\nEmbedding and storing {len(all_chunks)} chunks (dimension={provider.dimension})...")
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embed_and_store(batch, provider, store)
        print(f"  stored {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

    store.close()
    print("\nDone.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    main()
