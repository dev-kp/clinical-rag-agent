from core.providers.base import EmbeddingProvider
from core.stores.base import VectorStore
from ingest.chunk import Chunk


def embed_and_store(
    chunks: list[Chunk],
    provider: EmbeddingProvider,
    store: VectorStore,
) -> None:
    """Embed chunks and store them in the vector store.

    Args:
        chunks: List of chunks to embed and store.
        provider: EmbeddingProvider to use for embedding.
        store: VectorStore to store embeddings in.
    """
    if not chunks:
        return

    texts = [c.text for c in chunks]
    embeddings = provider.embed(texts)

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"expected {len(chunks)} embeddings, got {len(embeddings)}"
        )

    store.upsert(chunks, embeddings)
