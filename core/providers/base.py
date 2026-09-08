from typing import Protocol


class EmbeddingProvider(Protocol):
    """Anything that turns text into vectors. FakeEmbeddingProvider (no network,
    used in tests) and OllamaEmbeddingProvider (real, local model) both satisfy
    this without inheriting from it, structural typing, not nominal typing."""

    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
