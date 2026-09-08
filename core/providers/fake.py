import hashlib


class FakeEmbeddingProvider:
    """Deterministic, network-free stand-in for a real embedding model. Same text
    always produces the same vector, different text (almost certainly) produces
    a different one, good enough to exercise vector-store logic in tests without
    paying for or depending on a real embedding API."""

    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dimension)]
