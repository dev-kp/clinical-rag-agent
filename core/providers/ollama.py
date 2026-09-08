import httpx


class OllamaEmbeddingProvider:
    """Real embeddings from a local Ollama server. The HTTP client is injected
    (same pattern as ingest/fetch.py) so tests can substitute a MockTransport
    instead of needing Ollama actually running."""

    def __init__(
        self,
        base_url: str,
        model: str = "nomic-embed-text",
        dimension: int = 768,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimension = dimension
        self._client = client or httpx.Client(timeout=60.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        response = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        embedding: list[float] = response.json()["embedding"]
        return embedding
