import httpx

HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
)


class HuggingFaceEmbeddingProvider:
    """Real embeddings via Hugging Face's hosted Inference API.

    Same EmbeddingProvider shape as OllamaEmbeddingProvider, but calls a
    hosted endpoint with an API token instead of a local server. Default
    model is a small, fast sentence-transformer (384 dimensions).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.dimension = dimension
        self._client = client or httpx.Client(timeout=60.0)
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._url = HF_API_URL.format(model=model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            self._url,
            headers=self._headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        embeddings: list[list[float]] = response.json()
        return embeddings
