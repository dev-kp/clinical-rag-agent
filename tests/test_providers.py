import httpx

from core.providers.fake import FakeEmbeddingProvider
from core.providers.huggingface import HuggingFaceEmbeddingProvider
from core.providers.ollama import OllamaEmbeddingProvider


def test_fake_provider_is_deterministic():
    provider = FakeEmbeddingProvider(dimension=8)
    assert provider.embed(["hello"]) == provider.embed(["hello"])


def test_fake_provider_differs_for_different_text():
    provider = FakeEmbeddingProvider(dimension=8)
    a, b = provider.embed(["hello", "goodbye"])
    assert a != b


def test_fake_provider_respects_dimension():
    provider = FakeEmbeddingProvider(dimension=8)
    [vector] = provider.embed(["anything"])
    assert len(vector) == 8


def test_ollama_provider_calls_expected_endpoint_and_parses_embedding():
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", client=client)

    [vector] = provider.embed(["syphilis treatment"])

    assert vector == [0.1, 0.2, 0.3]
    assert seen_requests[0].url.path == "/api/embeddings"


def test_ollama_provider_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", client=client)

    raised = False
    try:
        provider.embed(["text"])
    except httpx.HTTPStatusError:
        raised = True
    assert raised


def test_huggingface_provider_calls_expected_endpoint_and_parses_embeddings():
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json=[[0.1, 0.2], [0.3, 0.4]])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = HuggingFaceEmbeddingProvider(api_key="fake-key", client=client)

    vectors = provider.embed(["syphilis treatment", "gonorrhea treatment"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert "all-MiniLM-L6-v2" in seen_requests[0].url.path
    assert seen_requests[0].headers["authorization"] == "Bearer fake-key"


def test_huggingface_provider_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "model loading"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = HuggingFaceEmbeddingProvider(api_key="fake-key", client=client)

    raised = False
    try:
        provider.embed(["text"])
    except httpx.HTTPStatusError:
        raised = True
    assert raised
