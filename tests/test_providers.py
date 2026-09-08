import httpx

from core.providers.fake import FakeEmbeddingProvider
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
