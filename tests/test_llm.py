import httpx

from core.llm.fake import FakeLLMProvider
from core.llm.groq import GroqLLMProvider, _parse_retry_wait


def test_parse_retry_wait_seconds_only():
    response = httpx.Response(429, headers={"x-ratelimit-reset-tokens": "25.77s"})
    assert _parse_retry_wait(response) == 26.27


def test_parse_retry_wait_minutes_and_seconds():
    response = httpx.Response(429, headers={"x-ratelimit-reset-tokens": "4m19.2s"})
    assert _parse_retry_wait(response) == 259.7


def test_parse_retry_wait_missing_header_falls_back():
    response = httpx.Response(429, headers={})
    assert _parse_retry_wait(response) == 5.0


def test_parse_retry_wait_malformed_header_falls_back():
    response = httpx.Response(429, headers={"x-ratelimit-reset-tokens": "garbage"})
    assert _parse_retry_wait(response) == 5.0


def test_groq_provider_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"x-ratelimit-reset-tokens": "1s"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "the answer"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GroqLLMProvider(api_key="fake-key", client=client)

    result = provider.generate(question="q", context="c")

    assert result == "the answer"
    assert call_count == 2


def test_groq_provider_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"x-ratelimit-reset-tokens": "1s"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GroqLLMProvider(api_key="fake-key", client=client)

    raised = False
    try:
        provider.generate(question="q", context="c")
    except RuntimeError as e:
        raised = "rate limit not resolved" in str(e)
    assert raised


def test_fake_llm_grade_sufficient_when_keywords_present():
    provider = FakeLLMProvider()
    context = "syphilis is treated with penicillin"
    result = provider.grade(question="syphilis treatment", context=context)
    assert result["verdict"] == "sufficient"


def test_fake_llm_grade_need_more_when_keywords_missing():
    provider = FakeLLMProvider()
    context = "syphilis is treated with penicillin"
    result = provider.grade(question="asthma inhaler dosage", context=context)
    assert result["verdict"] == "need_more"
    assert set(result["missing"]) == {"asthma", "inhaler", "dosage"}
