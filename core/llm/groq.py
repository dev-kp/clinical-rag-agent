import json
import time

import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 5


class GroqLLMProvider:
    """Real LLM via Groq's hosted, OpenAI-compatible chat API.

    Used as the free, no-local-install real LLM before an Azure account
    exists. Same LLMProvider shape as OllamaLLMProvider, so the agent
    graph doesn't care which one it's given.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        client: httpx.Client | None = None,
    ):
        self.model = model
        self._client = client or httpx.Client(timeout=60.0)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def grade(self, question: str, context: str) -> dict:
        """Ask LLM if context sufficiently answers the question."""
        prompt = (
            "You are a medical expert evaluating if provided context adequately "
            "answers a question.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Respond with ONLY a JSON object (no markdown, no backticks):\n"
            '{"verdict": "sufficient" or "need_more", '
            '"missing": ["term1", "term2"], "reason": "explanation"}\n\n'
            "If context has enough info to answer reliably, say 'sufficient' "
            "and leave missing empty. Otherwise say 'need_more' and list the "
            "key terms from the question that the context does not cover."
        )

        response_text = self._call_groq(prompt)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"verdict": "need_more", "reason": "Could not parse LLM response"}

    def generate(self, question: str, context: str) -> str:
        """Generate an answer from context, citing chunks as [chunk-id]."""
        prompt = (
            "You are a medical expert answering a question based ONLY on "
            "provided sources.\n\n"
            f"Question: {question}\n\n"
            f"Sources (each prefixed with [chunk-id]):\n{context}\n\n"
            "Rules:\n"
            "1. Answer ONLY using information in the sources above.\n"
            "2. Cite your sources inline by copying the chunk-id EXACTLY as it "
            "appears in the brackets above, character for character, including "
            "every ':' and hyphen. Do not shorten, reformat, or paraphrase it. "
            "For example if a source is prefixed [abc::def123::0], cite it as "
            "[abc::def123::0], not [abc::def123].\n"
            "3. If the sources don't contain enough information, say so explicitly.\n"
            "4. Be direct and concise."
        )

        return self._call_groq(prompt)

    def _call_groq(self, prompt: str) -> str:
        """Call Groq's chat completions API and return the response text.

        Retries on 429 (rate limit), waiting exactly as long as Groq's
        x-ratelimit-reset-tokens header says is needed rather than guessing
        with blind exponential backoff.
        """
        for attempt in range(MAX_RETRIES):
            response = self._client.post(
                GROQ_API_URL,
                headers=self._headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
            )
            if response.status_code != 429:
                response.raise_for_status()
                data = response.json()
                content: str = data["choices"][0]["message"]["content"]
                return content.strip()

            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Groq rate limit not resolved after {MAX_RETRIES} attempts"
                )

            time.sleep(_parse_retry_wait(response))

        raise AssertionError("unreachable")


def _parse_retry_wait(response: httpx.Response) -> float:
    """Extract how long to wait from Groq's rate limit headers.

    x-ratelimit-reset-tokens looks like "25.77s" or "4m19.2s". Falls back
    to a flat 5 seconds if the header is missing or in an unexpected shape.
    """
    reset = response.headers.get("x-ratelimit-reset-tokens", "")
    minutes = 0.0
    seconds = 0.0
    try:
        if "m" in reset:
            minutes_part, seconds_part = reset.split("m")
            minutes = float(minutes_part)
            seconds = float(seconds_part.rstrip("s"))
        elif reset:
            seconds = float(reset.rstrip("s"))
        else:
            return 5.0
    except ValueError:
        return 5.0

    return minutes * 60 + seconds + 0.5
