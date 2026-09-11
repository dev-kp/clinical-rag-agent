import json

import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


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
            "2. Cite your sources as [chunk-id] inline when referencing them.\n"
            "3. If the sources don't contain enough information, say so explicitly.\n"
            "4. Be direct and concise."
        )

        return self._call_groq(prompt)

    def _call_groq(self, prompt: str) -> str:
        """Call Groq's chat completions API and return the response text."""
        response = self._client.post(
            GROQ_API_URL,
            headers=self._headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        data = response.json()
        content: str = data["choices"][0]["message"]["content"]
        return content.strip()
