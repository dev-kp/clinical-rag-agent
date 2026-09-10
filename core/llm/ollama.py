import json

import httpx


class OllamaLLMProvider:
    """Real LLM via local Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "phi",
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client or httpx.Client(timeout=60.0)

    def grade(self, question: str, context: str) -> dict:
        """Ask LLM if context sufficiently answers the question."""
        prompt = (
            "You are a medical expert evaluating if provided context adequately "
            "answers a question.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Respond with ONLY a JSON object (no markdown, no backticks):\n"
            '{"verdict": "sufficient" or "need_more", "reason": "explanation"}\n\n'
            "If context has enough info to answer reliably, say 'sufficient'. "
            "Otherwise say 'need_more'."
        )

        response_text = self._call_ollama(prompt)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"verdict": "need_more", "reason": "Could not parse LLM response"}

    def generate(self, question: str, context: str) -> str:
        """Generate an answer from context, citing chunks as [chunk-id]."""
        prompt = f"""You are a medical expert answering a question based ONLY on provided sources.

Question: {question}

Sources (each prefixed with [chunk-id]):
{context}

Rules:
1. Answer ONLY using information in the sources above.
2. Cite your sources as [chunk-id] inline when referencing them.
3. If the sources don't contain enough information, say so explicitly.
4. Be direct and concise."""

        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API and return the response."""
        response = self._client.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
