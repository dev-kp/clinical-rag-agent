class FakeLLMProvider:
    """Deterministic LLM for testing. No network, no cost, reproducible.

    Grades based on keyword matching: if context contains key terms from
    the question, it's sufficient. Otherwise, need_more.
    """

    def grade(self, question: str, context: str) -> dict:
        """Grade: sufficient if context has key terms from question."""
        question_lower = question.lower()
        context_lower = context.lower()

        key_terms = [w for w in question_lower.split() if len(w) > 3]

        matches = sum(1 for term in key_terms if term in context_lower)
        match_ratio = matches / len(key_terms) if key_terms else 0

        if match_ratio >= 0.5:
            return {"verdict": "sufficient", "reason": "Key terms found"}
        return {"verdict": "need_more", "reason": "Insufficient context"}

    def generate(self, question: str, context: str) -> str:
        """Generate a fake answer citing the first chunk."""
        lines = context.split("\n")
        chunk_lines = [line for line in lines if line.startswith("[chunk-")]

        if not chunk_lines:
            return "No relevant information found."

        first_chunk = chunk_lines[0].split("]")[0] + "]"

        return f"Based on the available information, the answer is found in {first_chunk}."
