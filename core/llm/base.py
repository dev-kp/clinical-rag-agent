from typing import Protocol


class LLMProvider(Protocol):
    """Interface for LLM providers used in the agent graph.

    Implementations: FakeLLMProvider (testing), OllamaLLMProvider (real).
    """

    def grade(self, question: str, context: str) -> dict:
        """Grade whether context sufficiently answers the question.

        Args:
            question: The user's question.
            context: Concatenated text of all retrieved chunks.

        Returns:
            {"verdict": "sufficient" | "need_more", "reason": "..."}
        """
        ...

    def generate(self, question: str, context: str) -> str:
        """Generate an answer citing specific chunks.

        Answer must cite chunks as [chunk-id]. Only cite chunks actually
        in the context.

        Args:
            question: The user's question.
            context: Text prefixed with [chunk-id] markers.

        Returns:
            Answer string with inline [chunk-id] citations.
        """
        ...
