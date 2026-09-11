from dataclasses import dataclass, field

from core.stores.base import SearchResult


@dataclass
class AgentState:
    """State that flows through the agent graph.

    Tracks the question, all chunks retrieved across iterations, verdicts
    from the grader, and the final answer with citations.
    """

    question: str
    sub_queries: list[str] = field(default_factory=list)
    retrieved: list[SearchResult] = field(default_factory=list)
    seen_chunk_ids: set[str] = field(default_factory=set)
    iteration: int = 0
    max_iterations: int = 3
    verdict: str = ""
    missing_terms: list[str] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
