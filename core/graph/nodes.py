import re
from typing import Protocol

from core.graph.state import AgentState
from core.providers.base import EmbeddingProvider
from core.stores.base import VectorStore
from core.llm.base import LLMProvider

def plan_query(state: AgentState) -> AgentState:
    """Optionally refine the question for better retrieval.

    On first iteration, the original question is used.
    On subsequent iterations, the grader may have specified what's missing.
    """
    if state.iteration == 0:
        state.sub_queries = [state.question]
    return state


def retrieve(
    state: AgentState, vector_store: VectorStore, provider: EmbeddingProvider
) -> AgentState:
    """Search for chunks matching the current query.

    Deduplicates against already-retrieved chunks to avoid wasting tokens.
    """
    query = state.sub_queries[-1] if state.sub_queries else state.question

    query_embedding = provider.embed([query])[0]
    results = vector_store.search(query, query_embedding, k=5)

    new_results = [r for r in results if r.chunk_id not in state.seen_chunk_ids]

    state.retrieved.extend(new_results)
    state.seen_chunk_ids.update(r.chunk_id for r in new_results)
    state.iteration += 1

    return state


def grade_evidence(state: AgentState, llm_provider: LLMProvider) -> AgentState:
    """Ask LLM: do we have enough chunks to answer?

    Returns a verdict: "sufficient" (answer now) or "need_more" (search again).
    """
    if not state.retrieved:
        state.verdict = "need_more"
        return state

    context = "\n".join([r.text for r in state.retrieved])

    response = llm_provider.grade(question=state.question, context=context)
    state.verdict = response.get("verdict", "need_more")

    return state


def generate(state: AgentState, llm_provider: LLMProvider) -> AgentState:
    """Generate answer from retrieved chunks, with inline citations.

    Answer must cite only chunks in state.retrieved.
    """
    context = "\n".join(
        [f"[{r.chunk_id}] {r.text}" for r in state.retrieved]
    )

    state.answer = llm_provider.generate(question=state.question, context=context)

    state.citations = extract_citations(state.answer)

    return state


def verify_citations(state: AgentState) -> AgentState:
    """Verify that cited chunk ids are real (programmatic check only).

    If any citation is not in retrieved chunks, mark as hallucination.
    """
    for citation_id in state.citations:
        if citation_id not in state.seen_chunk_ids:
            state.answer = "I cannot provide a reliable answer based on available sources."
            state.citations = []
            return state

    return state


def extract_citations(text: str) -> list[str]:
    """Extract [chunk-id] references from text."""
    return re.findall(r"\[([a-z0-9\-]+)\]", text)
