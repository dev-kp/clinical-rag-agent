import re

from core.graph.state import AgentState
from core.llm.base import LLMProvider
from core.providers.base import EmbeddingProvider
from core.stores.base import VectorStore


def plan_query(state: AgentState) -> AgentState:
    """Refine the question for better retrieval.

    On the first pass, use the question as-is. On retries, narrow the
    query to just the terms the grader said were missing, so the next
    retrieval attempt actually searches for something different instead
    of repeating the same query and finding nothing new.
    """
    if state.iteration == 0:
        state.sub_queries = [state.question]
    elif state.missing_terms:
        state.sub_queries.append(" ".join(state.missing_terms))

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
    Also records what's missing so plan_query can target it next iteration.
    """
    if not state.retrieved:
        state.verdict = "need_more"
        return state

    context = "\n".join([r.text for r in state.retrieved])

    response = llm_provider.grade(question=state.question, context=context)
    state.verdict = response.get("verdict", "need_more")
    state.missing_terms = response.get("missing", [])

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
    """Extract [chunk-id] references from text.

    Real chunk ids look like "doc-id::hash::index" (see ingest/chunk.py's
    _chunk_id), so the pattern must allow colons alongside the simpler
    "chunk-001" style ids used in tests.
    """
    return re.findall(r"\[([a-zA-Z0-9\-:]+)\]", text)
