from core.graph.build import build_graph
from core.graph.nodes import extract_citations
from core.graph.state import AgentState
from core.llm.fake import FakeLLMProvider
from core.providers.fake import FakeEmbeddingProvider
from core.stores.base import SearchResult


class InMemoryVectorStore:
    """Fake VectorStore for graph tests: returns pre-seeded results for any
    query, no Postgres needed. Satisfies the VectorStore protocol shape."""

    def __init__(self, results: list[SearchResult]):
        self._results = results

    def upsert(self, chunks, embeddings) -> None:
        pass

    def search(self, query_text: str, query_embedding: list[float], k: int) -> list[SearchResult]:
        return self._results[:k]


def make_result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id="test-doc",
        section_path="Test > Section",
        anchor="test-anchor",
        source_url="https://example.com/test",
        text=text,
        score=1.0,
    )


def build_test_graph(vector_store: InMemoryVectorStore):
    return build_graph(
        llm_provider=FakeLLMProvider(),
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(dimension=8),
    )


def test_sufficient_evidence_answers_on_first_iteration():
    store = InMemoryVectorStore([make_result("chunk-1", "syphilis is treated with penicillin")])
    graph = build_test_graph(store)

    final_state = graph.invoke(AgentState(question="syphilis"))

    assert final_state["iteration"] == 1
    assert final_state["verdict"] == "sufficient"
    assert final_state["answer"] != ""
    assert "chunk-1" in final_state["citations"]


def test_loop_terminates_at_max_iterations_on_starved_corpus():
    # Corpus has nothing matching "asthma", so grading always says need_more.
    store = InMemoryVectorStore([make_result("chunk-1", "syphilis is treated with penicillin")])
    graph = build_test_graph(store)

    final_state = graph.invoke(AgentState(question="asthma inhaler dosage"))

    assert final_state["iteration"] == 3
    assert final_state["verdict"] == "need_more"


def test_abstains_on_starved_corpus_instead_of_answering():
    store = InMemoryVectorStore([make_result("chunk-1", "syphilis is treated with penicillin")])
    graph = build_test_graph(store)

    final_state = graph.invoke(AgentState(question="asthma inhaler dosage"))

    assert final_state["answer"] == ""
    assert final_state["citations"] == []


def test_dedupe_prevents_reretrieving_same_chunk():
    # Only one chunk exists; even after retries, retrieved should never
    # exceed the store's total content because the same id is filtered out.
    store = InMemoryVectorStore([make_result("chunk-1", "syphilis is treated with penicillin")])
    graph = build_test_graph(store)

    final_state = graph.invoke(AgentState(question="asthma inhaler dosage"))

    assert len(final_state["retrieved"]) == 1
    assert final_state["seen_chunk_ids"] == {"chunk-1"}


def test_empty_corpus_abstains_without_crashing():
    store = InMemoryVectorStore([])
    graph = build_test_graph(store)

    final_state = graph.invoke(AgentState(question="anything"))

    assert final_state["verdict"] == "need_more"
    assert final_state["answer"] == ""


def test_hallucinated_citation_is_caught_by_verify_citations():
    # generate() only cites chunks present in context, so to exercise the
    # hallucination guard directly we call verify_citations with a citation
    # that was never retrieved.
    from core.graph.nodes import verify_citations

    state = AgentState(question="syphilis")
    state.retrieved = [make_result("chunk-1", "syphilis is treated with penicillin")]
    state.seen_chunk_ids = {"chunk-1"}
    state.citations = ["chunk-999"]
    state.answer = "The answer is [chunk-999]."

    result = verify_citations(state)

    assert result.citations == []
    assert "cannot provide a reliable answer" in result.answer


def test_extract_citations_handles_real_chunk_id_format():
    # Real chunk ids from ingest/chunk.py's _chunk_id look like
    # "doc-id::hash::index", not the simple "chunk-001" style used in
    # most tests. The extraction regex must allow colons.
    text = (
        "Doxycycline is an alternative "
        "[sti-syphilis-primary-secondary::3836bbeb70::0] and ceftriaxone "
        "may also be used [sti-syphilis-neuro::a200e4c04d::0]."
    )

    citations = extract_citations(text)

    assert citations == [
        "sti-syphilis-primary-secondary::3836bbeb70::0",
        "sti-syphilis-neuro::a200e4c04d::0",
    ]


def test_extract_citations_normalizes_unicode_dash_variants():
    # Observed live: Groq's model sometimes renders the ASCII hyphen in a
    # doc-id as a Unicode non-breaking hyphen (U+2011) when formatting an
    # answer, which would otherwise silently fail to match seen_chunk_ids.
    text = "See [sti‑chlamydia::e7f6b57289::0] for the regimen."

    citations = extract_citations(text)

    assert citations == ["sti-chlamydia::e7f6b57289::0"]
