from typing import Any, Protocol

from langgraph.graph import END, StateGraph

from core.graph.nodes import (
    generate,
    grade_evidence,
    plan_query,
    retrieve,
    verify_citations,
)
from core.graph.state import AgentState
from core.stores.base import VectorStore
from core.providers.base import EmbeddingProvider
from core.llm.base import LLMProvider



def build_graph(
    llm_provider: LLMProvider,
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
) -> Any:
    """Build the agent graph.

    Nodes: plan_query -> retrieve -> grade_evidence -> (generate | retrieve | abstain)
    -> verify_citations -> end
    """
    graph = StateGraph(AgentState)

    graph.add_node("plan_query", lambda state: plan_query(state))
    graph.add_node(
        "retrieve",
        lambda state: retrieve(state, vector_store, embedding_provider),
    )
    graph.add_node("grade_evidence", lambda state: grade_evidence(state, llm_provider))
    graph.add_node("generate", lambda state: generate(state, llm_provider))
    graph.add_node("verify_citations", lambda state: verify_citations(state))

    graph.add_edge("plan_query", "retrieve")
    graph.add_edge("retrieve", "grade_evidence")

    def route_after_grade(state: AgentState) -> str:
        if state.verdict == "sufficient":
            return "generate"
        if state.iteration < state.max_iterations:
            return "retrieve"
        return "abstain"

    graph.add_conditional_edges(
        "grade_evidence",
        route_after_grade,
        {"generate": "generate", "retrieve": "retrieve", "abstain": END},
    )

    graph.add_edge("generate", "verify_citations")
    graph.add_edge("verify_citations", END)

    return graph.compile()
