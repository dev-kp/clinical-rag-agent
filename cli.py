import sys

from core.graph.build import build_graph
from core.graph.state import AgentState
from core.llm.fake import FakeLLMProvider
from core.providers.fake import FakeEmbeddingProvider
from core.stores.pgvector import PgVectorStore
from ingest.chunk import Chunk
from ingest.embed import embed_and_store


def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py [populate|ask|agent] <query>")
        print("Example: python cli.py populate")
        print("Example: python cli.py ask 'syphilis treatment'")
        print("Example: python cli.py agent 'syphilis treatment in pregnancy'")
        sys.exit(1)

    command = sys.argv[1]

    if command == "populate":
        populate_store()
    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: python cli.py ask <query>")
            sys.exit(1)

        query = " ".join(sys.argv[2:])
        ask_query(query)
    elif command == "agent":
        if len(sys.argv) < 3:
            print("Usage: python cli.py agent <query>")
            sys.exit(1)

        query = " ".join(sys.argv[2:])
        run_agent(query)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


def populate_store() -> None:
    """Populate the vector store with test chunks."""
    dsn = "postgresql://postgres:postgres@localhost:5433/clinical_rag_test"
    dimension = 16

    provider = FakeEmbeddingProvider(dimension=dimension)
    store = PgVectorStore(dsn, dimension=dimension)

    test_chunks = [
        Chunk(
            chunk_id="chunk-001",
            doc_id="cdc-syphilis",
            section_path="Treatment, Recommendations",
            anchor="primary-regimen",
            source_url="https://example.com/syphilis",
            chunk_index=0,
            text=(
                "The primary treatment for syphilis is penicillin G. For primary, "
                "secondary, and early latent syphilis, penicillin G benzathine is "
                "the preferred agent."
            ),
            token_count=30,
        ),
        Chunk(
            chunk_id="chunk-002",
            doc_id="cdc-syphilis",
            section_path="Treatment, Special Populations, Pregnancy",
            anchor="pregnancy-dosing",
            source_url="https://example.com/syphilis",
            chunk_index=1,
            text=(
                "Pregnant women with syphilis should be treated with the penicillin "
                "regimen appropriate for their stage of infection. Penicillin is the "
                "only recommended treatment for maternal syphilis."
            ),
            token_count=28,
        ),
        Chunk(
            chunk_id="chunk-003",
            doc_id="cdc-syphilis",
            section_path="Treatment, Dosing, Primary Syphilis",
            anchor="dosing-primary",
            source_url="https://example.com/syphilis",
            chunk_index=2,
            text=(
                "For primary syphilis, the recommended dose is penicillin G "
                "benzathine 2.4 million units IM in a single dose."
            ),
            token_count=20,
        ),
        Chunk(
            chunk_id="chunk-004",
            doc_id="cdc-gonorrhea",
            section_path="Treatment, Recommendations",
            anchor="gonorrhea-treatment",
            source_url="https://example.com/gonorrhea",
            chunk_index=0,
            text=(
                "Gonorrhea can be treated with ceftriaxone 250 mg IM as a single "
                "dose or cefixime 400 mg orally as a single dose."
            ),
            token_count=25,
        ),
        Chunk(
            chunk_id="chunk-005",
            doc_id="health-canada-immunization",
            section_path="Vaccines, HPV, Recommendations",
            anchor="hpv-dosing",
            source_url="https://example.com/immunization",
            chunk_index=0,
            text=(
                "HPV vaccine is recommended for all individuals aged 9-45 years. "
                "The vaccine series consists of 2 or 3 doses depending on age at "
                "initiation."
            ),
            token_count=27,
        ),
    ]

    print("Populating vector store with test chunks...")
    embed_and_store(test_chunks, provider, store)
    store.close()
    print(f"Stored {len(test_chunks)} chunks successfully.")


def run_agent(query: str) -> None:
    """Run the agent graph: retrieve, grade, generate, verify."""
    dsn = "postgresql://postgres:postgres@localhost:5433/clinical_rag_test"
    dimension = 16

    embedding_provider = FakeEmbeddingProvider(dimension=dimension)
    vector_store = PgVectorStore(dsn, dimension=dimension)
    llm_provider = FakeLLMProvider()

    graph = build_graph(
        llm_provider=llm_provider,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    print(f"Query: {query}\n")

    initial_state = AgentState(question=query)
    final_state = graph.invoke(initial_state)

    print(f"Iterations: {final_state.iteration}")
    print(f"Chunks retrieved: {len(final_state.retrieved)}")
    print(f"Verdict: {final_state.verdict}\n")

    if final_state.answer:
        print(f"Answer:\n{final_state.answer}\n")
        if final_state.citations:
            print(f"Citations: {', '.join(final_state.citations)}")
        else:
            print("No citations")
    else:
        print("No answer generated")

    vector_store.close()


def ask_query(query: str) -> None:
    """Search for and return top-k chunks matching the query."""
    dsn = "postgresql://postgres:postgres@localhost:5433/clinical_rag_test"
    dimension = 16

    provider = FakeEmbeddingProvider(dimension=dimension)
    store = PgVectorStore(dsn, dimension=dimension)

    print(f"Query: {query}\n")

    query_embedding = provider.embed([query])[0]
    results = store.search(query, query_embedding, k=5)

    if not results:
        print("No relevant chunks found.")
        return

    print(f"Found {len(results)} relevant chunks:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Chunk {result.chunk_id} (score: {result.score:.4f})")
        print(f"   Doc: {result.doc_id}")
        print(f"   Section: {result.section_path}")
        print(f"   Text: {result.text[:100]}...")
        print()

    store.close()


if __name__ == "__main__":
    main()
