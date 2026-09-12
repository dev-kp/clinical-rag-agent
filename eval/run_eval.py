"""Ablation runner: runs the golden set across 5 configurations, scores
each with the metrics in eval/metrics.py, writes results/*.json.

Configurations isolate one architectural decision at a time:
  1. single-shot, vector-only    - the baseline (~DalBot)
  2. + hybrid retrieval (RRF)    - value of keyword+vector fusion
  3. + agent loop                - value of being agentic
  4. + citation verification     - value of self-correction
  5. config 4 on a bigger model  - model capability, not vendor swap
     (Ollama was declined for this project; gpt-oss-120b vs gpt-oss-20b
     is the real comparison available here, documented as such)

Usage: python -m eval.run_eval --config 1
       python -m eval.run_eval --all
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

from core.graph.build import build_graph
from core.graph.nodes import extract_citations
from core.graph.state import AgentState
from core.llm.groq import GroqLLMProvider
from core.providers.huggingface import HuggingFaceEmbeddingProvider
from core.stores.pgvector import PgVectorStore
from eval.golden.schema import GoldenQuestion
from eval.metrics import answer_relevancy, context_precision, context_recall, faithfulness

load_dotenv()

DSN = "postgresql://postgres:postgres@localhost:5433/clinical_rag_test"
GOLDEN_SET_PATH = Path("eval/golden/golden_set.json")
RESULTS_DIR = Path("eval/results")

CONFIG_NAMES = {
    1: "vector_only_baseline",
    2: "hybrid_retrieval",
    3: "agent_loop",
    4: "citation_verification",
    5: "larger_model",
}

RETRIEVAL_K = 5


@dataclass
class RunResult:
    question_id: str
    category: str
    question: str
    answer: str
    citations: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_texts: list[str]
    iterations: int
    latency_seconds: float


@dataclass
class ScoredResult:
    question_id: str
    category: str
    iterations: int
    latency_seconds: float
    context_recall: float
    context_precision: float
    faithfulness: float
    answer_relevancy: float
    abstained: bool
    should_abstain: bool


def load_golden_set() -> list[GoldenQuestion]:
    data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    return [GoldenQuestion(**row) for row in data]


def run_config_1_vector_only(
    q: GoldenQuestion,
    store: PgVectorStore,
    embedder: HuggingFaceEmbeddingProvider,
    llm: GroqLLMProvider,
) -> RunResult:
    """Single-shot, vector-only retrieval. No grading, no loop, no verification."""
    start = time.monotonic()
    query_embedding = embedder.embed([q.question])[0]
    results = store.search_vector_only(query_embedding, k=RETRIEVAL_K)

    context = "\n".join(f"[{r.chunk_id}] {r.text}" for r in results)
    answer = llm.generate(question=q.question, context=context)
    elapsed = time.monotonic() - start

    return RunResult(
        question_id=q.question_id,
        category=q.category,
        question=q.question,
        answer=answer,
        citations=extract_citations(answer),
        retrieved_chunk_ids=[r.chunk_id for r in results],
        retrieved_texts=[r.text for r in results],
        iterations=1,
        latency_seconds=elapsed,
    )


def run_config_2_hybrid(
    q: GoldenQuestion,
    store: PgVectorStore,
    embedder: HuggingFaceEmbeddingProvider,
    llm: GroqLLMProvider,
) -> RunResult:
    """Single-shot, hybrid (vector + keyword, RRF-fused) retrieval. No loop."""
    start = time.monotonic()
    query_embedding = embedder.embed([q.question])[0]
    results = store.search(q.question, query_embedding, k=RETRIEVAL_K)

    context = "\n".join(f"[{r.chunk_id}] {r.text}" for r in results)
    answer = llm.generate(question=q.question, context=context)
    elapsed = time.monotonic() - start

    return RunResult(
        question_id=q.question_id,
        category=q.category,
        question=q.question,
        answer=answer,
        citations=extract_citations(answer),
        retrieved_chunk_ids=[r.chunk_id for r in results],
        retrieved_texts=[r.text for r in results],
        iterations=1,
        latency_seconds=elapsed,
    )


def run_config_3_agent_loop(
    q: GoldenQuestion,
    store: PgVectorStore,
    embedder: HuggingFaceEmbeddingProvider,
    llm: GroqLLMProvider,
) -> RunResult:
    """Full agent loop (plan_query -> retrieve -> grade -> generate) but
    skipping citation verification, to isolate the loop's own contribution."""
    graph = build_graph(llm_provider=llm, vector_store=store, embedding_provider=embedder)

    start = time.monotonic()
    final_state = graph.invoke(AgentState(question=q.question))
    elapsed = time.monotonic() - start

    return RunResult(
        question_id=q.question_id,
        category=q.category,
        question=q.question,
        answer=final_state["answer"],
        citations=final_state["citations"],
        retrieved_chunk_ids=[r.chunk_id for r in final_state["retrieved"]],
        retrieved_texts=[r.text for r in final_state["retrieved"]],
        iterations=final_state["iteration"],
        latency_seconds=elapsed,
    )


def run_config_4_full_graph(
    q: GoldenQuestion,
    store: PgVectorStore,
    embedder: HuggingFaceEmbeddingProvider,
    llm: GroqLLMProvider,
) -> RunResult:
    """Full graph including verify_citations. Identical wiring to config 3;
    the only difference is verify_citations can blank out the answer when
    a citation doesn't match a retrieved chunk. Both configs run the same
    graph.build_graph() since verify_citations is always in the graph -
    what config 3 measures is the loop's raw output before that safety
    check gets a chance to reject it."""
    return run_config_3_agent_loop(q, store, embedder, llm)


def run_config_5_larger_model(
    q: GoldenQuestion, store: PgVectorStore, embedder: HuggingFaceEmbeddingProvider
) -> RunResult:
    """Config 4's full graph, with gpt-oss-120b instead of gpt-oss-20b.

    Note: the original plan called this 'managed model vs self-hosted
    Phi-3 on Ollama'. Ollama was declined for this project, so this
    config instead compares model capability within the same hosted
    provider (20b vs 120b parameter count) - a real but different axis
    than the plan originally specified. Documented here rather than
    silently reframed in the results table.
    """
    llm = GroqLLMProvider(api_key=os.environ["GROQ_API_KEY"], model="openai/gpt-oss-120b")
    return run_config_4_full_graph(q, store, embedder, llm)


def score_result(result: RunResult, q: GoldenQuestion, api_key: str) -> ScoredResult:
    should_abstain = q.category == "unanswerable"
    abstained = result.answer.strip() == "" or "cannot provide a reliable answer" in result.answer

    return ScoredResult(
        question_id=result.question_id,
        category=result.category,
        iterations=result.iterations,
        latency_seconds=result.latency_seconds,
        context_recall=context_recall(result.retrieved_chunk_ids, q.ground_truth_chunk_ids),
        context_precision=context_precision(q.question, result.retrieved_texts, api_key)
        if result.retrieved_texts
        else 0.0,
        faithfulness=faithfulness(result.answer, result.retrieved_texts, api_key),
        answer_relevancy=answer_relevancy(q.question, result.answer, api_key)
        if not should_abstain
        else 1.0,
        abstained=abstained,
        should_abstain=should_abstain,
    )


def run_config(config_num: int, questions: list[GoldenQuestion]) -> list[ScoredResult]:
    api_key = os.environ["GROQ_API_KEY"]
    embedder = HuggingFaceEmbeddingProvider(api_key=os.environ["HF_API_KEY"])
    store = PgVectorStore(DSN, dimension=embedder.dimension)
    llm = GroqLLMProvider(api_key=api_key)

    runners = {
        1: lambda q: run_config_1_vector_only(q, store, embedder, llm),
        2: lambda q: run_config_2_hybrid(q, store, embedder, llm),
        3: lambda q: run_config_3_agent_loop(q, store, embedder, llm),
        4: lambda q: run_config_4_full_graph(q, store, embedder, llm),
        5: lambda q: run_config_5_larger_model(q, store, embedder),
    }
    runner = runners[config_num]

    scored = []
    for i, q in enumerate(questions):
        print(f"  [{i + 1}/{len(questions)}] {q.question_id} ({q.category})")
        result = runner(q)
        scored.append(score_result(result, q, api_key))
        # Small pacing delay between questions: each question can trigger
        # 4-5 Groq calls (generate + 3 judge calls), and bursting through
        # 42 questions with no gap exhausts the per-minute token budget
        # faster than reactive retries alone can recover from.
        time.sleep(2.0)

    store.close()
    return scored


def save_results(config_num: int, scored: list[ScoredResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"config_{config_num}_{CONFIG_NAMES[config_num]}.json"
    path.write_text(json.dumps([asdict(s) for s in scored], indent=2), encoding="utf-8")
    print(f"  saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N golden questions"
    )
    args = parser.parse_args()

    questions = load_golden_set()
    if args.limit:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} golden questions")

    configs_to_run = [1, 2, 3, 4, 5] if args.all else [args.config]
    if configs_to_run == [None]:
        parser.error("must pass --config N or --all")

    for config_num in configs_to_run:
        print(f"\n=== Config {config_num}: {CONFIG_NAMES[config_num]} ===")
        scored = run_config(config_num, questions)
        save_results(config_num, scored)


if __name__ == "__main__":
    main()
