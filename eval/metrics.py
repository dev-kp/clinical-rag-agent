"""Evaluation metrics for the ablation study.

context_recall is computed exactly (set overlap against ground truth, no
LLM needed). faithfulness, answer_relevancy, and context_precision are
genuinely judgment calls, so they're computed with an LLM judge. This
mirrors ragas's own approach for context_recall while avoiding the
dependency (see pyproject.toml's eval extras and the commit removing
ragas: it currently fails to import due to an upstream langchain-community
break).
"""

import json
import time

import httpx

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_MODEL = "openai/gpt-oss-120b"
# Batch eval runs make many more calls per minute than interactive use, so
# this needs to tolerate longer bursts of rate limiting than a single query
# would (see the commit fixing the ablation runner's first failed sweep).
MAX_RETRIES = 10


def context_recall(retrieved_chunk_ids: list[str], ground_truth_chunk_ids: list[str]) -> float:
    """Fraction of ground-truth chunks that were actually retrieved.

    1.0 means every chunk needed to answer the question was retrieved.
    0.0 means none were. Undefined (returns 1.0) for unanswerable
    questions, which have no ground-truth chunks by definition.
    """
    if not ground_truth_chunk_ids:
        return 1.0

    retrieved = set(retrieved_chunk_ids)
    found = sum(1 for cid in ground_truth_chunk_ids if cid in retrieved)
    return found / len(ground_truth_chunk_ids)


def context_precision(
    question: str, retrieved_texts: list[str], api_key: str
) -> float:
    """Fraction of retrieved chunks the judge considers actually relevant
    to answering the question. Isolates retrieval precision from whether
    the final answer happened to be good."""
    if not retrieved_texts:
        return 0.0

    numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(retrieved_texts))
    prompt = (
        "You are judging retrieval quality for a clinical question-answering system.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved passages:\n{numbered}\n\n"
        "For each numbered passage, decide if it is actually relevant to "
        "answering the question (not just topically related, but useful "
        "evidence for the specific question asked).\n\n"
        'Respond with ONLY JSON: {"relevant_indices": [0, 2, ...]}'
    )
    result = _ask_judge_json(prompt, api_key)
    if result is None:
        return 0.0

    relevant = result.get("relevant_indices", [])
    return len(relevant) / len(retrieved_texts)


def faithfulness(answer: str, retrieved_texts: list[str], api_key: str) -> float:
    """Fraction of claims in the answer that are actually supported by the
    retrieved context. Catches hallucination: an answer can be fluent and
    relevant while still asserting things the sources never said."""
    if not answer.strip():
        return 1.0  # an empty/abstained answer makes no unsupported claims

    context = "\n\n".join(retrieved_texts)
    prompt = (
        "You are checking a clinical answer for hallucination.\n\n"
        f"Answer to check:\n{answer}\n\n"
        f"Source context the answer should be based on:\n{context}\n\n"
        "Break the answer into its individual factual claims. A statement "
        "that the sources lack information (e.g. 'the sources do not "
        "mention X') is not itself a factual claim to verify against the "
        "context and should not be counted.\n\n"
        'Respond with ONLY JSON: {"total_claims": N, "supported_claims": M}'
    )
    result = _ask_judge_json(prompt, api_key)
    if result is None:
        return 0.0  # judge failed to respond at all - genuinely unscoreable

    total_claims = result.get("total_claims", 0)
    if total_claims == 0:
        return 1.0  # no verifiable claims made means no unsupported claims

    return result["supported_claims"] / total_claims


def answer_relevancy(question: str, answer: str, api_key: str) -> float:
    """How directly the answer addresses the question asked, independent
    of whether the answer is factually correct. Catches evasive, off-topic,
    or incomplete answers that still happen to cite real sources."""
    if not answer.strip():
        return 0.0

    prompt = (
        "You are judging whether an answer actually addresses the question "
        "asked, regardless of whether the answer is medically correct.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Rate how directly and completely the answer addresses the "
        "question, from 0.0 (does not address it at all) to 1.0 (fully "
        "and directly addresses it).\n\n"
        'Respond with ONLY JSON: {"relevancy_score": 0.0 to 1.0}'
    )
    result = _ask_judge_json(prompt, api_key)
    if result is None:
        return 0.0

    score = result.get("relevancy_score", 0.0)
    return max(0.0, min(1.0, float(score)))


def _ask_judge_json(prompt: str, api_key: str) -> dict | None:
    raw = _call_groq(prompt, api_key)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _call_groq(prompt: str, api_key: str) -> str:
    for attempt in range(MAX_RETRIES):
        response = httpx.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=60.0,
        )
        if response.status_code != 429:
            response.raise_for_status()
            content: str = response.json()["choices"][0]["message"]["content"]
            return content.strip()

        if attempt == MAX_RETRIES - 1:
            raise RuntimeError(f"Groq rate limit not resolved after {MAX_RETRIES} attempts")
        time.sleep(6.0)

    raise AssertionError("unreachable")
