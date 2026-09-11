"""Generate candidate golden-set questions from the real corpus for hand review.

This does NOT produce the final golden set. It produces a JSON file of
LLM-drafted candidates (question, category, draft answer, source chunk ids)
that a human must read and correct before they count as ground truth. See
eval/golden/README.md for the review workflow.

Usage: python -m eval.golden.generate_candidates
"""

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import psycopg
from dotenv import load_dotenv

load_dotenv()

DSN = "postgresql://postgres:postgres@localhost:5433/clinical_rag_test"
OUTPUT_PATH = Path("eval/golden/candidates.json")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 5

# Doc pairs that genuinely cross-reference each other, chosen by reading
# SOURCES.md's description of the corpus's multi-hop structure, not guessed.
MULTI_HOP_DOC_PAIRS = [
    ("sti-syphilis-primary-secondary", "sti-penicillin-allergy"),
    ("sti-syphilis-latent", "sti-penicillin-allergy"),
    ("sti-syphilis-pregnancy", "sti-penicillin-allergy"),
    ("sti-syphilis-neuro", "sti-penicillin-allergy"),
    ("sti-syphilis", "sti-syphilis-hiv"),
    ("sti-gonorrhea-adults", "sti-pid"),
    ("cig-populations", "cig-influenza"),
    ("cig-populations", "cig-hepb"),
    ("cig-adults", "cig-schedules"),
]

# Plausible-sounding clinical topics genuinely absent from this corpus.
UNANSWERABLE_TOPICS = [
    "the recommended treatment for tuberculosis",
    "dosing of metformin for type 2 diabetes",
    "the CDC's guidance on seasonal allergy treatment",
    "recommended chemotherapy regimens for breast cancer",
    "the vaccination schedule for rabies post-exposure prophylaxis",
    "treatment guidelines for migraine headaches",
    "the recommended antibiotic for a urinary tract infection in a non-pregnant adult",
    "guidance on treating high blood pressure in adults",
    "the recommended dose of ibuprofen for adults",
    "screening guidelines for colorectal cancer",
    "treatment for seasonal influenza in children under 2",
    "guidance on managing type 1 diabetes in pediatric patients",
]


@dataclass
class Candidate:
    category: str
    source_chunk_ids: list[str]
    draft_question: str
    draft_answer: str


def ask_llm(prompt: str) -> str:
    """One-off Groq call for authoring, separate from LLMProvider since this
    is a candidate-generation tool, not part of the agent's runtime interface.
    Retries on 429 the same way GroqLLMProvider does."""
    for attempt in range(MAX_RETRIES):
        response = httpx.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json={
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
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


def ask_llm_json(prompt: str) -> dict | None:
    raw = ask_llm(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def fetch_chunk_text(conn: psycopg.Connection, chunk_id: str) -> str:
    row = conn.execute("SELECT text FROM chunks WHERE chunk_id = %s", (chunk_id,)).fetchone()
    if row is None:
        raise ValueError(f"chunk_id not found: {chunk_id}")
    text: str = row[0]
    return text


def sample_chunk_ids(conn: psycopg.Connection, doc_id: str, n: int = 1) -> list[str]:
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id = %s ORDER BY random() LIMIT %s",
        (doc_id, n),
    ).fetchall()
    return [r[0] for r in rows]


def generate_single_hop(conn: psycopg.Connection, n: int) -> list[Candidate]:
    all_ids = [r[0] for r in conn.execute("SELECT chunk_id FROM chunks").fetchall()]
    chosen = random.sample(all_ids, n)

    candidates = []
    for chunk_id in chosen:
        text = fetch_chunk_text(conn, chunk_id)
        prompt = (
            "Given this single excerpt from a clinical guideline, write ONE "
            "specific question that this excerpt alone fully answers, plus the "
            "correct answer using only this excerpt.\n\n"
            f"Excerpt:\n{text}\n\n"
            'Respond with ONLY JSON: {"question": "...", "answer": "..."}'
        )
        result = ask_llm_json(prompt)
        if result:
            candidates.append(
                Candidate(
                    category="single-hop",
                    source_chunk_ids=[chunk_id],
                    draft_question=result.get("question", ""),
                    draft_answer=result.get("answer", ""),
                )
            )
    return candidates


def generate_multi_hop(conn: psycopg.Connection, n: int) -> list[Candidate]:
    pairs = random.sample(MULTI_HOP_DOC_PAIRS, min(n, len(MULTI_HOP_DOC_PAIRS)))

    candidates = []
    for doc_a, doc_b in pairs:
        ids_a = sample_chunk_ids(conn, doc_a, 1)
        ids_b = sample_chunk_ids(conn, doc_b, 1)
        if not ids_a or not ids_b:
            continue
        text_a = fetch_chunk_text(conn, ids_a[0])
        text_b = fetch_chunk_text(conn, ids_b[0])

        prompt = (
            "Given these two excerpts from DIFFERENT chapters of a clinical "
            "guideline, write ONE question that genuinely requires BOTH excerpts "
            "to answer correctly (not answerable from either alone), plus the "
            "correct combined answer.\n\n"
            f"Excerpt A:\n{text_a}\n\n"
            f"Excerpt B:\n{text_b}\n\n"
            'Respond with ONLY JSON: {"question": "...", "answer": "..."}'
        )
        result = ask_llm_json(prompt)
        if result:
            candidates.append(
                Candidate(
                    category="multi-hop",
                    source_chunk_ids=[ids_a[0], ids_b[0]],
                    draft_question=result.get("question", ""),
                    draft_answer=result.get("answer", ""),
                )
            )
    return candidates


def generate_unanswerable(n: int) -> list[Candidate]:
    topics = random.sample(UNANSWERABLE_TOPICS, min(n, len(UNANSWERABLE_TOPICS)))
    return [
        Candidate(
            category="unanswerable",
            source_chunk_ids=[],
            draft_question=f"What is {topic}?",
            draft_answer="",
        )
        for topic in topics
    ]


def main() -> None:
    with psycopg.connect(DSN, autocommit=True) as conn:
        print("Generating single-hop candidates...")
        single_hop = generate_single_hop(conn, n=32)

        print("Generating multi-hop candidates...")
        multi_hop = generate_multi_hop(conn, n=9)

        print("Generating unanswerable candidates...")
        unanswerable = generate_unanswerable(n=12)

    all_candidates = single_hop + multi_hop + unanswerable
    OUTPUT_PATH.write_text(
        json.dumps([asdict(c) for c in all_candidates], indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {len(all_candidates)} candidates to {OUTPUT_PATH}")
    print(
        f"  single-hop: {len(single_hop)}, multi-hop: {len(multi_hop)}, "
        f"unanswerable: {len(unanswerable)}"
    )
    print("\nThese are DRAFTS. Every one must be hand-reviewed before use.")


if __name__ == "__main__":
    main()
