# Clinical RAG Agent

An agentic RAG system over public clinical guidelines. Instead of a single
retrieve-then-answer pass, the agent retrieves, grades whether the evidence it has is
sufficient, retrieves again if not (up to a hard iteration cap), then answers strictly
from what it found — with inline citations that are programmatically and semantically
verified before being returned.

**This is a document-QA system over public guidelines, not a source of medical advice.**
It is designed to abstain when the corpus doesn't support a confident answer, and that
abstention behavior is itself measured (see [Evaluation](#evaluation)).

## Why this exists

Built to close three specific gaps against real Toronto-area GenAI engineering job
postings: no Azure experience, no named vector database, no evaluation tooling. See
`docs/` (coming in later phases) for the full architecture writeup.

## Architecture

```
question → plan_query → retrieve ⇄ grade_evidence → generate → verify_citations → answer
                            (loops up to MAX_RETRIEVAL_ITERATIONS if evidence insufficient)
```

- **Orchestration**: LangGraph (explicit state machine — see the Step Functions
  comparison in the design notes)
- **Vector store**: pgvector (local/default) or Azure AI Search, behind one interface
- **Models**: Azure OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) or local
  Ollama/Phi-3, behind one interface
- **Evaluation**: ragas (faithfulness, answer relevancy, context precision/recall) +
  custom abstention and cost/latency metrics, run as an ablation matrix

## Status

Early scaffold. See phase-by-phase progress below.

- [x] Phase 0 — Project scaffold, Docker, CI
- [ ] Phase 1 — Ingestion & chunking
- [ ] Phase 2 — Vector store & hybrid retrieval
- [ ] Phase 3 — Agent loop
- [ ] Phase 4 — Evaluation
- [ ] Phase 5 — Azure port
- [ ] Phase 6 — Frontend
- [ ] Phase 7 — Deployment & CI polish

## Running locally

```bash
cp .env.example .env
docker compose up
```

## Evaluation

Results will be published here as an ablation table once Phase 4 lands: baseline
single-shot retrieval vs. hybrid retrieval vs. the full agent loop vs. citation
verification, each with faithfulness/relevancy/recall numbers and cost per query.
