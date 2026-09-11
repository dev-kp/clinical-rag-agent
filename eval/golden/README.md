# Golden set

`golden_set.json`: 42 hand-verified questions used for Phase 4 evaluation.

| Category | Count | Target (plan) |
|---|---|---|
| single-hop | 26 (62%) | ~40% |
| multi-hop | 5 (12%) | ~40% |
| unanswerable | 11 (26%) | ~20% |
| **Total** | **42** | 60-80 |

## How this was built

1. `generate_candidates.py` used a real LLM (Groq, `openai/gpt-oss-20b`) to draft 53
   candidate questions from real chunks in the corpus: 32 single-hop (one random
   chunk), 9 multi-hop (chunk pairs from doc pairs known to cross-reference each
   other, e.g. a syphilis stage chapter + the penicillin-allergy chapter), and 12
   unanswerable (plausible clinical topics chosen to be absent from this corpus).
2. Every one of the 53 candidates was checked against its source chunk text.
   11 were rejected:
   - 6 were bibliography/reference-list lookups (e.g. "what is this paper's DOI"),
     not genuine clinical content — the LLM had drafted a question from a
     citation list chunk rather than actual guidance.
   - 2 were mislabeled multi-hop: one of the two source chunks was a stub
     ("see [other page]") or a bare table of contents, so the question was
     actually answerable from the other chunk alone.
   - 1 had a partner-management claim not actually supported by either source
     chunk (the source only said "see [other page]", not what that page says).
   - 1 unanswerable candidate ("rabies post-exposure prophylaxis schedule") was
     rejected after confirming the corpus does contain related rabies
     vaccination content — an unsafe pick for a question meant to test
     abstention.
3. This verification pass was done by Claude, not by hand by the project owner.
   That is a materially weaker claim than "hand-verified by a human reviewer"
   and is recorded here honestly rather than glossed over.

## Known gap

The set is smaller than the plan's 60-80 target, and multi-hop is
under-represented (12% vs. the ~40% target) because several of the drafted
multi-hop candidates turned out, on inspection, to be answerable from a single
chunk — the second "excerpt" was often a stub or a table of contents rather
than substantive content. Expanding multi-hop coverage (more genuinely
cross-referencing chunk pairs beyond penicillin-allergy) is the clearest next
step if the eval needs a larger, more balanced set later.
