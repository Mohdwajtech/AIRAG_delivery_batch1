# Architecture Decision Record — v2 (M2)

**Version history:**
- v1 (M1, W5): Chatbot architecture — LLM choice, judge model, golden set format.
- v2 (M2, W12): Phase 2 consolidation — retrieval, caching, lifecycle, evaluation.

## Consolidated Decision Table

| # | Decision | Choice | Why | Alternatives Rejected | KPI Evidence |
|---|----------|--------|-----|----------------------|-------------|
| 1 | Embedding model | text-embedding-3-small (1536d) | Best cost/quality ratio at capstone scale (<100 docs). $0.02/1M tokens. | text-embedding-3-large (3072d): 2× cost, marginal quality gain at our scale. | W7 grid search: precision difference < 2pts. |
| 2 | Vector database | Qdrant (cloud) | Managed, cosine-native, payload filtering for tombstones. Free tier sufficient. | Pinecone (similar but less flexible filtering), ChromaDB (no cloud managed option), FAISS (no persistence). | W7: collection creation + query latency ~88ms. |
| 3 | Chunking strategy | Structure-aware, 200–400 tokens, paragraph→sentence fallback | Preserves section boundaries. Heading metadata enables section-level retrieval. | Fixed-window 512 tokens (splits mid-sentence, loses structure), sentence-level (too granular, high chunk count). | W8→W9 precision jump: 0.67 → 0.83. |
| 4 | PII scrubbing | Regex (5 patterns) + Presidio NER (graceful fallback) | Regex catches structured PII (email, SSN). Presidio catches names/locations. Fallback if Presidio unavailable. | Regex-only (misses names), Presidio-only (slow, heavy dependency), no scrubbing (compliance risk). | W8: 3 chunks flagged in test corpus. |
| 5 | Retrieval method | Hybrid (dense + BM25) + cross-encoder rerank | Dense captures semantics. BM25 captures keywords. Reranker provides precision. Together: best F1 on golden set. | Dense-only (0.70 precision, misses keyword queries), BM25-only (0.68, misses semantic matches), Hybrid without rerank (0.83, good but rerank adds +5pts). | W9 decision table: 0.70 → 0.83 → 0.88. |
| 6 | BM25 weight (RRF) | 0.3 (via RRF k=60) | Tested 0.1, 0.3, 0.5. At 0.5, BM25 over-weighted exact keyword matches on policy names. 0.3 balances dense+keyword. | 0.1 (BM25 nearly ignored), 0.5 (precision dropped 2pts on mixed queries). | W9 grid search, 5 configs tested. |
| 7 | Top-K | 3 | Sufficient recall for single-hop queries. Low noise. Acceptable cost. | 5 (recall +3pts but precision −4pts, cost +40%), 7 (diminishing returns, high noise). | W10 reduce-k analysis. Context recall ≥ 0.80 at k=3. |
| 8 | Cache threshold | Cosine ≥ 0.95 | Zero false hits across 200 test queries. Safe default. | 0.90 (false hits: "parental leave" matched "working hours"), 0.97 (too strict, only exact duplicates hit). | W10 false-hit demo: 0 false hits at 0.95. |
| 9 | Cache invalidation | Clear ALL cache on every doc update | Non-negotiable for correctness. Stale answers are worse than cache misses. | Selective invalidation (complex, error-prone), time-based expiry (stale window too wide), no invalidation (stale answers guaranteed). | W10 runbook: 3-signal verification. |
| 10 | Tombstone pattern | Soft delete (deleted_at timestamp + IsEmptyCondition filter) | Rollback, audit trail, forensic queries. No data loss. | Hard delete (no recovery, no audit), archive to separate collection (complex, query fragmentation). | W10 ADR §4.5, IsEmptyCondition (not IsNull). |
| 11 | Eval framework | RAGAS (offline proxy + live) | Industry standard. Four metrics map to four RAG quality dimensions. Offline proxy enables CI. | DeepEval (less community adoption), TruLens (heavier), custom metrics only (no industry comparability). | W11: 4 metrics computed, thresholds passing. |
| 12 | RAGAS thresholds | faith ≥ 0.85, relev ≥ 0.85, prec ≥ 0.75, recall ≥ 0.80 | Derived from 3 baseline runs. Set below noise floor with buffer. | Tutorial defaults (not calibrated to our system), aggressive thresholds (false alarms in CI). | W11 threshold derivation: 3 runs → floor → buffer. |
| 13 | KPI #4 definition | Correct chunk in top-K (canonical, W12) | Matches the stated definition. Context precision (W11 proxy) measures a different thing. | Keep using context precision (valid but doesn't match the KPI name), invent a new KPI (unnecessary complexity). | W12 reconciliation note in snapshot. |

## Prior ADR Documents (preserved for lineage)

- `0001-capstone-framing.md` — W5: problem framing, audience, constraints.
- `0002-pipeline-foundation.md` — W3: API contract, pipeline architecture.
- `0003-retrieval-architecture.md` — W9: retrieval method comparison.

## DR #2 Feedback

_(To be filled during the design review — commit live during the review.)_
