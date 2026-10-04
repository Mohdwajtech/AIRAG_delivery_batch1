# DR #2 One-Pager: Retrieval + KB Lifecycle Defense

## 1. Problem Framing

We built a RAG knowledge assistant for ~50 employees over 15 corporate policy
documents (HR, security, IT, compliance). Constraints: sub-$0.001/query cost,
p50 latency < 300ms, zero hallucinated policy claims (compliance risk — a wrong
answer about leave entitlement or security protocol creates liability).

We chose hybrid retrieval (dense + BM25 + cross-encoder reranking) because
dense-only missed keyword-specific queries (precision dropped 8pts in W9
testing). We added semantic caching because uncached cost ($0.0009/query)
exceeded the budget by 9×. The cache brought cost to $0.00013 — within budget.

## 2. Trade-offs

Key decisions with evidence:

**Hybrid + rerank vs dense-only:** Dense-only precision 0.70. Hybrid: 0.83.
Hybrid + rerank: 0.88. Each addition justified by measured lift. BM25 weight
tuned at 0.3 (0.5 dropped precision 2pts due to keyword over-matching).
See ADR v2, rows 5–6.

**Top-K = 3 vs 5:** k=3 gives precision 0.88 with recall 0.80. k=5 gives
recall 0.83 (+3pts) but precision 0.84 (−4pts) and 40% higher cost. We chose
precision over recall because hallucination risk (from noisy context) outweighs
incomplete-answer risk. See ADR v2, row 7.

**Cache threshold 0.95:** Tested 0.90, 0.93, 0.95. At 0.90, "parental leave"
matched "working hours" (false hit). At 0.95, zero false hits across 200
queries. Conservative but safe. See ADR v2, row 8.

## 3. KPI Reporting

**Trajectory (W6 → W12):**
- Task success: 0.72 → 0.92 (+20pts). Biggest driver: W9 hybrid retrieval (+8pts).
- Cost: $0.0009 → $0.00013 (7× reduction). Driver: W10 semantic cache.
- Latency p50: 1100ms → 184ms (6× reduction). Driver: W10 cache hits.

**Instrument change at W11:** W5 judge → RAGAS faithfulness. Numbers dropped
(0.94 → 0.91) due to stricter measurement, not regression. Documented.

**KPI #4 reconciliation at W12:** Context precision (proxy) → correct-chunk-in-top-K
(canonical). Numbers changed due to definition, not system. Documented.

**N/A honesty:** KPIs #5 (tool success) and #8 (escalation) are N/A. Not zero,
not fabricated. Arrival weeks noted (W13, W17).

Full dashboard: `docs/kpi/wk12-snapshot.md`.

## 4. Failure Handling

Four failure modes identified from the W12 debugging drill:

**Recall miss (retrieval):** Multi-hop queries (g015) miss the 2nd chunk at k=3.
Defense: query expansion planned for Phase 3 (W14). Stopgap: raise k to 5 for
detected multi-hop, with precision guard.

**Stale cache (infra):** Document update without cache clear → stale answers.
Defense: cache cleared on EVERY update (non-negotiable, enforced in
`ingest_or_update_source()`). kb_version matching rejects cross-version hits.
Runbook: `docs/runbooks/document_update.md`.

**Prompt drift (prompt):** Grounding instruction removed accidentally → hallucination.
Defense: CI check `assert "only from" in SYSTEM_PROMPT.lower()`. Catches every
prompt regression before deployment.

**API timeout (infra):** LLM call exceeds 30s → no answer.
Defense: retry with backoff (planned). Fallback to smaller model (GPT-4o-mini
→ GPT-4o-mini-instant if available).

**Open gap:** Multi-hop with > 2 hops. Diagnosis documented in
`docs/debugging/wk12-debug-trace.md` (stretch activity).

## 5. ADR Quality

ADR v2 (M2): 13 decisions consolidated in `docs/adr/decisions.md`.
Evolution visible: v1 (M1, W5 chatbot) → v2 (M2, Phase 2 consolidation).
Each decision has: choice, why, alternatives rejected, KPI evidence.

Definition lock documented: KPI #4 reconciliation is an ADR entry (row 13).
Instrument change documented: W11 RAGAS transition noted in methodology.

DR #2 feedback will be committed live during the review.
