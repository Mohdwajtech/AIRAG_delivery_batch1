# W9 KPI Snapshot

## Three-Method Staircase

| Method            | Precision@3 | Latency p50 | Cost/Query | Notes                          |
|-------------------|-------------|-------------|------------|--------------------------------|
| Dense (W8)        | --/%        | --ms        | $--        | Baseline — cosine via Qdrant   |
| Hybrid            | --/%        | --ms        | $--        | + BM25 + RRF (k=60)           |
| Hybrid + Rerank   | --/%        | --ms        | $--        | + ms-marco-MiniLM-L-6-v2      |

## Expected Pattern
dense ≤ hybrid ≤ hybrid_rerank

If inverted:
- hybrid < dense → check BM25 tokenizer (same function for build + query?)
- hybrid_rerank < hybrid → check text key in rerank() (full text, not title?)

## Per-Query Breakdown

| Query | Dense top-3? | Hybrid top-3? | H+Rerank top-3? | Diagnosis |
|-------|-------------|--------------|-----------------|-----------|
| Q1    |             |              |                 |           |
| Q2    |             |              |                 |           |
| Q3    |             |              |                 |           |
| ...   |             |              |                 |           |

## Latency Budget (from notebook Cell 13)

| Stage          | Latency | Notes                         |
|----------------|---------|-------------------------------|
| Embed query    | --ms    | OpenAI API (network-bound)    |
| Dense (Qdrant) | --ms    | HNSW search (network-bound)   |
| BM25           | --ms    | In-memory (~0.2ms)            |
| RRF fusion     | --ms    | Pure Python (<1ms)            |
| Rerank (k=10)  | --ms    | Cross-encoder CPU             |
| **TOTAL**      | **--ms**| Without LLM generation        |

## Decisions
- Default retrieval method: `hybrid_rerank`
- BM25 tokenizer: `simple_tokenize` (keeps hyphens/slashes for identifiers)
- Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Rerank candidates: 10-20 (default 20)
- RRF k=60 (canonical default, unchanged)
