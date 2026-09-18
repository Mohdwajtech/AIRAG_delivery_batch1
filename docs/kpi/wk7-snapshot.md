# W7 KPI Snapshot

## Baseline Metrics (Winner Configuration)

| KPI              | Value     | Notes                                    |
|------------------|-----------|------------------------------------------|
| Hit rate @ top-3 | --/20     | Correct doc in top-3 results             |
| Cost / query     | $--       | Embedding API cost                       |
| Latency p50      | --ms      | End-to-end retrieval (embed + search)    |
| Latency p95      | --ms      |                                          |
| Embedding model  | text-embedding-3-small | 1536 dims                    |
| Chunking method  | sentence-aware         | Respects sentence boundaries |
| Vector store     | Qdrant (capstone_chunks) |                              |
| Index size       | -- points |                                          |

## Delta vs W6

| Metric       | W6       | W7       | Delta    |
|--------------|----------|----------|----------|
| Hit rate     | --/20    | --/20    | +/- --   |
| Cost/query   | $--      | $--      | --       |
| Latency p50  | --ms     | --ms     | --       |

## Key Observations
1. (fill in — e.g., "Embedding model choice moved hit rate by ±1-3 points")
2. (fill in — e.g., "Sentence-aware chunking consistently outperformed fixed-window")
3. (fill in — e.g., "Local model is viable for offline/privacy scenarios at -2 points")

## Next Steps
- W8: Upgrade ingestion pipeline (structure-aware chunking, metadata, PII)
- W9: Add hybrid retrieval (BM25 + dense) and cross-encoder reranking
