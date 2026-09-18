# ADR 0003 — Retrieval Architecture

## Status
Proposed (W7 draft). Revisited W9 (hybrid + reranking). Locks at M2 (W12).

## Date
Week 7

## Context
We need to choose an embedding model, chunking strategy, and vector store
for our RAG pipeline. The choices affect retrieval quality (hit rate),
operational cost (API spend), and latency (user-facing response time).

We ran controlled experiments in W7 measuring 5 combinations of embedding
model × chunking strategy on our 20-question golden set.

## Decision

### Embedding Model
**text-embedding-3-small** (OpenAI, 1536 dimensions, $0.02/M tokens)

### Chunking Strategy
**Sentence-aware** (respects sentence boundaries, ~400 char target)

### Vector Store
**Qdrant** (self-hosted or Qdrant Cloud, HNSW indexing, payload filtering)

## Alternatives Considered

### Embedding Models
| Model                    | Dims  | Cost       | Hit Rate vs Baseline | Verdict  |
|--------------------------|-------|------------|---------------------|----------|
| text-embedding-3-small   | 1536  | $0.02/M    | baseline            | CHOSEN   |
| text-embedding-3-large   | 3072  | $0.13/M    | +1-3 pts            | Rejected — 6.5× cost for marginal gain |
| all-MiniLM-L6-v2 (local) | 384   | $0.00      | −1-3 pts            | Viable backup for offline/privacy |

### Chunking Strategies
| Strategy        | Hit Rate vs Baseline | Notes                        | Verdict  |
|-----------------|---------------------|------------------------------|----------|
| Sentence-aware  | baseline            | Respects sentence boundaries | CHOSEN   |
| Fixed-400       | −1-2 pts            | Cuts mid-sentence            | Rejected |
| Fixed-200       | −2-3 pts            | More fragments, less context | Rejected |
| Structure-aware | (W8)                | Uses document headings       | Adopted in W8 |

## Consequences
- We accept dependency on OpenAI's embedding API for production use.
- We accept that text-embedding-3-small may underperform on semantically
  complex or domain-specific queries (mitigated by hybrid search in W9).
- We accept Qdrant as the vector store (mitigated by the abstraction in
  src/rag/ — swapping stores requires changing one module).
- Embedding model is a LOWER-PRIORITY optimisation target; retrieval
  strategy (hybrid, reranking) is HIGHER-PRIORITY.

## Open Questions (to revisit in W9–W12)
- Does hybrid search (BM25 + dense via RRF) change the model value ranking?
- Does cross-encoder reranking compensate for using the smaller model?
- Should we add Matryoshka dimensionality reduction (e.g., 512 dims)?
- Is the local model viable for data-sovereignty requirements?
