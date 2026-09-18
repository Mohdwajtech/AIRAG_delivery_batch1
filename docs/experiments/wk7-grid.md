# W7 Experiment Grid — Embedding × Chunker

## Purpose
Controlled experiments measuring how embedding model and chunking strategy
affect retrieval quality on our capstone corpus.

## Setup
- **Corpus:** 5 markdown documents, ~20 chunks
- **Golden set:** 20 questions with expected answers
- **Metric:** Hit rate @ top-3 (correct doc in top-3 results)
- **Control:** Change ONE variable per experiment row

## Experiment Results

| # | Embedding Model           | Dims  | Chunker         | Hit Rate | Cost/Query | Latency p50 | Notes                          |
|---|---------------------------|-------|-----------------|----------|------------|-------------|--------------------------------|
| 1 | text-embedding-3-small    | 1536  | sentence-aware  | --/20    | $--        | --ms        | **BASELINE** (W6 config)       |
| 2 | text-embedding-3-large    | 3072  | sentence-aware  | --/20    | $--        | --ms        | 6.5× cost, marginal quality ↑  |
| 3 | all-MiniLM-L6-v2 (local)  | 384   | sentence-aware  | --/20    | $0.00      | --ms        | Free, lower dims               |
| 4 | text-embedding-3-small    | 1536  | fixed-400       | --/20    | $--        | --ms        | Boundary-blind baseline        |
| 5 | text-embedding-3-small    | 1536  | fixed-200       | --/20    | $--        | --ms        | Smaller chunks, more fragments |

## Analysis

### Embedding Model Effect (rows 1 vs 2 vs 3, chunker held constant)
- small → large: +/- -- hit-rate points at 6.5× cost
- small → local: +/- -- hit-rate points at $0 cost
- **Observation:** (fill in after running experiments)

### Chunker Effect (rows 1 vs 4 vs 5, model held constant)
- sentence-aware vs fixed-400: +/- -- hit-rate points
- fixed-400 vs fixed-200: +/- -- hit-rate points
- **Observation:** (fill in after running experiments)

### Interaction Effect
- Does model choice matter MORE with fixed chunking or sentence chunking?
- **Observation:** (fill in after running experiments)

## Decision
**Winner:** (fill in — expected: text-embedding-3-small + sentence-aware)

**Rationale:** (fill in — expected: lowest cost within 1-2 hit-rate points of best)

## Null Findings
Document findings where the change made NO difference:
- (fill in — e.g., "Increasing dims from 1536 to 3072 added <1 point")
