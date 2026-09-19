# W8 KPI Snapshot

## Metrics (capstone_chunks_v2)

| KPI              | Value     | Delta vs W7 | Notes                                |
|------------------|-----------|-------------|--------------------------------------|
| Hit rate @ top-3 | --/20     | +-- pts     | Structure-aware chunking             |
| Cost / query     | $--       | --          | Same embedding model                 |
| Latency p50      | --ms      | --          |                                      |
| Chunks created   | --        | N/A         | capstone_chunks_v2 collection        |
| PII-flagged      | -- chunks | N/A         | Scrubbed via regex + Presidio        |
| Metadata fields  | 11        | N/A         | Per chunk in Qdrant payload          |

## Ingestion Pipeline Stats

| Metric           | Value     |
|------------------|-----------|
| Documents parsed | --        |
| Parse failures   | --        |
| Total chunks     | --        |
| Embed cost       | $--       |
| Ingest time      | --s       |

## Key Observations
1. (fill in — e.g., "Structure-aware chunking improved coherence vs W7 sliding window")
2. (fill in — e.g., "PII scrubber caught N emails, M phone numbers, K person names")
3. (fill in — e.g., "Metadata filtering by doc_type narrowed search results meaningfully")

## PII Audit Summary
- Sampled: 10 random chunks
- Residual PII found: 0 / 10
- See docs/pii-audit.md for details

## Decisions Documented
- Parsing: PyMuPDF for PDF, BeautifulSoup for HTML, python-docx for DOCX
- Chunking: structure-aware (headings) with recursive fallback
- PII: regex (EMAIL, PHONE, EMP_ID) + Presidio (PERSON, LOCATION)
- Residency: embeddings via OpenAI API (US), vectors in Qdrant Cloud
