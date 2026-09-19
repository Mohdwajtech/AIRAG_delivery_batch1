# PII Audit — W8

## Protocol
Sampled 10 random chunks from `capstone_chunks_v2` after ingestion.
Each chunk checked against the 7-item PII checklist.

## Checklist (per chunk)
- [ ] Names — any first+last name pairs?
- [ ] Emails — anything matching name@domain?
- [ ] Phone numbers — any digit sequences that could be phone numbers?
- [ ] Addresses — streets, postcodes, ZIPs?
- [ ] IDs — employee/customer IDs, SSN, passport, NI?
- [ ] DOBs — any date-of-birth patterns?
- [ ] Financial — card numbers, IBANs, account numbers?

## Findings

| # | Chunk ID | Source | PII Found? | Details |
|---|----------|--------|------------|---------|
| 1 |          |        | --         |         |
| 2 |          |        | --         |         |
| 3 |          |        | --         |         |
| 4 |          |        | --         |         |
| 5 |          |        | --         |         |
| 6 |          |        | --         |         |
| 7 |          |        | --         |         |
| 8 |          |        | --         |         |
| 9 |          |        | --         |         |
| 10|          |        | --         |         |

## Result
- Residual PII items found: -- / 10 sampled chunks
- Scrubber coverage: regex (EMAIL, PHONE, EMP_ID, SSN, CREDIT_CARD) + Presidio (PERSON, LOCATION)

## Improvements Made (if any)
- (fill in — e.g., "Added UK phone format regex after finding +44 number in chunk #3")
- (fill in — e.g., "Lowered Presidio PERSON threshold from 0.5 to 0.4 after missed name")

## Data Residency Stance
- **Embedding API:** OpenAI (US servers) — text transits to OpenAI for embedding
- **Vector store:** Qdrant Cloud (region: --)
- **Source documents:** local / Vocareum environment
- **Regulated data in scope?** No EU personal data or PHI in current corpus
- **Self-hosted alternative:** all-MiniLM-L6-v2 for embedding (no data leaves machine)
