# Failure Analysis — W9

## Protocol
Picked 5 queries where `hybrid_rerank` still fails (correct doc NOT in top-3).
Diagnosed root cause using the W9 decision table (8 rows).

## Decision Table Reference

| Row | Query pattern                           | Root cause                    | Fix                         |
|-----|-----------------------------------------|-------------------------------|-----------------------------|
| 1   | Contains exact codes, IDs, policy numbers | Rare-term smearing in dense   | BM25 + hybrid               |
| 2   | Uses different words than the corpus    | Vocabulary gap                | Multi-query paraphrases     |
| 3   | Very short (1–3 words)                  | Insufficient embedding context| HyDE hypothetical answer    |
| 4   | Asks to compare two things              | Single vector, two directions | Multi-query decomposition   |
| 5   | Hyper-specific (version, plan, date)    | Specificity mismatch          | Step-back generalisation    |
| 6   | Correct doc retrieved but at wrong rank | Bi-encoder can't distinguish  | Cross-encoder rerank        |
| 7   | Correct docs, wrong LLM answer         | Generation failure            | Better prompt / model       |
| 8   | Correct doc not in corpus               | Corpus gap                    | Add the document            |

## Failures

### Query 1: "___"
- Dense top-3: ✗ / ✓
- Hybrid top-3: ✗ / ✓
- Hybrid+Rerank top-3: ✗
- Correct doc: ___
- Retrieved instead: ___
- **Root cause:** Row _ — ___
- **Proposed fix:** ___

### Query 2: "___"
- Dense top-3:
- Hybrid top-3:
- Hybrid+Rerank top-3: ✗
- Correct doc:
- Retrieved instead:
- **Root cause:** Row _ — ___
- **Proposed fix:** ___

### Query 3: "___"
- Root cause: Row _ — ___
- Proposed fix: ___

### Query 4: "___"
- Root cause: Row _ — ___
- Proposed fix: ___

### Query 5: "___"
- Root cause: Row _ — ___
- Proposed fix: ___

## Distribution Summary

| Root cause                  | Count | % of failures |
|-----------------------------|-------|---------------|
| Row 1: Rare terms           |       |               |
| Row 2: Vocabulary mismatch  |       |               |
| Row 3: Short query          |       |               |
| Row 4: Multi-hop            |       |               |
| Row 5: Specificity mismatch |       |               |
| Row 6: Cross-reference      |       |               |
| Row 7: Generation failure   |       |               |
| Row 8: Corpus gap           |       |               |

## Top Observation
(fill in — e.g., "4 of 5 failures are Row 2 (vocabulary mismatch) →
multi-query is the highest-value Stretch technique for this corpus")

## Action Items
- [ ] Implement the highest-impact fix (based on distribution above)
- [ ] Re-run golden set after fix
- [ ] Document delta in docs/kpi/wk9-stretch.md (if Stretch attempted)
