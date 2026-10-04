# Runbook: Document Update (W10)

## Steps
1. Tombstone the old version
2. Clear the semantic cache
3. Ingest the new version
4. Bump KB_VERSION if chunking/model changed

## Verification (3 signals)
- 3a: New content reachable
- 3b: Old content gone
- 3c: Tombstones queryable for audit

## NON-NEGOTIABLE
Cache MUST be cleared on EVERY document update.
