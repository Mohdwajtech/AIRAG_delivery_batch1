#!/usr/bin/env bash
# scripts/smoke_test.sh — CI smoke test for the eval framework (W11).
#
# Runs offline (EVAL_OFFLINE=1) — no API key, no cost, deterministic.
# Usage:
#     bash scripts/smoke_test.sh

set -euo pipefail

export EVAL_OFFLINE=1

echo "=== W11 Smoke Test ==="
echo ""

echo "1. Golden set validates..."
python3 -c "
from src.eval.golden import load_golden
g = load_golden('data/golden_set.jsonl')
assert len(g) >= 10, f'Expected ≥10 entries, got {len(g)}'
print(f'   ✓ {len(g)} entries loaded and validated')
"

echo "2. Eval rows build (offline)..."
python3 -c "
from src.eval.harness import build_rows, active_kb_version
rows = build_rows()
assert len(rows) >= 10, f'Expected ≥10 rows, got {len(rows)}'
print(f'   ✓ {len(rows)} rows built, kb_version={active_kb_version()}')
"

echo "3. Four RAGAS metrics score (offline proxy)..."
python3 -c "
from src.eval.harness import build_rows
from src.eval.ragas_metrics import score_all
rows = build_rows()
results = score_all(rows)
for name, r in results.items():
    assert 0 <= r.aggregate <= 1, f'{name} out of range: {r.aggregate}'
    print(f'   ✓ {name}={r.aggregate:.3f} ({r.mode})')
"

echo "4. 8-KPI dashboard computes..."
python3 -c "
from src.eval.harness import build_rows, active_kb_version
from src.eval.kpi import compute_kpis, reportable_count
rows = build_rows()
out = compute_kpis(rows, active_kb_version())
numeric, na = reportable_count(out['kpis'])
assert numeric == 6 and na == 2, f'Expected 6+2, got {numeric}+{na}'
print(f'   ✓ {numeric} numeric + {na} N/A KPIs')
"

echo ""
echo "=== All steps passed ✓ ==="
