"""scripts/m2_check.py — verify M2 milestone deliverables (W12).

Checks items 3-5 (documents). Prints commands for items 1-2 (code).

Usage:
    python3 scripts/m2_check.py
"""
from pathlib import Path
import sys

CHECKS = [
    ("docs/kpi/wk12-snapshot.md", "8-KPI dashboard snapshot"),
    ("docs/adr/decisions.md", "ADR v2 consolidated decisions"),
]

# Also accept older ADR file names
ADR_ALTERNATIVES = [
    "docs/adr/0001-capstone-framing.md",
    "docs/adr/0002-pipeline-foundation.md",
    "docs/adr/0003-retrieval-architecture.md",
]


def main():
    print("=" * 50)
    print("  M2 Milestone Checker")
    print("=" * 50)
    print()

    all_ok = True

    # Check item 3: KPI snapshot
    snap = Path("docs/kpi/wk12-snapshot.md")
    if snap.exists():
        content = snap.read_text()
        kpi_count = sum(1 for i in range(1, 9) if f"{i}_" in content or f"KPI {i}" in content or f"#{i}" in content)
        if kpi_count >= 6:
            print(f"✓ {snap} found, ~{kpi_count} KPIs referenced")
        else:
            print(f"⚠ {snap} found but only ~{kpi_count} KPIs detected (need 8)")
            all_ok = False
    else:
        print(f"✗ {snap} NOT FOUND")
        all_ok = False

    # Check item 4: ADR
    adr_main = Path("docs/adr/decisions.md")
    if adr_main.exists():
        print(f"✓ {adr_main} found (consolidated ADR)")
    else:
        found_any = [p for p in ADR_ALTERNATIVES if Path(p).exists()]
        if found_any:
            print(f"⚠ No consolidated ADR at {adr_main}, but found: {', '.join(found_any)}")
            print(f"  → Consider consolidating into {adr_main} for M2")
        else:
            print(f"✗ No ADR files found")
            all_ok = False

    # Check item 5: DR #2 one-pager
    onepager_candidates = [
        Path("docs/dr2-one-pager.md"),
        Path("docs/DR2_one_pager.md"),
        Path("docs/dr2_defense.md"),
    ]
    found_op = next((p for p in onepager_candidates if p.exists()), None)
    if found_op:
        content = found_op.read_text()
        sections = sum(1 for heading in ["Problem", "Trade", "KPI", "Failure", "ADR"]
                       if heading.lower() in content.lower())
        print(f"✓ {found_op} found, ~{sections}/5 rubric sections detected")
    else:
        print(f"✗ DR #2 one-pager NOT FOUND (tried: {', '.join(str(p) for p in onepager_candidates)})")
        all_ok = False

    # Items 1-2: manual verification
    print()
    print("Items 1-2 (verify manually):")
    print("  1. API:  curl -s localhost:8000/health | python3 -m json.tool")
    print("  2. Eval: python3 -m pytest eval/ -q")

    print()
    if all_ok:
        print("✓ All document checks passed.")
    else:
        print("⚠ Some checks failed — see above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
