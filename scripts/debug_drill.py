#!/usr/bin/env python3
"""scripts/debug_drill.py — CLI for the W12 debugging drill.

Usage:
    python3 scripts/debug_drill.py list              # list all scenarios
    python3 scripts/debug_drill.py run S1             # run + show trace
    python3 scripts/debug_drill.py grade S1 retrieval # check your guess
    python3 scripts/debug_drill.py answers            # full answer key
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.debug.scenarios import SCENARIOS, get, grade


def cmd_list():
    print("W12 Debugging Drill — Six Scenarios\n")
    for s in SCENARIOS:
        print(f"  {s.id}  {s.title}")
        print(f"       symptom: {s.symptom}\n")


def cmd_run(scenario_id: str):
    s = get(scenario_id)
    print(f"### {s.id} — {s.title}  (question {s.question_id})")
    print(f"SYMPTOM: {s.symptom}\n")
    print("(Run this scenario in the notebook to see the full trace.)")
    print(f"\nClassify: is this retrieval, generation, prompt, or infra?")
    print(f"Check:    python3 scripts/debug_drill.py grade {s.id} <your_guess>")


def cmd_grade(scenario_id: str, guess: str):
    ok, msg = grade(scenario_id, guess)
    print(msg)
    return 0 if ok else 1


def cmd_answers():
    print("W12 Debugging Drill — Answer Key\n")
    for s in SCENARIOS:
        print(f"{s.id} — {s.title}")
        print(f"   bucket:     {s.bucket}")
        print(f"   trace tell: {s.trace_tell}")
        print(f"   fix:        {s.fix}\n")


def main():
    args = sys.argv[1:]
    if not args or args[0] == "list":
        cmd_list()
    elif args[0] == "run" and len(args) >= 2:
        cmd_run(args[1].upper())
    elif args[0] == "grade" and len(args) >= 3:
        sys.exit(cmd_grade(args[1].upper(), args[2]))
    elif args[0] == "answers":
        cmd_answers()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
