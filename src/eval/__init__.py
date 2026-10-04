"""Evaluation harness (W5 → W11).

- golden.py         : load + validate the golden set
- judge.py          : LLM-as-judge (rubric scoring)
- ragas_metrics.py  : four RAGAS metrics (W11) — offline proxy + live
- harness.py        : build eval rows — synthetic or live (W11)
- kpi.py            : 8-KPI dashboard engine (W11)
- critic_creator.py : critic-creator loop (W5)
- pairwise.py       : pairwise comparison (W5)
"""
