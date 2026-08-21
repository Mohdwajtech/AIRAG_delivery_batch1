"""Evaluation harness (Week 5).

Sits *beside* the app: reads the app's answers and scores their quality.
- golden.py         : load + validate the golden set
- judge.py          : LLM-as-judge (rubric scoring)  [judge model = gpt-4o]
- pairwise.py       : pairwise comparison with position-flip
- critic_creator.py : the critic/creator improvement loop
"""
