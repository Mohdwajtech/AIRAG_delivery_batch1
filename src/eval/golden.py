"""Golden-set loader + validator (Week 5).

A golden entry pairs a question with what a great answer looks like. Stored as
JSONL (one JSON object per line) so it appends cleanly and diffs cleanly.
"""
from __future__ import annotations
import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenEntry(BaseModel):
    """One golden-set row. Validated on load — a blank field fails loudly."""
    id:           str
    question:     str = Field(min_length=1)
    ideal_answer: str = Field(min_length=1)
    notes:           str = ""
    must_mention:    list[str] = []       # facts a good answer should contain
    expected_source: str = ""             # W6: corpus doc that should be retrieved (for hit rate)


def load_golden(path: str | Path = "data/golden_set.jsonl") -> list[GoldenEntry]:
    """Read a .jsonl file, validate every non-blank line, return the entries."""
    out: list[GoldenEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(GoldenEntry(**json.loads(line)))
    return out
