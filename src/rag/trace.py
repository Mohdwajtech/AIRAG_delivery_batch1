"""src/rag/trace.py — a readable trace of one RAG query (Week 12).

Trace reading is the production skill of W12. A trace records the input/output
of every pipeline stage for a single query: embed → cache → retrieve → rerank →
prompt → generate. Debug by scanning for the first stage whose output is wrong.

Dependency-free (stdlib only) so the debugging drill runs on any machine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class StageRecord:
    """One pipeline stage's input/output, as it would appear in a real trace."""
    stage: str                       # embed | cache | retrieve | rerank | prompt | generate
    summary: str                     # one-line human-readable headline
    detail: dict = field(default_factory=dict)
    ok: bool = True                  # False if the stage errored
    latency_ms: int = 0


@dataclass
class Trace:
    """The full trace of a single query."""
    question: str
    stages: list[StageRecord] = field(default_factory=list)
    answer: str = ""

    def add(self, stage: str, summary: str, detail: dict | None = None,
            ok: bool = True, latency_ms: int = 0) -> StageRecord:
        rec = StageRecord(stage, summary, detail or {}, ok, latency_ms)
        self.stages.append(rec)
        return rec

    def get(self, stage: str) -> StageRecord | None:
        for s in self.stages:
            if s.stage == stage:
                return s
        return None

    @property
    def total_latency_ms(self) -> int:
        return sum(s.latency_ms for s in self.stages)

    def to_json(self) -> str:
        return json.dumps(
            {"question": self.question,
             "stages": [asdict(s) for s in self.stages],
             "answer": self.answer,
             "total_latency_ms": self.total_latency_ms},
            indent=2,
        )

    def render(self) -> str:
        """Human-readable trace — scan for the outlier."""
        lines = [
            "─" * 68,
            f"TRACE · {self.question}",
            "─" * 68,
        ]
        for i, s in enumerate(self.stages, 1):
            flag = " " if s.ok else "✗"
            lines.append(f"[{i}] {flag} {s.stage.upper():9} {s.latency_ms:>5}ms  {s.summary}")
            for k, v in s.detail.items():
                text = v if isinstance(v, str) else json.dumps(v)
                if len(text) > 88:
                    text = text[:85] + "…"
                lines.append(f"        {k}: {text}")
        lines.append("─" * 68)
        lines.append(f"ANSWER: {self.answer}")
        lines.append(f"(total {self.total_latency_ms}ms)")
        return "\n".join(lines)
