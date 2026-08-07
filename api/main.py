"""api/main.py — REFERENCE for Week 3 Lab Step 1.

The completed version. Wraps the W2 async pipeline behind a stable HTTP
contract (Question/Answer with field names question/content) while delegating
to the W2 pipeline internally.

Run with:
    uvicorn api.main:app --reload --port 8000

Hit with curl:
    curl -X POST http://localhost:8000/ask_batched \
         -H "Content-Type: application/json" \
         -d '{"question": "What is RAG?"}'
"""
import asyncio
import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# W2 pipeline — the underlying engine
from src.pipeline.pipeline import ask_llm as _pipeline_ask_llm
from src.pipeline.pipeline import stream_answer as _pipeline_stream
from src.pipeline.pipeline import Question as _PipelineQuestion


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public W3 API models — locked in ADR 0002
# These are intentionally separate from the W2 internal models (which use
# `text` field names). The endpoint handlers translate between the two.
# ─────────────────────────────────────────────────────────────────────────────
class Question(BaseModel):
    """Public request shape — locked in ADR 0002."""
    question: str


class Answer(BaseModel):
    """Public response shape — locked in ADR 0002 (grown additively in W4)."""
    content: str
    cost_usd: float
    retries: int
    confidence: float = 1.0
    sources: list[str] = []


app = FastAPI(
    title="Capstone API",
    description="Wraps the W2 async pipeline. Contract locked in ADR 0002 (W3); internals upgraded W4+.",
    version="1.0.0",
)


# ─────────────────────────────────────────────────────────────────────────────
# /ask_batched — non-streaming reference endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ask_batched", response_model=Answer)
async def ask_batched(q: Question) -> Answer:
    """Non-streaming. Returns the full Answer in a single JSON body."""
    log.info("ask_batched  question=%r", q.question[:80])
    pipeline_q = _PipelineQuestion(text=q.question)
    pipeline_ans = await _pipeline_ask_llm(pipeline_q)
    return Answer(
        content=pipeline_ans.text,
        cost_usd=pipeline_ans.cost_usd,
        retries=pipeline_ans.retries,
        confidence=pipeline_ans.confidence,
        sources=pipeline_ans.sources,
    )


# ─────────────────────────────────────────────────────────────────────────────
# /health — liveness probe
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# /ask — streaming endpoint (the contracted one)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ask")
async def ask(q: Question):
    """Streaming /ask — now backed by the pipeline's REAL token stream (W4)."""
    log.info("ask  question=%r", q.question[:80])
    return StreamingResponse(
        _pipeline_stream(q.question),
        media_type="text/plain",
    )
