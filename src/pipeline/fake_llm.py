"""fake_llm.py — A no-API stand-in for the real LLM call.

Why this file exists
--------------------
During the W2 live session we want to demonstrate async patterns, retries,
and structured logging without burning OpenAI API quota — and we want
*reliable* failures to show retry-with-backoff in action.

This module exposes the same Pydantic shape as the real path and one
async function, ``fake_ask_llm``, which:

* sleeps for a random latency in ``[min_latency, max_latency]`` to simulate
  network I/O — that latency is what makes async parallelism visible;
* returns a Pydantic ``Answer`` with a canned response for a handful of
  keywords, falling back to a generic message;
* raises ``FakeLLMError`` at a configurable rate so we can teach retry.

For the lab learners swap to the real ``AsyncOpenAI`` client. The interface
(``Question`` → ``Answer``) is identical, so the rest of the pipeline
doesn't change.
"""

from __future__ import annotations

import asyncio
import random

from pydantic import BaseModel


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    question: str
    text: str
    cost_usd: float
    retries: int = 0
    confidence: float = 1.0
    sources: list[str] = []


class FakeLLMError(Exception):
    """Raised by ``fake_ask_llm`` when configured to fail (simulated transient error)."""


_CANNED: dict[str, str] = {
    "rag": (
        "RAG combines retrieval over a document corpus with an LLM, so answers "
        "are grounded in your source material rather than the model's training data."
    ),
    "vector": (
        "Vector databases power semantic search, RAG context retrieval, and "
        "recommendation systems by storing high-dimensional embeddings of text."
    ),
    "hallucin": (
        "LLMs hallucinate when they produce confident text that isn't grounded in "
        "evidence — usually because the prompt invites speculation or training "
        "coverage is thin."
    ),
    "pydantic": (
        "Pydantic gives you typed Python classes that validate data at runtime — "
        "ideal for checking LLM outputs match the shape your code expects."
    ),
    "async": (
        "Async Python lets one program work on many I/O-bound tasks at once: while "
        "awaiting a network call, the event loop runs other tasks instead of blocking."
    ),
    "embedding": (
        "An embedding is a fixed-length vector representation of text whose "
        "geometric distance correlates with semantic similarity."
    ),
    "agent": (
        "An agent is a loop that uses an LLM to decide what to do next — often "
        "combining tools, memory, and multi-step planning."
    ),
    "chunk": (
        "Chunking splits long documents into smaller passages so each one fits in "
        "the LLM's context and retrieval can return only the relevant pieces."
    ),
    "retriev": (
        "A retriever takes a user query, finds the most relevant chunks in the "
        "vector store, and passes them to the LLM as grounding context."
    ),
    "system prompt": (
        "A system prompt sets the model's role, style, and guardrails before it "
        "sees any user message — like a job description for the conversation."
    ),
    "temperature": (
        "Temperature controls randomness in the LLM's output: lower values are "
        "more focused and repeatable, higher values are more creative."
    ),
    "fine-tun": (
        "Fine-tuning teaches a model a stable style or skill from labelled examples; "
        "prefer it over RAG when the answer depends on *how* to respond, not what is known."
    ),
    "backoff": (
        "Exponential backoff doubles the wait between retries so a flaky API has "
        "time to recover and you don't hammer it with rapid-fire requests."
    ),
    "chatbot": (
        "A chatbot replies to a prompt; an agent decides what to do next in a loop "
        "and can call tools, take multi-step actions, and revise its plan."
    ),
    "gather": (
        "asyncio.gather runs many coroutines concurrently and waits for all of them "
        "to finish, returning their results in the original order."
    ),
    ".env": (
        "A .env file holds environment variables (like API keys) for local development. "
        "Load it with python-dotenv and never commit it — keep .env in .gitignore."
    ),
    "long-context": (
        "Long-context prompting trades cost and latency for simplicity: it works for "
        "small, self-contained content but doesn't scale to large or growing corpora."
    ),
    "structured log": (
        "Structured logs emit one JSON record per event so they can be filtered, "
        "aggregated, and fed into dashboards without regex parsing."
    ),
    "streaming": (
        "Streaming returns tokens as the model generates them; batching collects items "
        "and processes them together — streaming optimises perceived latency, batching "
        "optimises throughput."
    ),
    "sync": (
        "Sync I/O blocks the whole program until the call returns; async I/O yields to "
        "the event loop so other tasks can progress while one waits."
    ),
}


def _pick(text: str) -> str:
    """Return a canned answer if a keyword matches, otherwise a generic line."""
    lower = text.lower()
    for key, answer in _CANNED.items():
        if key in lower:
            return answer
    return f"(simulated answer) A grounded, concise response to: {text}"


async def fake_ask_llm(
    q: Question,
    *,
    min_latency: float = 0.3,
    max_latency: float = 1.5,
    fail_rate: float = 0.0,
) -> Answer:
    """Drop-in replacement for the real LLM call — no network, no API key.

    Parameters
    ----------
    q : Question
        The question to answer (same Pydantic model the real path uses).
    min_latency, max_latency : float
        Sleep range in seconds — simulates network jitter so async parallelism
        is visible.
    fail_rate : float
        Probability in ``[0.0, 1.0]`` of raising :class:`FakeLLMError`,
        simulating a transient API failure. Use this to teach
        retry-with-backoff during the live session.
    """
    await asyncio.sleep(random.uniform(min_latency, max_latency))
    if fail_rate > 0.0 and random.random() < fail_rate:
        raise FakeLLMError(f"simulated transient failure: {q.text[:50]}")
    return Answer(
        question=q.text,
        text=_pick(q.text),
        cost_usd=0.0001,
    )


if __name__ == "__main__":
    # Quick smoke test — `python fake_llm.py`
    async def _demo() -> None:
        questions = [
            Question(text="What is RAG in one sentence?"),
            Question(text="Why might an LLM hallucinate?"),
            Question(text="What does temperature do in an LLM call?"),
        ]
        for q in questions:
            a = await fake_ask_llm(q)
            print(f"Q: {q.text}\nA: {a.text}\n")

    asyncio.run(_demo())
