"""
practice_examples/week04/streaming_demo.py
============================================================================
TOPIC 3 OF WEEK 4: real streaming. Replace the W3 simulation (a finished answer
sliced into words) with real OpenAI streaming (stream=True + delta.content) —
the same async-generator shape, a new source of chunks.

Run ONE stage at a time (needs OPENAI_API_KEY — real calls, tiny):

    export OPENAI_API_KEY=sk-...
    python practice_examples/week04/streaming_demo.py 1   # non-streaming baseline
    python practice_examples/week04/streaming_demo.py 2   # stream=True
    python practice_examples/week04/streaming_demo.py 3   # dissect the chunks / deltas
    python practice_examples/week04/streaming_demo.py 4   # robust consume + rebuild
    python practice_examples/week04/streaming_demo.py 5   # usage while streaming
    python practice_examples/week04/streaming_demo.py 6   # TTFT vs total (why streaming wins UX)

All outputs are REPRESENTATIVE (streaming is a real call). The STRUCTURE is exact.
============================================================================
"""
import os
import sys
import time

client = None  # constructed lazily in __main__
MODEL = "gpt-4o-mini"


# ===========================================================================
# STAGE 01 — Non-streaming baseline (where we are)
# ---------------------------------------------------------------------------
# One request, ONE response object, the full text at choices[0].message.content.
# ===========================================================================
def stage_01_non_streaming():
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    )
    print("ONE object arrives, all at once:")
    print(resp.choices[0].message.content)

    # EXPECTED OUTPUT (representative)
    # ONE object arrives, all at once:
    # 1, 2, 3, 4, 5
    #
    # TAKEAWAY: the user waits for the WHOLE answer, then sees it. Fine for /ask_batched.
    # For /ask we want it to appear as it's generated. That's streaming.


# ===========================================================================
# STAGE 02 — Turn on streaming
# ---------------------------------------------------------------------------
# Add stream=True. Now you get an ITERATOR of chunks, each with a piece.
# ===========================================================================
def stage_02_streaming():
    stream = client.chat.completions.create(
        model=MODEL, stream=True,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    )
    print("pieces arrive one at a time:")
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(repr(chunk.choices[0].delta.content))

    # EXPECTED OUTPUT (representative)
    # pieces arrive one at a time:
    # '1'
    # ','
    # ' 2'
    # ','
    # ' 3'
    # ...
    #
    # TAKEAWAY: instead of one object you loop over many chunks. Each carries a small piece at
    # chunk.choices[0].delta.content. Note: delta (a fragment), not message (the whole thing).


# ===========================================================================
# STAGE 03 — Dissect the chunks: the role-only first chunk, the empty tail
# ---------------------------------------------------------------------------
# Print the raw structure of the first several chunks so the guards make sense.
# ===========================================================================
def stage_03_dissect():
    stream = client.chat.completions.create(
        model=MODEL, stream=True,
        messages=[{"role": "user", "content": "Say hi."}],
    )
    for i, chunk in enumerate(stream):
        has_choices = bool(chunk.choices)
        c = chunk.choices[0] if has_choices else None
        content = c.delta.content if c else None
        finish = c.finish_reason if c else None
        print(f"chunk {i:2d}: choices={has_choices} content={content!r} finish={finish}")

    # EXPECTED OUTPUT (representative)
    # chunk  0: choices=True content=None finish=None      <- FIRST chunk is role-only: content is None
    # chunk  1: choices=True content='Hi' finish=None
    # chunk  2: choices=True content='!' finish=None
    # chunk  3: choices=True content=None finish='stop'    <- LAST content chunk: finish_reason set
    #
    # WHAT TO NOTICE
    # - The very first chunk often has delta.content = None (it just announces role='assistant').
    # - Some chunks can have empty choices entirely.
    # - The final chunk sets finish_reason and carries no content.
    # => you must GUARD against None/empty before using delta.content. Stage 4.


# ===========================================================================
# STAGE 04 — Robust consume + rebuild the full string
# ---------------------------------------------------------------------------
# The two guards, plus concatenating deltas back into the whole answer.
# ===========================================================================
def stage_04_robust():
    stream = client.chat.completions.create(
        model=MODEL, stream=True,
        messages=[{"role": "user", "content": "Explain RAG in about 20 words."}],
    )
    pieces = []
    for chunk in stream:
        if not chunk.choices:                 # GUARD 1: skip empty-choices chunks
            continue
        delta = chunk.choices[0].delta
        if delta.content:                     # GUARD 2: skip role-only / None-content chunks
            pieces.append(delta.content)
            print(delta.content, end="", flush=True)   # live, as it arrives
    full = "".join(pieces)                     # rebuild the whole answer
    print("\n\nrebuilt full answer:", repr(full))

    # EXPECTED OUTPUT (representative)
    # RAG retrieves relevant documents and feeds them to an LLM so its answer is grounded...
    # rebuilt full answer: 'RAG retrieves relevant documents and feeds them to an LLM...'
    #
    # THIS IS THE PATTERN your real stream_answer() uses: guard, yield/collect delta.content,
    # concatenate. The async version just uses `async for` and `yield` instead of a list.


# ===========================================================================
# STAGE 05 — Usage while streaming
# ---------------------------------------------------------------------------
# Streaming responses don't include usage by default. Ask for it explicitly.
# ===========================================================================
def stage_05_usage():
    stream = client.chat.completions.create(
        model=MODEL, stream=True,
        stream_options={"include_usage": True},   # <-- adds a final usage-only chunk
        messages=[{"role": "user", "content": "Say hi."}],
    )
    usage = None
    for chunk in stream:
        if getattr(chunk, "usage", None):          # the LAST chunk carries usage
            usage = chunk.usage
    print("final usage chunk:", usage)

    # EXPECTED OUTPUT (representative)
    # final usage chunk: CompletionUsage(completion_tokens=9, prompt_tokens=11, total_tokens=20)
    #
    # WHY: without stream_options include_usage, a streamed response gives you no token counts
    # (you'd have to estimate with tiktoken). With it, a final usage-only chunk arrives after the
    # content — so you can compute REAL cost even on the streaming path.


# ===========================================================================
# STAGE 06 — TTFT vs total: why streaming wins the UX battle
# ---------------------------------------------------------------------------
# Time the FIRST token vs the whole answer. Users feel the first.
# ===========================================================================
def stage_06_ttft():
    start = time.time()
    stream = client.chat.completions.create(
        model=MODEL, stream=True,
        messages=[{"role": "user", "content": "Explain RAG in about 60 words."}],
    )
    ttft = None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.time() - start        # time to FIRST token
            print(chunk.choices[0].delta.content, end="", flush=True)
    total = time.time() - start
    print(f"\n\nTTFT = {ttft:.2f}s   total = {total:.2f}s")

    # EXPECTED OUTPUT (representative)
    # (words stream in) ...
    # TTFT = 0.45s   total = 2.10s
    #
    # TAKEAWAY: the user sees the answer START in ~0.5s even though it FINISHES in ~2s. That
    # "it started fast" feeling is why streaming wins UX even when total time is identical to a
    # batched call. This is the whole reason /ask streams.


# ===========================================================================
# FIT IT INTO YOUR APP  (the handoff)
# ---------------------------------------------------------------------------
# Replace the SIMULATED stream_answer in src/pipeline/pipeline.py with this real loop.
#
#   BEFORE (W4-Day1, simulated): get the whole answer, then yield words with a sleep.
#   AFTER  (real): call with stream=True and yield delta.content as it arrives:
#
#     async def stream_answer(question, settings=None):
#         settings = settings or Settings()
#         if settings.use_fake:                       # keep the fork: fake still simulates
#             ans = await ask_llm(Question(question=question), settings)
#             for w in ans.content.split(" "):
#                 yield w + " "; await asyncio.sleep(0.05)
#             return
#         client = AsyncOpenAI(api_key=settings.openai_api_key or None)
#         stream = await client.chat.completions.create(
#             model=settings.model, stream=True,
#             messages=[{"role": "user", "content": question}])
#         async for chunk in stream:                  # async version of Stage 4
#             if not chunk.choices:
#                 continue
#             delta = chunk.choices[0].delta
#             if delta.content:
#                 yield delta.content
#
# VERIFY (the --no-buffer flag is essential or curl hides the streaming):
#     USE_FAKE=1 ... curl -s --no-buffer -X POST localhost:8000/ask ...   # simulated flow
#     OPENAI_API_KEY=sk-... USE_FAKE=0 ... curl -s --no-buffer ...        # REAL tokens streaming in
#
# HOW IT HELPS OUR APP: /ask goes from a convincing fake to real token-by-token output — the
# same generator shape, real chunks — which is the UX win TTFT buys (Stage 6).
# ===========================================================================


STAGES = {
    "1": stage_01_non_streaming, "2": stage_02_streaming, "3": stage_03_dissect,
    "4": stage_04_robust, "5": stage_05_usage, "6": stage_06_ttft,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print("Run one stage at a time:\n")
        for k, fn in STAGES.items():
            print(f"  python {sys.argv[0]} {k}   # {fn.__name__}")
        print("\nAll stages make a real streaming call. Set OPENAI_API_KEY first.")
        sys.exit(0)
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY first  ->  export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    STAGES[sys.argv[1]]()
