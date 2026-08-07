"""
practice_examples/week04/model_compare_cost_demo.py
============================================================================
TOPIC 4 OF WEEK 4: turn cost_usd from a placeholder into a REAL number using
response.usage, then compare gpt-4o-mini vs gpt-4o so a model choice is
defensible with numbers, not vibes.

Run ONE stage at a time:

    python practice_examples/week04/model_compare_cost_demo.py 1   # read usage off a call
    python practice_examples/week04/model_compare_cost_demo.py 2   # compute cost (offline)
    export OPENAI_API_KEY=sk-...
    python practice_examples/week04/model_compare_cost_demo.py 3   # same Q, two models
    python practice_examples/week04/model_compare_cost_demo.py 4   # several Qs, total spend
    python practice_examples/week04/model_compare_cost_demo.py 5   # judging QUALITY, not just cost

NOTES
- Stage 2 is offline (pure arithmetic) — its output is exact.
- Stages 1,3,4,5 make real calls; outputs are REPRESENTATIVE.
- RATES are $ per 1M tokens, REPRESENTATIVE. VERIFY AGAINST TODAY'S PRICING.
============================================================================
"""
import os
import sys

client = None  # constructed lazily in __main__

RATES = {
    "gpt-4o-mini": (0.15, 0.60),     # (input_rate, output_rate) per 1M tokens
    "gpt-4o":      (2.50, 10.00),
}


def compute_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    """cost = (input*in_rate + output*out_rate) / 1e6. This is your cost.py in miniature."""
    in_rate, out_rate = RATES[model]
    return (in_tokens * in_rate + out_tokens * out_rate) / 1_000_000


# ===========================================================================
# STAGE 01 — Read the real token usage off a response
# ---------------------------------------------------------------------------
# Every response carries a `usage` object — the BILLING TRUTH.
# ===========================================================================
def stage_01_read_usage():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is RAG in one sentence?"}],
    )
    u = resp.usage
    print("prompt_tokens     :", u.prompt_tokens)       # what you SENT
    print("completion_tokens :", u.completion_tokens)   # what the model WROTE
    print("total_tokens      :", u.total_tokens)

    # EXPECTED OUTPUT (representative)
    # prompt_tokens     : 15
    # completion_tokens : 24
    # total_tokens      : 39
    #
    # TAKEAWAY: you can't know completion_tokens until the model has written — so REAL cost is
    # an AFTER-the-call number. usage is where it comes from.


# ===========================================================================
# STAGE 02 — Compute the real cost (offline — from token counts)
# ---------------------------------------------------------------------------
# The formula, applied. No call needed; we feed in representative counts.
# ===========================================================================
def stage_02_compute_cost():
    in_tok, out_tok = 15, 24
    print(f"tokens: {in_tok} in / {out_tok} out\n")
    for model in RATES:
        print(f"{model:12s}: ${compute_cost(model, in_tok, out_tok):.8f}")

    # EXPECTED OUTPUT (exact arithmetic)
    # tokens: 15 in / 24 out
    # gpt-4o-mini : $0.00001665
    # gpt-4o      : $0.00027750
    #
    # TAKEAWAY: same tokens, ~16x the cost. This function IS your cost.py.compute_cost_usd().


# ===========================================================================
# STAGE 03 — Same question, two models, side by side
# ---------------------------------------------------------------------------
# The comparison, in miniature: cost + answer for mini vs 4o.
# ===========================================================================
def stage_03_two_models():
    q = "What is RAG in one sentence?"
    for model in ("gpt-4o-mini", "gpt-4o"):
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": q}])
        u = resp.usage
        cost = compute_cost(model, u.prompt_tokens, u.completion_tokens)
        print(f"{model:12s} | ${cost:.8f} | {resp.choices[0].message.content[:70]}")

    # EXPECTED OUTPUT (representative)
    # gpt-4o-mini  | $0.00001665 | RAG combines retrieval over a corpus with an LLM so answers...
    # gpt-4o       | $0.00027750 | Retrieval-augmented generation grounds an LLM's answer in...
    #
    # TAKEAWAY: both answers are good here. The cost differs ~16x. On an EASY question the cheap
    # model is fine — so why pay 16x? The next stage totals it up over several questions.


# ===========================================================================
# STAGE 04 — Several questions, total spend + ratio
# ---------------------------------------------------------------------------
# This is the seed of your Step-3 compare_models.py (which runs 10 questions).
# ===========================================================================
def stage_04_batch_compare():
    questions = ["What is RAG?", "What is a vector database?", "What is an embedding?"]
    totals = {"gpt-4o-mini": 0.0, "gpt-4o": 0.0}
    for q in questions:
        for model in ("gpt-4o-mini", "gpt-4o"):
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": q}])
            u = resp.usage
            totals[model] += compute_cost(model, u.prompt_tokens, u.completion_tokens)
    for model, t in totals.items():
        print(f"{model:12s}: ${t:.6f}  ({len(questions)} questions)")
    print(f"\ngpt-4o costs {totals['gpt-4o']/totals['gpt-4o-mini']:.0f}x more for the same run.")

    # EXPECTED OUTPUT (representative)
    # gpt-4o-mini : $0.000052  (3 questions)
    # gpt-4o      : $0.000840  (3 questions)
    # gpt-4o costs 16x more for the same run.
    #
    # TAKEAWAY: the ratio is consistent (~15-16x), the absolute numbers are tiny (your whole W4
    # lab spend is under ~15 cents). At SCALE that ratio is the difference between viable and not.


# ===========================================================================
# STAGE 05 — Cost is easy; QUALITY is the real question
# ---------------------------------------------------------------------------
# Cost is a number. Quality needs judgement — print both answers and EYEBALL them.
# ===========================================================================
def stage_05_quality():
    # A HARDER question, where the cheap model might actually fall short.
    q = ("A user asks for a 3-step migration plan from a keyword search system to RAG, "
         "with one risk per step. Answer concisely.")
    for model in ("gpt-4o-mini", "gpt-4o"):
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": q}])
        print(f"\n===== {model} =====")
        print(resp.choices[0].message.content)

    # EXPECTED OUTPUT (representative — you must READ these, not measure them)
    # ===== gpt-4o-mini =====
    # 1) Index your docs... risk: chunking...  2) Add a retriever... risk: recall...  3) ...
    # ===== gpt-4o =====
    # 1) ...(often more structured / covers risks more specifically)...
    #
    # TAKEAWAY: this is where you decide whether to ESCAPE to the expensive model. Cost you can
    # compute; quality you must JUDGE (and in W5 you'll build eval to judge it systematically).
    # The rule: default to mini; escape to 4o only when a comparison like this shows mini falling
    # short on YOUR questions — and record that decision, with the numbers, in the ADR.


# ===========================================================================
# FIT IT INTO YOUR APP  (the handoff)
# ---------------------------------------------------------------------------
# 1) Create src/pipeline/cost.py with the RATES table + compute_cost_usd() (Stage 2).
# 2) In ask_llm's REAL branch, after the call, replace the 0.0001 placeholder with real cost:
#        u = resp.usage
#        cost = compute_cost_usd(settings.model, u.prompt_tokens, u.completion_tokens)
#        return Answer(..., cost_usd=cost, ...)
# 3) VERIFY (real path + key):
#        curl -s -X POST localhost:8000/ask_batched -H 'Content-Type: application/json' \
#             -d '{"question":"What is RAG?"}'      # -> cost_usd is now a real, non-round number
# 4) Stages 3-4 ARE your scripts/compare_models.py (Step 3): run 10 questions through both models,
#    print cost + answers, and paste the verdict into your ADR.
#
# HOW IT HELPS OUR APP: cost_usd stops being fiction. Every answer records what it cost, which
# powers the Week-1 cost KPI and lets you DEFEND "why gpt-4o-mini" with a real comparison.
# ===========================================================================


STAGES = {
    "1": stage_01_read_usage, "2": stage_02_compute_cost, "3": stage_03_two_models,
    "4": stage_04_batch_compare, "5": stage_05_quality,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print("Run one stage at a time:\n")
        for k, fn in STAGES.items():
            print(f"  python {sys.argv[0]} {k}   # {fn.__name__}")
        print("\nStage 2 needs no key. Stages 1,3,4,5 make real calls.")
        sys.exit(0)
    if sys.argv[1] != "2":
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: set OPENAI_API_KEY first  ->  export OPENAI_API_KEY=sk-...")
            sys.exit(1)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    STAGES[sys.argv[1]]()
