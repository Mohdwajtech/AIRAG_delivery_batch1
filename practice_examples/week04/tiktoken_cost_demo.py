"""
practice_examples/week04/tiktoken_cost_demo.py
============================================================================
TOPIC 1 OF WEEK 4: tokens, cost, and why the OUTPUT half of the bill is the
one that bites. We count tokens LOCALLY with tiktoken (before any call), apply
the cost formula, then check our estimate against the real response.usage.

Run ONE stage at a time:

    python practice_examples/week04/tiktoken_cost_demo.py 1   # count tokens (no key)
    python practice_examples/week04/tiktoken_cost_demo.py 2   # tokens scale with length
    python practice_examples/week04/tiktoken_cost_demo.py 3   # the cost formula (input)
    python practice_examples/week04/tiktoken_cost_demo.py 4   # add output -> it dominates
    python practice_examples/week04/tiktoken_cost_demo.py 5   # same tokens, 15x cost across models
    export OPENAI_API_KEY=sk-...
    python practice_examples/week04/tiktoken_cost_demo.py 6   # tiktoken BEFORE vs usage AFTER

NOTES
- Stages 1-5 need NO api key (tiktoken counts locally). On first run tiktoken
  downloads its encoding file once (needs internet), then caches it.
- Rates below are REPRESENTATIVE ($ per 1M tokens). VERIFY AGAINST TODAY'S PRICING.
============================================================================
"""
import os
import sys

import tiktoken

client = None  # constructed lazily in __main__ (only stage 6 needs it)

# $ per 1,000,000 tokens — (input_rate, output_rate). VERIFY AGAINST TODAY'S DOCS.
RATES = {
    "gpt-4o-mini": (0.15, 0.60),     # cheap
    "gpt-4o":      (2.50, 10.00),    # ~15x mini
}


def compute_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    """cost = (input_tokens * input_rate + output_tokens * output_rate) / 1e6."""
    in_rate, out_rate = RATES[model]
    return (in_tokens * in_rate + out_tokens * out_rate) / 1_000_000


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# ===========================================================================
# STAGE 01 — Count the tokens in one string
# ---------------------------------------------------------------------------
# Everything is billed and limited in TOKENS, not words. Rule of thumb:
# ~4 characters of English per token.
# ===========================================================================
def stage_01_count():
    text = "What is RAG?"
    n = count_tokens(text)
    print(f"text      : {text!r}")
    print(f"characters: {len(text)}")
    print(f"tokens    : {n}")
    print(f"chars/token ~ {len(text)/n:.1f}")

    # EXPECTED OUTPUT (representative)
    # text      : 'What is RAG?'
    # characters: 12
    # tokens    : 4
    # chars/token ~ 3.0
    #
    # TAKEAWAY: the model counts in tokens (~4 chars each), not words. This is the
    # unit of cost, of context limits, and of latency. Think in tokens.


# ===========================================================================
# STAGE 02 — Tokens scale with length
# ---------------------------------------------------------------------------
# Count a short vs a long string and watch the ratio hold near ~4 chars/token.
# ===========================================================================
def stage_02_scale():
    short = "What is RAG?"
    long = ("Retrieval-augmented generation pairs a retrieval step over a document "
            "corpus with a large language model, so that generated answers are grounded "
            "in retrieved source text rather than the model's parametric memory. " * 5)
    for label, text in [("short", short), ("long", long)]:
        n = count_tokens(text)
        print(f"{label:5s}: {len(text):4d} chars -> {n:4d} tokens  (~{len(text)/n:.1f} chars/token)")

    # EXPECTED OUTPUT (representative)
    # short:   12 chars ->    4 tokens  (~3.0 chars/token)
    # long : 1030 chars ->  190 tokens  (~5.4 chars/token)
    #
    # TAKEAWAY: longer text = proportionally more tokens. You can ESTIMATE a prompt's
    # token cost from its length before you ever call — that's your budgeting lever.


# ===========================================================================
# STAGE 03 — The cost formula (input side)
# ---------------------------------------------------------------------------
# cost = tokens * rate / 1,000,000. Start with just the INPUT.
# ===========================================================================
def stage_03_cost_input():
    text = "Summarize the history of databases in two sentences."
    n = count_tokens(text)
    in_rate, _ = RATES["gpt-4o-mini"]
    cost = n * in_rate / 1_000_000
    print(f"prompt tokens : {n}")
    print(f"input rate    : ${in_rate} / 1M tokens")
    print(f"input cost    : ${cost:.8f}")

    # EXPECTED OUTPUT (representative)
    # prompt tokens : 10
    # input rate    : $0.15 / 1M tokens
    # input cost    : $0.00000150
    #
    # TAKEAWAY: one prompt is fractions of a cent. But this is only HALF the bill —
    # we haven't counted what the model WRITES back. Stage 4.


# ===========================================================================
# STAGE 04 — Add the output — and watch it DOMINATE
# ---------------------------------------------------------------------------
# Output tokens cost ~4x input tokens. A chatty answer is where the money goes.
# ===========================================================================
def stage_04_output_dominates():
    prompt = "Explain RAG."
    in_tok = count_tokens(prompt)
    print(f"prompt: {in_tok} input tokens\n")
    print(f"{'answer length':>16} | {'out tok':>7} | {'input $':>12} | {'output $':>12} | {'total $':>12}")
    for out_tok in (20, 200, 800):
        c = compute_cost("gpt-4o-mini", in_tok, out_tok)
        in_only = compute_cost("gpt-4o-mini", in_tok, 0)
        out_only = compute_cost("gpt-4o-mini", 0, out_tok)
        print(f"{str(out_tok)+' tokens':>16} | {out_tok:7d} | ${in_only:.8f} | ${out_only:.8f} | ${c:.8f}")

    # EXPECTED OUTPUT (representative)
    #    answer length | out tok |      input $ |     output $ |      total $
    #        20 tokens |      20 | $0.00000045 | $0.00001200 | $0.00001245
    #       200 tokens |     200 | $0.00000045 | $0.00012000 | $0.00012045
    #       800 tokens |     800 | $0.00000045 | $0.00048000 | $0.00048045
    #
    # TAKEAWAY: input cost is fixed and tiny; the bill is driven almost entirely by the
    # OUTPUT, which is ~4x the input rate. A verbose feature bleeds money on the output
    # side. Your biggest cost lever is often "answer in 2-4 sentences" in the prompt.


# ===========================================================================
# STAGE 05 — Same tokens, ~15x cost across models
# ---------------------------------------------------------------------------
# Model choice IS a cost decision. Compare mini vs 4o for identical usage.
# ===========================================================================
def stage_05_model_spread():
    in_tok, out_tok = 500, 500
    print(f"assume {in_tok} input + {out_tok} output tokens\n")
    for model in RATES:
        c = compute_cost(model, in_tok, out_tok)
        print(f"{model:12s}: ${c:.6f}")
    mini = compute_cost("gpt-4o-mini", in_tok, out_tok)
    big = compute_cost("gpt-4o", in_tok, out_tok)
    print(f"\ngpt-4o is {big/mini:.0f}x the cost of gpt-4o-mini for the same work.")

    # EXPECTED OUTPUT (representative)
    # assume 500 input + 500 output tokens
    # gpt-4o-mini : $0.000375
    # gpt-4o      : $0.006250
    # gpt-4o is 17x the cost of gpt-4o-mini for the same work.
    #
    # TAKEAWAY: identical work, ~15-17x the price. Default to the cheap model; reach for
    # the expensive one only when you can show it earns its keep (Day 2's model comparison).


# ===========================================================================
# STAGE 06 — tiktoken BEFORE vs response.usage AFTER  (needs a key)
# ---------------------------------------------------------------------------
# tiktoken is your ESTIMATE before the call (a guardrail). response.usage is the
# TRUTH after the call (what you bill). Compare them.
# ===========================================================================
def stage_06_estimate_vs_truth():
    prompt = "What is RAG in one sentence?"
    est_in = count_tokens(prompt)                     # BEFORE: local estimate
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    u = resp.usage                                    # AFTER: the truth
    print(f"tiktoken estimate (input) : {est_in}")
    print(f"usage.prompt_tokens        : {u.prompt_tokens}")
    print(f"usage.completion_tokens    : {u.completion_tokens}")
    print(f"usage.total_tokens         : {u.total_tokens}")
    real_cost = compute_cost("gpt-4o-mini", u.prompt_tokens, u.completion_tokens)
    print(f"real cost (from usage)     : ${real_cost:.8f}")

    # EXPECTED OUTPUT (representative)
    # tiktoken estimate (input) : 8
    # usage.prompt_tokens        : 8          <- matches the estimate (usually)
    # usage.completion_tokens    : 22
    # usage.total_tokens         : 30
    # real cost (from usage)     : $0.00001440
    #
    # TAKEAWAY: tiktoken lets you estimate INPUT cost before calling (reject a too-long
    # prompt, budget a batch). But only usage tells you the OUTPUT tokens — you can't know
    # those until the model has written. So: tiktoken to GUARD before, usage to BILL after.


# ===========================================================================
# FIT IT INTO YOUR APP  (the handoff)
# ---------------------------------------------------------------------------
# This is why Answer has a cost_usd field. On Day 2 you'll make it REAL:
#
# 1) Add src/pipeline/cost.py with RATES + compute_cost_usd() (Stages 3-5 above).
# 2) In ask_llm's real branch, after the call, read usage and compute the cost:
#        u = resp.usage
#        cost = compute_cost_usd(settings.model, u.prompt_tokens, u.completion_tokens)
#        return Answer(..., cost_usd=cost, ...)     # replaces the 0.0001 placeholder
# 3) VERIFY:
#        USE_FAKE unset + key, then:
#        curl -s -X POST localhost:8000/ask_batched -H 'Content-Type: application/json' \
#             -d '{"question":"What is RAG?"}'
#        -> cost_usd is now a REAL number derived from usage, not a placeholder.
#
# HOW IT HELPS OUR APP: cost_usd stops being a fiction. Every answer now carries what it
# actually cost, which feeds the Week-1 cost KPI and the Day-2 two-model comparison
# (mini vs 4o) where you defend a model choice with real numbers.
# ===========================================================================


STAGES = {
    "1": stage_01_count, "2": stage_02_scale, "3": stage_03_cost_input,
    "4": stage_04_output_dominates, "5": stage_05_model_spread, "6": stage_06_estimate_vs_truth,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print("Run one stage at a time:\n")
        for k, fn in STAGES.items():
            print(f"  python {sys.argv[0]} {k}   # {fn.__name__}")
        print("\nStages 1-5 need no key (tiktoken counts locally). Stage 6 makes ONE real call.")
        sys.exit(0)
    if sys.argv[1] == "6":
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: stage 6 needs a key  ->  export OPENAI_API_KEY=sk-...")
            sys.exit(1)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    STAGES[sys.argv[1]]()
