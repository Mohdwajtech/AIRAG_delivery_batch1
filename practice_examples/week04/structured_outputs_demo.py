"""
practice_examples/week04/structured_outputs_demo.py
============================================================================
THE HEADLINE OF WEEK 4: three ways to force the model to return STRUCTURED
output instead of free text — and why tool-calling is the default.

Run ONE stage at a time:

    export OPENAI_API_KEY=sk-...
    python practice_examples/week04/structured_outputs_demo.py 1   # parse (happy path)
    python practice_examples/week04/structured_outputs_demo.py 2   # parse BREAKS (no key needed)
    python practice_examples/week04/structured_outputs_demo.py 3   # JSON mode
    python practice_examples/week04/structured_outputs_demo.py 4   # tool-calling
    python practice_examples/week04/structured_outputs_demo.py 5   # dissect the tool-call parse
    python practice_examples/week04/structured_outputs_demo.py 6   # multi-shape routing

NOTES
- Real-call outputs are REPRESENTATIVE (structure is real; text/numbers vary).
- Stage 2 makes NO api call (it's deterministic) so you can always see the failure.
============================================================================
"""
import json
import os
import sys

from openai import OpenAI

client = None  # constructed lazily in __main__ so the menu / stage 2 work without a key
MODEL = "gpt-4o-mini"

# The question we ask throughout. Note it lists the fields we want.
Q = "What is RAG? Give fields: content, confidence (0-1), sources (list of strings)."


# ===========================================================================
# STAGE 01 — Approach 1: ask for JSON in the prompt, then json.loads (happy path)
# ---------------------------------------------------------------------------
# The natural first instinct: tell the model "return JSON", then parse the text.
# It WORKS when the model behaves — which is exactly what makes it dangerous.
# ===========================================================================
def stage_01_parse_happy():
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": Q + " Return ONLY JSON."}],
    )
    raw = resp.choices[0].message.content
    print("raw text from model:\n", raw)
    # The fragile step — wrapped in try/except so the LESSON lands either way.
    try:
        data = json.loads(raw)
        print("\nparsed dict:", data)
        print("content:", data.get("content", "")[:60])
        print("\n=> It happened to parse THIS time. But you are one code-fence away from Stage 2.")
    except json.JSONDecodeError as e:
        print("\njson.loads FAILED ->", e)
        print("=> The model wrapped the JSON (```json ...``` or a preamble). THIS is the "
              "brittleness — you did not control whether it happened. On to Stages 3 & 4.")

    # EXPECTED OUTPUT (representative — you will see ONE of these two)
    #
    # (A) the model returned bare JSON:
    #   parsed dict: {'content': 'RAG combines...', 'confidence': 0.9, 'sources': []}
    #   => It happened to parse THIS time...
    #
    # (B) the model wrapped it in a markdown code fence (VERY common):
    #   raw text from model:
    #    ```json
    #    { "content": "RAG stands for...", "confidence": 0.95, "sources": [...] }
    #    ```
    #   json.loads FAILED -> Expecting value: line 1 column 1 (char 0)
    #   => The model wrapped the JSON...
    #
    # TAKEAWAY: the model OFTEN wraps JSON in ```json ... ``` (or a preamble). json.loads
    # sees the backtick before the '{' and dies at char 0. You do not control which way it
    # breaks. People "fix" it with .strip("`") / removeprefix("json") / regex — and now they
    # maintain a parser for every way a model might decorate output. That trap is the whole
    # argument for Stages 3 (json mode: no fence) and 4 (tool-calling: no text at all).


# ===========================================================================
# STAGE 02 — Approach 1 BREAKS (deterministic — no API call)
# ---------------------------------------------------------------------------
# Sometimes the model is "helpful" and wraps the JSON in prose. Here is exactly
# what that looks like, and exactly how json.loads reacts.
# ===========================================================================
def stage_02_parse_breaks():
    # This is a real shape the model sometimes returns:
    model_reply = 'Sure! Here is the JSON you asked for:\n{"content": "RAG is...", "confidence": 0.9, "sources": []}'
    print("model reply (note the preamble):\n", model_reply, "\n")
    try:
        data = json.loads(model_reply)
        print("parsed:", data)
    except json.JSONDecodeError as e:
        print("json.loads FAILED ->", e)

    # EXPECTED OUTPUT (verified — this stage is deterministic)
    # model reply (note the preamble):
    #  Sure! Here is the JSON you asked for:
    # {"content": "RAG is...", "confidence": 0.9, "sources": []}
    #
    # json.loads FAILED -> Expecting value: line 1 column 1 (char 0)
    #
    # WHY: json.loads sees the letter 'S' (of "Sure") before the opening brace and dies.
    # The JSON was perfectly valid — the model just wrapped it in politeness. In production
    # this fails INTERMITTENTLY (only when the model gets chatty), so it passes every test
    # and breaks in the wild. The common "fix" is a regex to strip preamble — brittle in a
    # new way. The real fix is to stop parsing free text: Stages 3 and 4.


# ===========================================================================
# STAGE 03 — Approach 2: JSON mode (provider guarantees parseable JSON)
# ---------------------------------------------------------------------------
# One parameter, response_format={"type":"json_object"}, and the body is
# guaranteed to be valid JSON — no preamble possible.
# ===========================================================================
def stage_03_json_mode():
    resp = client.chat.completions.create(
        model=MODEL,
        # RULE: with json_object, your messages MUST contain the word "json" somewhere,
        # or the API rejects the request with a 400. We add it explicitly here.
        messages=[{"role": "user", "content": Q + " Respond as a JSON object."}],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)   # always parses now
    print("parsed dict:", data)

    # EXPECTED OUTPUT (representative)
    # parsed dict: {'content': 'RAG combines retrieval with generation...', 'confidence': 0.88, 'sources': []}
    #
    # WHAT CHANGED / CAVEATS
    # - json.loads now succeeds EVERY time — the preamble problem is gone.
    # - BUT json mode guarantees JSON-*ness*, not your SCHEMA. The model could return valid
    #   JSON with the WRONG fields. So you still validate the shape yourself (e.g. Pydantic).
    # - The "must contain 'json'" rule is a guardrail: it stops you enabling json mode while
    #   asking for prose (which would trap the model). Forget it and you get:
    #     BadRequestError 400: 'messages' must contain the word 'json' ...


# ===========================================================================
# STAGE 04 — Approach 3: tool-calling (declare the schema; provider enforces it)
# ---------------------------------------------------------------------------
# Instead of asking for a shape and hoping, we DECLARE it as a tool and force the
# model to call it. The provider enforces the schema at its own level.
# ===========================================================================
ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "answer_question",
        "description": "Return a structured answer with content, confidence, and sources.",
        "parameters": {
            "type": "object",
            "properties": {
                "content":    {"type": "string"},
                "confidence": {"type": "number"},
                "sources":    {"type": "array", "items": {"type": "string"}},
            },
            "required": ["content", "confidence", "sources"],
        },
    },
}


def stage_04_tool_calling():
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is RAG?"}],   # no "return JSON" needed
        tools=[ANSWER_TOOL],
        tool_choice={"type": "function", "function": {"name": "answer_question"}},
    )
    args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    print("structured args:", args)
    print("confidence (from the MODEL):", args["confidence"])

    # EXPECTED OUTPUT (representative)
    # structured args: {'content': 'RAG combines retrieval over a corpus with an LLM...',
    #                   'confidence': 0.9, 'sources': []}
    # confidence (from the MODEL): 0.9
    #
    # WHY THIS WINS
    # - We didn't ask for JSON in the prose at all — we DECLARED the shape as a schema and
    #   forced the call with tool_choice. The provider guarantees the args fit the schema.
    # - No preamble is even possible: we get structured function ARGUMENTS, not a text message.
    # - This is exactly what your app's ask_llm() real branch does.


# ===========================================================================
# STAGE 05 — Dissect the tool-call parse (the line everyone trips on)
# ---------------------------------------------------------------------------
# Walk resp.choices[0].message.tool_calls[0].function.arguments piece by piece.
# ===========================================================================
def stage_05_parse_dissected():
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is a vector database?"}],
        tools=[ANSWER_TOOL],
        tool_choice={"type": "function", "function": {"name": "answer_question"}},
    )
    msg = resp.choices[0].message
    print("1) message.content      :", msg.content)            # usually None with tools
    print("2) message.tool_calls    :", type(msg.tool_calls), "len", len(msg.tool_calls))
    call = msg.tool_calls[0]
    print("3) tool_calls[0].id      :", call.id)
    print("4) .function.name        :", call.function.name)    # 'answer_question'
    print("5) .function.arguments   :", repr(call.function.arguments))  # a JSON *string*
    args = json.loads(call.function.arguments)                 # -> dict
    print("6) json.loads(arguments) :", args)

    # EXPECTED OUTPUT (representative)
    # 1) message.content      : None
    # 2) message.tool_calls    : <class 'list'> len 1
    # 3) tool_calls[0].id      : call_abc123
    # 4) .function.name        : answer_question
    # 5) .function.arguments   : '{"content":"A vector database stores embeddings...","confidence":0.9,"sources":[]}'
    # 6) json.loads(arguments) : {'content': 'A vector database stores embeddings...', 'confidence': 0.9, 'sources': []}
    #
    # THE KEY GOTCHAS
    # - content is None when the model makes a tool call — read tool_calls, not content.
    # - tool_calls is a LIST (a model can call several tools); we take [0].
    # - .function.arguments is a JSON *string*, NOT a dict — you must json.loads it.
    # - The easiest bug is forgetting the '.function.' step between tool_calls[0] and arguments.


# ===========================================================================
# STAGE 06 — Multi-shape: tool-calling makes MULTIPLE shapes trivial
# ---------------------------------------------------------------------------
# Give the model two tools and let it CHOOSE: answer, or ask for clarification.
# ===========================================================================
CLARIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_for_clarification",
        "description": "Use when the question is too vague to answer well.",
        "parameters": {
            "type": "object",
            "properties": {"question_back": {"type": "string"}},
            "required": ["question_back"],
        },
    },
}


def stage_06_multi_shape():
    for user_q in ["What is RAG?", "Can you help me with the thing?"]:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": user_q}],
            tools=[ANSWER_TOOL, CLARIFY_TOOL],
            tool_choice="auto",                # let the model pick which tool
        )
        call = resp.choices[0].message.tool_calls[0]
        print(f"Q: {user_q!r:40s} -> tool: {call.function.name}")
        print("   args:", call.function.arguments)

    # EXPECTED OUTPUT (representative)
    # Q: 'What is RAG?'                          -> tool: answer_question
    #    args: {"content":"RAG combines...","confidence":0.9,"sources":[]}
    # Q: 'Can you help me with the thing?'       -> tool: ask_for_clarification
    #    args: {"question_back":"Sure — which 'thing' do you mean?"}
    #
    # WHY THIS IS THE POINT
    # Multiple output shapes are just multiple tools. The model ROUTES to the right one,
    # and each shape is still provider-enforced. This is the seed of agent behaviour:
    # a model choosing among structured actions. Try adding a third tool yourself.


# ===========================================================================
# FIT IT INTO YOUR APP  (the handoff)
# ---------------------------------------------------------------------------
# You just proved tool-calling is the reliable way to get structure. Now wire it in:
#
# 1) Put ANSWER_TOOL (from Stage 4) at the top of src/pipeline/pipeline.py.
# 2) In ask_llm's REAL branch, call with tools=[ANSWER_TOOL] + tool_choice forcing it,
#    then parse EXACTLY as Stage 5:
#        args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
#        return Answer(content=args["content"], confidence=args["confidence"],
#                      sources=args.get("sources", []), cost_usd=0.0001, retries=attempt)
# 3) VERIFY it works:
#        # fake flow first (free):
#        USE_FAKE=1 uvicorn api.main:app --port 8000 &  ; sleep 3
#        curl -s -X POST localhost:8000/ask_batched -H 'Content-Type: application/json' \
#             -d '{"question":"What is RAG?"}'      # -> 6-field Answer, confidence=1.0 (default)
#        # then real quality:
#        kill %1 ; OPENAI_API_KEY=sk-... uvicorn api.main:app --port 8000 &  ; sleep 3
#        curl ... same ...                          # -> confidence is now the MODEL's number
#
# HOW IT HELPS OUR APP: this is what turns our answer from a blob of text into a
# verifiable object (content + confidence + sources) that everything downstream —
# storage, evaluation (W5), retrieval (W6), agents (Phase 5) — can build on.
# ===========================================================================


STAGES = {
    "1": stage_01_parse_happy, "2": stage_02_parse_breaks, "3": stage_03_json_mode,
    "4": stage_04_tool_calling, "5": stage_05_parse_dissected, "6": stage_06_multi_shape,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print("Run one stage at a time:\n")
        for k, fn in STAGES.items():
            print(f"  python {sys.argv[0]} {k}   # {fn.__name__}")
        print("\nStage 2 needs no key. Stages 1,3,4,5,6 make ONE real call each.")
        sys.exit(0)
    if sys.argv[1] != "2":                      # stage 2 is deterministic, no key needed
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: set OPENAI_API_KEY first  ->  export OPENAI_API_KEY=sk-...")
            sys.exit(1)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    STAGES[sys.argv[1]]()
