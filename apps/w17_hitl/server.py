import json, sqlite3, uuid, time
from datetime import datetime
from typing import TypedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from tools import (
    MockLLM, SearchWebTool, QuerySqlTool, CalculateTool, SendEmailTool
)

TOOLS = {t.name: t for t in [
    SearchWebTool(offline=True), QuerySqlTool(), CalculateTool(), SendEmailTool()
]}
llm = MockLLM()

DB_PATH = Path("data/capstone_side.db")

GATE_POLICY = {
    "search_web": "auto",
    "query_sql":  "auto",
    "calculate":  "auto",
    "send_email": "gate",
}

def needs_approval(tool_name: str) -> bool:
    return GATE_POLICY.get(tool_name, "gate") == "gate"


# --- HITL Graph --------------------------------------------------------------

class GateState(TypedDict):
    tool: str
    args: dict
    result: str
    decision: str

def gated_tool(s):
    decision = interrupt({
        "type": "approval_request",
        "tool": s["tool"],
        "args": s["args"],
    })
    if decision == "approve":
        return {"result": TOOLS[s["tool"]].call(s["args"]), "decision": "approved"}
    return {"result": "(rejected by user)", "decision": "rejected"}

g = StateGraph(GateState)
g.add_node("gated_tool", gated_tool)
g.add_edge(START, "gated_tool")
g.add_edge("gated_tool", END)
gate = g.compile(checkpointer=MemorySaver())


# --- Audit Trail -------------------------------------------------------------

def log_approval(tool, args, decision, user_id="anonymous"):
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS approval_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user_id TEXT,
        tool_name TEXT, args TEXT, decision TEXT)""")
    con.execute(
        "INSERT INTO approval_log(ts,user_id,tool_name,args,decision) VALUES(?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), user_id, tool,
         json.dumps(args), decision))
    con.commit()
    con.close()

def get_audit_rows(limit=50):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM approval_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# --- Streaming ---------------------------------------------------------------

def stream_agent_events(question: str):
    yield {"type": "planning", "data": f"Analyzing: {question}"}
    time.sleep(1.2)

    response = llm.complete(
        [{"role": "user", "content": question}],
        list(TOOLS.values())
    )

    if response.kind == "tool_use":
        tool_name = response.tool_name
        tool_args = response.tool_args or {}

        if needs_approval(tool_name):
            yield {"type": "approval_request", "data": {
                "tool": tool_name, "args": tool_args}}
            return

        yield {"type": "tool_call", "data": {"tool": tool_name, "args": tool_args}}
        time.sleep(1.5)

        obs = TOOLS[tool_name].call(tool_args)
        yield {"type": "tool_result", "data": str(obs)[:200]}
        time.sleep(1.2)

        yield {"type": "reflection", "data": "ok"}
        time.sleep(0.8)

        answer = f"Based on the tool result: {str(obs)[:120]}"
    else:
        answer = response.content or "I can answer that directly."

    for word in answer.split(" "):
        yield {"type": "token", "data": word + " "}
        time.sleep(0.08)

    yield {"type": "final", "data": "done"}


# --- FastAPI App -------------------------------------------------------------

app = FastAPI(title="W17 HITL Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_pending = {}


class AskRequest(BaseModel):
    question: str
    user: str = "anonymous"

class ActionRequest(BaseModel):
    tool: str
    args: dict
    user: str = "anonymous"

class DecisionRequest(BaseModel):
    thread_id: str
    decision: str
    user: str = "anonymous"


@app.get("/health")
def health():
    return {"status": "ok", "tools": list(TOOLS.keys()), "gate_policy": GATE_POLICY}


@app.post("/act")
def act(req: ActionRequest):
    if req.tool not in TOOLS:
        raise HTTPException(400, f"Unknown tool: {req.tool}")

    if not needs_approval(req.tool):
        result = TOOLS[req.tool].call(req.args)
        log_approval(req.tool, req.args, "auto-executed", req.user)
        return {"status": "completed", "result": result}

    thread_id = f"t-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": thread_id}}
    gate.invoke({"tool": req.tool, "args": req.args, "result": "", "decision": ""}, cfg)
    _pending[thread_id] = {"tool": req.tool, "args": req.args, "user": req.user}
    return {
        "status": "awaiting_approval",
        "thread_id": thread_id,
        "tool": req.tool,
        "args": req.args,
    }


@app.post("/decide")
def decide(req: DecisionRequest):
    if req.thread_id not in _pending:
        raise HTTPException(404, "Thread not found or already resolved.")
    info = _pending.pop(req.thread_id)
    cfg = {"configurable": {"thread_id": req.thread_id}}
    result = gate.invoke(Command(resume=req.decision), cfg)
    log_approval(info["tool"], info["args"], result["decision"], req.user)
    return {"status": "completed", "decision": result["decision"], "result": result["result"]}


@app.get("/stream")
async def stream(q: str = "how many documents?"):
    async def gen():
        for ev in stream_agent_events(q):
            yield json.dumps(ev) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/audit")
def audit(limit: int = 50):
    return get_audit_rows(limit)


@app.get("/pending")
def pending():
    return [
        {"thread_id": tid, "tool": info["tool"], "args": info["args"], "requested_by": info["user"]}
        for tid, info in _pending.items()
    ]
