# W17 HITL Agent — Streaming + Approval + Audit Trail

A FastAPI + Streamlit application demonstrating Human-in-the-Loop agent patterns:
streaming UX, approval gates on irreversible tools, and a compliance-grade audit trail.

## Quick Start

```bash
pip install -r requirements.txt
python seed_db.py
```

Open two terminals:

```bash
# Terminal 1 — API server
uvicorn server:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run ui.py --server.port 8501
```

Open http://localhost:8501 in your browser.

## Demo Accounts

| Username  | Password     | Role    |
|-----------|-------------|---------|
| alice     | alice123    | Analyst |
| bob       | bob123      | Manager |
| charlie   | charlie123  | Admin   |

## Gate Policy

| Tool        | Policy | Why                           |
|-------------|--------|-------------------------------|
| search_web  | AUTO   | Read-only, no side effects    |
| query_sql   | AUTO   | Read-only SELECT              |
| calculate   | AUTO   | Pure computation              |
| send_email  | GATE   | Irreversible — human approval |

## Demo Scenarios

### Scenario 1: Auto-execute (no gate)

1. Login as `alice`
2. Agent Chat → "How many documents are in the knowledge base?"
3. Watch events stream: planning → tool_call (query_sql) → tool_result → tokens → final
4. Check Audit Trail → entry shows `auto-executed` by `alice`

### Scenario 2: HITL approval gate

1. Agent Chat → "Send an email to bob@acme.com about the leave policy"
2. Watch: planning → approval_request (stream stops)
3. Go to Pending Approvals → see the email action with full args
4. Click Approve
5. Check Audit Trail → entry shows `approved` by `alice`

### Scenario 3: HITL rejection

1. Agent Chat → "Email charlie@acme.com the financial data"
2. Go to Pending Approvals → click Reject
3. Check Audit Trail → entry shows `rejected` by `alice`

### Scenario 4: Multi-user audit

1. Login as `alice` → approve one email
2. Sign out → login as `bob` → reject another email
3. Check Audit Trail → see both decisions with different user_ids
4. "Decisions by User" table shows who approved/rejected what

### Scenario 5: Direct tool execution

1. Direct Tool Execution → select `query_sql`
2. Enter: `SELECT department, COUNT(*) FROM document_metadata GROUP BY department`
3. Click Execute → instant result (auto, no gate)
4. Switch to `send_email` → fill in fields → Execute → redirected to Pending Approvals

### Scenario 6: Streaming with different queries

Try these in Agent Chat to see different tool routing:
- "How many IT documents are there?" → query_sql
- "What is 1350 * 0.18?" → calculate
- "What are the latest AI agent trends?" → search_web
- "Send alice an email about the meeting" → send_email (GATED)

## Architecture

```
┌──────────────────┐     HTTP      ┌──────────────────┐
│  Streamlit UI    │ ◄──────────►  │  FastAPI Server   │
│  (ui.py)         │               │  (server.py)      │
│                  │               │                   │
│  - Login         │   /stream     │  - MockLLM        │
│  - Chat stream   │   /act        │  - TOOLS dict     │
│  - Approve/Reject│   /decide     │  - LangGraph gate │
│  - Audit table   │   /audit      │  - Audit log      │
│                  │   /pending    │  - MemorySaver    │
└──────────────────┘               └──────────┬────────┘
                                              │
                                   ┌──────────▼────────┐
                                   │  SQLite            │
                                   │  capstone_side.db  │
                                   │  - document_metadata│
                                   │  - approval_log    │
                                   └───────────────────┘
```

## Files

| File             | Purpose                                    |
|------------------|--------------------------------------------|
| server.py        | FastAPI backend — streaming, HITL, audit   |
| ui.py            | Streamlit frontend — chat, approvals, logs |
| tools.py         | Tool classes + MockLLM (from W13-W14)      |
| seed_db.py       | Create and populate the SQLite database    |
| requirements.txt | Python dependencies                        |
| data/            | SQLite database + web search fixtures      |
