import json
import requests
import streamlit as st
import pandas as pd

API = "http://localhost:8000"

st.set_page_config(page_title="W17 HITL Agent", layout="wide", page_icon="🛡️")


# --- Session State Init ------------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.messages = []
    st.session_state.pending_actions = []


# --- User Login --------------------------------------------------------------

USERS = {
    "alice":   {"password": "alice123",   "role": "Analyst"},
    "bob":     {"password": "bob123",     "role": "Manager"},
    "charlie": {"password": "charlie123", "role": "Admin"},
}

def login_page():
    st.markdown("## 🛡️ W17 HITL Agent — Login")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("#### Sign in to continue")
        username = st.text_input("Username", placeholder="alice / bob / charlie")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("Sign In", use_container_width=True, type="primary"):
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = USERS[username]["role"]
                st.rerun()
            else:
                st.error("Invalid credentials.")

        st.markdown("---")
        st.markdown(
            "**Demo accounts:**  \n"
            "`alice` / `alice123` (Analyst)  \n"
            "`bob` / `bob123` (Manager)  \n"
            "`charlie` / `charlie123` (Admin)"
        )


# --- Sidebar ------------------------------------------------------------------

def sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.caption(f"Role: {st.session_state.role}")
        st.markdown("---")

        page = st.radio("Navigation", [
            "💬 Agent Chat",
            "⚡ Direct Tool Execution",
            "📋 Pending Approvals",
            "📊 Audit Trail",
        ])

        st.markdown("---")
        if st.button("Sign Out"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.caption("Gate Policy")
        try:
            r = requests.get(f"{API}/health", timeout=3)
            policy = r.json().get("gate_policy", {})
            for tool, action in policy.items():
                icon = "🟢" if action == "auto" else "🔴"
                st.caption(f"{icon} {tool}: {action}")
        except Exception:
            st.caption("⚠️ API not reachable")

    return page


# --- Agent Chat (Streaming) ---------------------------------------------------

def chat_page():
    st.markdown("## 💬 Agent Chat")
    st.caption("Ask a question — watch the agent plan, call tools, and respond in real time.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("events"):
                render_event_log(msg["events"])
            st.markdown(msg["content"])

    question = st.chat_input("Ask the agent something...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        event_log_area = st.container()
        token_area = st.empty()
        tokens = []
        final_content = ""
        approval_needed = None
        event_trail = []

        try:
            with requests.get(f"{API}/stream", params={"q": question}, stream=True, timeout=60) as resp:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    ev = json.loads(line)
                    etype = ev["type"]
                    edata = ev["data"]

                    if etype == "planning":
                        event_trail.append(("🧠", "Planning", str(edata)))
                        with event_log_area:
                            render_event_log(event_trail)

                    elif etype == "tool_call":
                        tool_info = edata if isinstance(edata, dict) else {}
                        tname = tool_info.get("tool", str(edata))
                        event_trail.append(("🔧", "Tool Call", f"**{tname}** → `{json.dumps(tool_info.get('args', {}))[:80]}`"))
                        with event_log_area:
                            render_event_log(event_trail)

                    elif etype == "tool_result":
                        event_trail.append(("📄", "Result", f"`{str(edata)[:120]}`"))
                        with event_log_area:
                            render_event_log(event_trail)

                    elif etype == "reflection":
                        icon = "✅" if edata == "ok" else "🔄"
                        label = "Reflection: OK" if edata == "ok" else "Reflection: REDO"
                        event_trail.append((icon, label, ""))
                        with event_log_area:
                            render_event_log(event_trail)

                    elif etype == "token":
                        tokens.append(str(edata))
                        token_area.markdown("".join(tokens) + "▌")

                    elif etype == "approval_request":
                        approval_needed = edata
                        event_trail.append(("🛑", "Approval Required",
                            f"**{edata['tool']}** with `{json.dumps(edata['args'])[:80]}`"))
                        with event_log_area:
                            render_event_log(event_trail)

                    elif etype == "final":
                        if tokens:
                            final_content = "".join(tokens)
                            token_area.markdown(final_content)

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Is the server running?")
            return

        if approval_needed:
            final_content = (
                f"⚠️ The agent wants to execute `{approval_needed['tool']}` "
                f"with args `{approval_needed['args']}`. "
                f"Go to **Pending Approvals** to approve or reject."
            )

            tool = approval_needed["tool"]
            args = approval_needed["args"]
            r = requests.post(f"{API}/act", json={
                "tool": tool, "args": args,
                "user": st.session_state.username
            })
            if r.ok and r.json().get("status") == "awaiting_approval":
                thread_id = r.json()["thread_id"]
                st.session_state.pending_actions.append({
                    "thread_id": thread_id,
                    "tool": tool,
                    "args": args,
                })

            token_area.markdown(final_content)

        if not final_content and not approval_needed:
            final_content = "(no response)"
            token_area.markdown(final_content)

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_content,
            "events": event_trail,
        })


def render_event_log(events):
    if not events:
        return
    lines = []
    for icon, label, detail in events:
        if detail:
            lines.append(f"{icon} **{label}** — {detail}")
        else:
            lines.append(f"{icon} **{label}**")
    st.markdown("\n\n".join(lines))


# --- Direct Tool Execution ----------------------------------------------------

def tool_page():
    st.markdown("## ⚡ Direct Tool Execution")
    st.caption("Execute a tool directly and see the gate in action.")

    col1, col2 = st.columns(2)

    with col1:
        tool_name = st.selectbox("Tool", ["query_sql", "calculate", "search_web", "send_email"])

    with col2:
        gate_status = "🔴 GATED" if tool_name == "send_email" else "🟢 AUTO"
        st.markdown(f"**Gate status:** {gate_status}")

    st.markdown("**Arguments:**")
    if tool_name == "query_sql":
        sql = st.text_area("SQL", "SELECT department, COUNT(*) as doc_count FROM document_metadata GROUP BY department")
        args = {"sql": sql, "purpose": "direct query"}
    elif tool_name == "calculate":
        expr = st.text_input("Expression", "1350 * 0.18")
        args = {"expression": expr}
    elif tool_name == "search_web":
        query = st.text_input("Search query", "latest AI agent trends")
        args = {"query": query, "max_results": 3}
    elif tool_name == "send_email":
        to = st.text_input("To", "alice@acme.com")
        subject = st.text_input("Subject", "Q4 Report")
        body = st.text_area("Body", "Please find attached the Q4 report.")
        args = {"to": to, "subject": subject, "body": body}

    if st.button("Execute", type="primary", use_container_width=True):
        with st.spinner("Executing..."):
            r = requests.post(f"{API}/act", json={
                "tool": tool_name, "args": args,
                "user": st.session_state.username
            })

        if r.ok:
            data = r.json()
            if data["status"] == "completed":
                st.success(f"**Result:** {data['result']}")
            elif data["status"] == "awaiting_approval":
                st.warning(
                    f"**Awaiting approval.** Thread: `{data['thread_id']}`  \n"
                    f"Go to **Pending Approvals** to approve or reject."
                )
                st.session_state.pending_actions.append({
                    "thread_id": data["thread_id"],
                    "tool": tool_name,
                    "args": args,
                })
        else:
            st.error(f"Error: {r.text}")


# --- Pending Approvals --------------------------------------------------------

def approvals_page():
    st.markdown("## 📋 Pending Approvals")
    st.caption(f"Logged in as **{st.session_state.username}** ({st.session_state.role})")

    try:
        r = requests.get(f"{API}/pending", timeout=3)
        server_pending = r.json() if r.ok else []
    except Exception:
        server_pending = []

    local_threads = {p["thread_id"] for p in st.session_state.pending_actions}
    for sp in server_pending:
        if sp["thread_id"] not in local_threads:
            st.session_state.pending_actions.append({
                "thread_id": sp["thread_id"],
                "tool": sp["tool"],
                "args": sp["args"],
            })

    if not st.session_state.pending_actions:
        st.info("No pending approvals. Try sending an email from Agent Chat or Direct Tool Execution.")
        return

    for i, action in enumerate(list(st.session_state.pending_actions)):
        with st.container(border=True):
            st.markdown(f"**Tool:** `{action['tool']}`")
            st.json(action["args"])
            st.caption(f"Thread: `{action['thread_id']}`")

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("✅ Approve", key=f"approve-{action['thread_id']}", type="primary"):
                    r = requests.post(f"{API}/decide", json={
                        "thread_id": action["thread_id"],
                        "decision": "approve",
                        "user": st.session_state.username,
                    })
                    if r.ok:
                        st.success(f"Approved! Result: {r.json()['result']}")
                        st.session_state.pending_actions.remove(action)
                        st.rerun()
                    else:
                        st.error(f"Error: {r.text}")

            with col2:
                if st.button("❌ Reject", key=f"reject-{action['thread_id']}"):
                    r = requests.post(f"{API}/decide", json={
                        "thread_id": action["thread_id"],
                        "decision": "reject",
                        "user": st.session_state.username,
                    })
                    if r.ok:
                        st.warning(f"Rejected. {r.json()['result']}")
                        st.session_state.pending_actions.remove(action)
                        st.rerun()
                    else:
                        st.error(f"Error: {r.text}")


# --- Audit Trail --------------------------------------------------------------

def audit_page():
    st.markdown("## 📊 Audit Trail")
    st.caption("Every approval decision is logged here — who, when, what, decision.")

    col1, col2 = st.columns([3, 1])
    with col2:
        limit = st.number_input("Rows to show", min_value=5, max_value=200, value=25)
        if st.button("🔄 Refresh"):
            st.rerun()

    try:
        r = requests.get(f"{API}/audit", params={"limit": limit}, timeout=3)
        rows = r.json() if r.ok else []
    except Exception:
        st.error("Cannot reach API.")
        return

    if not rows:
        st.info("No audit entries yet. Execute some tools to generate entries.")
        return

    df = pd.DataFrame(rows)
    display_cols = ["id", "ts", "user_id", "tool_name", "decision", "args"]
    available = [c for c in display_cols if c in df.columns]
    df = df[available]

    col_config = {
        "decision": st.column_config.TextColumn("Decision", width="small"),
        "user_id": st.column_config.TextColumn("User", width="small"),
        "tool_name": st.column_config.TextColumn("Tool", width="small"),
    }

    st.dataframe(df, use_container_width=True, hide_index=True, column_config=col_config)

    st.markdown("---")
    st.markdown("### Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Decisions", len(df))
    with col2:
        approved = len(df[df["decision"] == "approved"]) if "decision" in df.columns else 0
        st.metric("Approved", approved)
    with col3:
        rejected = len(df[df["decision"] == "rejected"]) if "decision" in df.columns else 0
        st.metric("Rejected", rejected)
    with col4:
        auto = len(df[df["decision"] == "auto-executed"]) if "decision" in df.columns else 0
        st.metric("Auto-executed", auto)

    if "user_id" in df.columns:
        st.markdown("### Decisions by User")
        user_counts = df.groupby(["user_id", "decision"]).size().reset_index(name="count")
        st.dataframe(user_counts, use_container_width=True, hide_index=True)

    if "tool_name" in df.columns:
        st.markdown("### Decisions by Tool")
        tool_counts = df.groupby(["tool_name", "decision"]).size().reset_index(name="count")
        st.dataframe(tool_counts, use_container_width=True, hide_index=True)


# --- Main Router --------------------------------------------------------------

def main():
    if not st.session_state.logged_in:
        login_page()
        return

    page = sidebar()

    if "Agent Chat" in page:
        chat_page()
    elif "Direct Tool" in page:
        tool_page()
    elif "Pending" in page:
        approvals_page()
    elif "Audit" in page:
        audit_page()


if __name__ == "__main__":
    main()
