from __future__ import annotations
import ast, json, os, re, operator, sqlite3
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field


class Tool:
    name: str = ""
    description: str = ""
    Args: type[BaseModel] = BaseModel
    idempotent: bool = True
    has_side_effects: bool = False

    def run(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def call(self, arguments: dict[str, Any]) -> str:
        args = self.Args(**(arguments or {}))
        return self.run(**args.model_dump())

    def parameters_schema(self) -> dict:
        return self.Args.model_json_schema()

    def openai_schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters_schema()}}


# --- Calculate ---------------------------------------------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
}

def _eval(node):
    if isinstance(node, ast.Expression): return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


class CalculateArgs(BaseModel):
    expression: str = Field(..., description="Arithmetic expression, e.g. '1350 * 0.18'")

class CalculateTool(Tool):
    name = "calculate"
    description = "Evaluate an arithmetic expression and return the result."
    Args = CalculateArgs
    idempotent = True
    has_side_effects = False

    def run(self, expression: str) -> str:
        try:
            result = _eval(ast.parse(expression, mode="eval"))
        except Exception as e:
            return f"calculate error: {e}"
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{expression} = {result}"


# --- Query SQL ---------------------------------------------------------------

DB_PATH = Path("data/capstone_side.db")
_FORBIDDEN = {"insert","update","delete","drop","alter","create","replace",
              "truncate","attach","detach","pragma","vacuum","reindex"}
_MAX_ROWS = 50

def _strip_comments(sql):
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return sql.strip()

def is_read_only(sql):
    clean = _strip_comments(sql)
    if not clean: return False, "empty query"
    statements = [s for s in clean.split(";") if s.strip()]
    if len(statements) > 1: return False, "multiple statements not allowed"
    first = statements[0].lstrip().split(None, 1)[0].lower()
    if first not in ("select", "with"): return False, f"only SELECT allowed (got '{first.upper()}')"
    words = set(re.findall(r"[a-z_]+", clean.lower()))
    hit = words & _FORBIDDEN
    if hit: return False, f"forbidden keyword: {', '.join(sorted(hit))}"
    return True, "ok"


class QuerySqlArgs(BaseModel):
    sql: str = Field(..., description="A read-only SQL SELECT query.")
    purpose: str = Field("", description="What the query is for (logged).")

class QuerySqlTool(Tool):
    name = "query_sql"
    description = "Run a read-only SQL SELECT against document_metadata."
    Args = QuerySqlArgs
    idempotent = True
    has_side_effects = False

    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)

    def run(self, sql: str, purpose: str = "") -> str:
        ok, reason = is_read_only(sql)
        if not ok:
            return f"query_sql DENIED: {reason}"
        try:
            con = sqlite3.connect(self.db_path)
            cur = con.execute(sql)
            rows = cur.fetchmany(_MAX_ROWS)
            cols = [d[0] for d in cur.description] if cur.description else []
            con.close()
        except Exception as e:
            return f"query_sql error: {e}"
        if not rows:
            return "query_sql: (0 rows)"
        header = " | ".join(cols)
        body = "\n".join(" | ".join(str(v) for v in r) for r in rows)
        return f"{header}\n{'-'*len(header)}\n{body}"


# --- Search Web --------------------------------------------------------------

FIXTURES_PATH = Path("data/web_fixtures.json")

class SearchWebArgs(BaseModel):
    query: str = Field(..., description="A web-search query.")
    max_results: int = Field(3, ge=1, le=5)

class SearchWebTool(Tool):
    name = "search_web"
    description = "Search the public web and return snippets."
    Args = SearchWebArgs
    idempotent = True
    has_side_effects = False

    def __init__(self, offline=None):
        self.offline = (not os.environ.get("TAVILY_API_KEY")) if offline is None else offline

    def run(self, query: str, max_results: int = 3) -> str:
        try:
            data = json.loads(FIXTURES_PATH.read_text())
        except Exception:
            data = {"_default": [{"title": "No fixture", "snippet": "offline stub."}]}
        results = None
        for key, res in data.items():
            if key != "_default" and key.lower() in query.lower():
                results = res
                break
        results = (results or data.get("_default", []))[:max_results]
        lines = [f"[web:{query}]"]
        for r in results:
            lines.append(f"- {r['title']}: {r['snippet']}")
        return "\n".join(lines)


# --- Send Email (irreversible) -----------------------------------------------

class SendEmailArgs(BaseModel):
    to: str = Field(..., description="Recipient email address.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field("", description="Email body.")

class SendEmailTool(Tool):
    name = "send_email"
    description = "Send an email. Irreversible — gated behind human approval."
    Args = SendEmailArgs
    idempotent = False
    has_side_effects = True

    def run(self, to: str, subject: str, body: str = "") -> str:
        return f"Email sent to {to}: '{subject}'"


# --- Mock LLM ----------------------------------------------------------------

from dataclasses import dataclass
from typing import Callable

@dataclass
class LLMResponse:
    kind: str
    tool_name: str | None = None
    tool_args: dict | None = None
    content: str | None = None

    @staticmethod
    def tool(name, **args): return LLMResponse("tool_use", tool_name=name, tool_args=args)
    @staticmethod
    def final(content): return LLMResponse("final", content=content)


class MockLLM:
    def __init__(self, script=None, router=None):
        self.script = list(script) if script else None
        self.router = router
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.script:
            return self.script.pop(0) if self.script else LLMResponse.final("(no more)")
        if self.router:
            return self.router(messages, tools)
        return self._keyword_route(messages, tools)

    def _keyword_route(self, messages, tools):
        obs = _last_observation(messages)
        if obs is not None:
            return LLMResponse.final(f"Based on the tool result: {obs}")
        q = _last_user(messages).lower()
        names = {t.name for t in tools}
        if "calculate" in names and re.search(r"\d\s*[-+*/%]\s*\d|percent|average|sum|multiply", q):
            return LLMResponse.tool("calculate", expression=_extract_expr(q))
        sql_kw = ("how many doc","which department","what version","list the",
                  "count","how many hr","how many it","how many finance","updated","chunk")
        if "query_sql" in names and any(k in q for k in sql_kw):
            return LLMResponse.tool("query_sql",
                sql="SELECT COUNT(*) AS total_documents FROM document_metadata", purpose="count")
        web_kw = ("latest","current","today","news","price","recent","search","trend","who won")
        if "search_web" in names and any(k in q for k in web_kw):
            return LLMResponse.tool("search_web", query=_last_user(messages), max_results=3)
        email_kw = ("send","email","mail","notify","message to")
        if "send_email" in names and any(k in q for k in email_kw):
            return LLMResponse.tool("send_email", to="user@example.com",
                subject="Agent-generated message", body=_last_user(messages))
        return LLMResponse.final("I can answer that directly: no tool needed.")


def _last_user(messages):
    for m in reversed(messages):
        if m.get("role") == "user": return str(m.get("content", ""))
    return ""

def _last_observation(messages):
    for m in reversed(messages):
        if m.get("role") == "tool": return str(m.get("content", ""))
    return None

def _extract_expr(q):
    m = re.search(r"[0-9][0-9\.\s\+\-\*/%\(\)]*[0-9\)]", q)
    return m.group(0).strip() if m else "0"
