from __future__ import annotations
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field
from .base import Tool

DB_PATH = Path("data/capstone_side.db")
_FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "truncate", "attach", "detach", "pragma", "vacuum", "reindex",
}
_MAX_ROWS = 50


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return sql.strip()


def is_read_only(sql: str) -> tuple[bool, str]:
    clean = _strip_comments(sql)
    if not clean:
        return False, "empty query"
    statements = [s for s in clean.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "multiple statements not allowed"
    first = statements[0].lstrip().split(None, 1)[0].lower()
    if first not in ("select", "with"):
        return False, f"only SELECT allowed (got '{first.upper()}')"
    words = set(re.findall(r"[a-z_]+", clean.lower()))
    hit = words & _FORBIDDEN
    if hit:
        return False, f"forbidden keyword: {', '.join(sorted(hit))}"
    return True, "ok"


class QuerySqlArgs(BaseModel):
    sql: str = Field(..., description="A single read-only SQL SELECT query.")
    purpose: str = Field("", description="What the query is for (logged).")


class QuerySqlTool(Tool):
    name = "query_sql"
    description = (
        "Run a read-only SQL SELECT against the knowledge-base metadata database. "
        "Table: document_metadata (id, source, title, doc_type, department, version, "
        "updated_at, chunk_count).")
    Args = QuerySqlArgs
    idempotent = True
    has_side_effects = False

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)

    def _audit(self, request, sql, decision, row_count, error):
        con = sqlite3.connect(self.db_path)
        con.execute(
            "INSERT INTO audit_log (ts,actor,request,sql,decision,row_count,error) "
            "VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "agent", request, sql, decision, row_count, error))
        con.commit()
        con.close()

    def run(self, sql: str, purpose: str = "") -> str:
        ok, reason = is_read_only(sql)
        if not ok:
            self._audit(purpose, sql, "denied", None, reason)
            return f"query_sql DENIED: {reason}"
        try:
            con = sqlite3.connect(self.db_path)
            cur = con.execute(sql)
            rows = cur.fetchmany(_MAX_ROWS)
            cols = [d[0] for d in cur.description] if cur.description else []
            con.close()
        except Exception as e:
            self._audit(purpose, sql, "error", None, str(e))
            return f"query_sql error: {e}"
        self._audit(purpose, sql, "allowed", len(rows), None)
        if not rows:
            return "query_sql: (0 rows)"
        header = " | ".join(cols)
        body = "\n".join(" | ".join(str(v) for v in r) for r in rows)
        more = "" if len(rows) < _MAX_ROWS else f"\n... (capped at {_MAX_ROWS} rows)"
        return f"{header}\n{'-' * len(header)}\n{body}{more}"
