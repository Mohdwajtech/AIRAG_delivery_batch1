import sqlite3
from pathlib import Path

DB_PATH = Path("data/capstone_side.db")

DOCS = [
    ("hr-001", "Leave & Time-Off Policy",      "policy",    "HR",      "v3", "2026-01-14", 42),
    ("hr-002", "Code of Conduct",               "policy",    "HR",      "v2", "2025-11-02", 55),
    ("hr-003", "Remote Work Guidelines",        "guideline", "HR",      "v4", "2026-02-20", 28),
    ("hr-004", "Onboarding Handbook",           "handbook",  "HR",      "v1", "2025-06-30", 90),
    ("it-001", "Acceptable Use Policy",         "policy",    "IT",      "v5", "2026-01-30", 37),
    ("it-002", "Security Incident Runbook",     "runbook",   "IT",      "v2", "2026-02-11", 33),
    ("it-003", "VPN & Access Setup",            "guideline", "IT",      "v3", "2025-12-05", 19),
    ("it-004", "Data Classification Standard",  "standard",  "IT",      "v2", "2026-01-08", 24),
    ("fin-001","Expense Reimbursement Policy",  "policy",    "Finance", "v3", "2026-01-22", 31),
    ("fin-002","Procurement Guidelines",        "guideline", "Finance", "v1", "2025-09-15", 40),
    ("fin-003","Travel Policy",                 "policy",    "Finance", "v2", "2025-10-19", 26),
    ("fin-004","Corporate Card Terms",          "terms",     "Finance", "v4", "2026-02-01", 18),
]

def seed(db_path=DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""CREATE TABLE document_metadata (
        id INTEGER PRIMARY KEY, source TEXT, title TEXT, doc_type TEXT,
        department TEXT, version TEXT, updated_at TEXT, chunk_count INTEGER)""")
    cur.executemany(
        "INSERT INTO document_metadata (source,title,doc_type,department,version,updated_at,chunk_count) VALUES (?,?,?,?,?,?,?)",
        DOCS)
    cur.execute("""CREATE TABLE audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT,
        request TEXT, sql TEXT, decision TEXT, row_count INTEGER, error TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS approval_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user_id TEXT,
        tool_name TEXT, args TEXT, decision TEXT)""")
    con.commit()
    con.close()
    return db_path

if __name__ == "__main__":
    seed()
    print(f"Database ready: {DB_PATH}")
