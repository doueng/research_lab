"""The dataroom: plain SQLite so you can poke at it with `sqlite3 data/<subject>.db`."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS leads (
  id INTEGER PRIMARY KEY,
  target TEXT NOT NULL UNIQUE,      -- a URL or a free-form question
  reason TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 50,
  status TEXT NOT NULL DEFAULT 'open',  -- open | done | dropped
  attempts INTEGER NOT NULL DEFAULT 0,
  from_tick INTEGER                 -- NULL means it came from the subject's seeds file
);
CREATE TABLE IF NOT EXISTS findings (
  id INTEGER PRIMARY KEY,
  url TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  why_valuable TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '',
  lead_id INTEGER REFERENCES leads(id),
  tick INTEGER NOT NULL,
  rating INTEGER                    -- your 1-5 rating, the ground truth for later evals
);
CREATE TABLE IF NOT EXISTS ticks (
  id INTEGER PRIMARY KEY,
  lead_id INTEGER,
  summary TEXT,
  finished INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def get_meta(conn, key: str) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else ""


def set_meta(conn, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
    conn.commit()


def add_lead(conn, target: str, reason: str, priority: int = 50, from_tick=None) -> bool:
    """Return False when the lead already exists (dedupe by target)."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO leads (target, reason, priority, from_tick) VALUES (?, ?, ?, ?)",
        (target.strip(), reason, max(0, min(100, priority)), from_tick),
    )
    conn.commit()
    return cur.rowcount == 1


def frontier(conn, limit: int = 15) -> list[dict]:
    rows = conn.execute(
        "SELECT id, target, reason, priority, attempts FROM leads WHERE status='open' "
        "ORDER BY priority DESC, id LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


def set_priority(conn, lead_id: int, priority: int) -> None:
    conn.execute("UPDATE leads SET priority=? WHERE id=?", (max(0, min(100, priority)), lead_id))
    conn.commit()


def finish_lead(conn, lead_id: int, status: str = "done") -> None:
    conn.execute("UPDATE leads SET status=?, attempts=attempts+1 WHERE id=?", (status, lead_id))
    conn.commit()


def add_finding(conn, f: dict, lead_id, tick: int) -> bool:
    """Idempotent on URL, so replaying a crashed tick cannot duplicate findings."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO findings (url, title, summary, why_valuable, tags, lead_id, tick) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (f["url"], f["title"], f["summary"], f["why_valuable"], ",".join(f.get("tags", [])), lead_id, tick),
    )
    conn.commit()
    return cur.rowcount == 1


def known_urls(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT url FROM findings")}


def start_tick(conn) -> int:
    """Reuse the last unfinished tick (crash recovery), else open a new one."""
    row = conn.execute("SELECT id FROM ticks WHERE finished=0 ORDER BY id DESC LIMIT 1").fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO ticks DEFAULT VALUES")
    conn.commit()
    return cur.lastrowid


def end_tick(conn, tick: int, lead_id, summary: str) -> None:
    conn.execute("UPDATE ticks SET finished=1, lead_id=?, summary=? WHERE id=?", (lead_id, summary, tick))
    conn.commit()
