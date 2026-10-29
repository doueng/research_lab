"""Command line: lab [-s SUBJECT] seed | status | tick | findings | rate ID N | lead TARGET REASON [PRI] | subjects."""

import argparse
import os
import sqlite3
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from lab import db

load_dotenv(find_dotenv(usecwd=True))  # .env from the working directory up; already-set env vars win
SEEDS_DIR, DATA_DIR = Path("seeds"), Path("data")
MODEL = os.environ.get("LAB_MODEL", "codex:gpt-6-sol")


@dataclass(frozen=True)
class Subject:
    """A research subject exists iff seeds/<name>.toml does. Each one gets its own dataroom."""

    name: str

    @property
    def seeds(self) -> Path:
        return SEEDS_DIR / f"{self.name}.toml"

    @property
    def db(self) -> Path:
        return DATA_DIR / f"{self.name}.db"

    @property
    def ckpt(self) -> Path:
        return DATA_DIR / f"{self.name}.ckpt.db"


def subjects() -> list[Subject]:
    return [Subject(p.stem) for p in sorted(SEEDS_DIR.glob("*.toml"))]


def cmd_seed(conn, a):
    with open(a.subject.seeds, "rb") as fh:
        seeds = tomllib.load(fh)
    db.set_meta(conn, "objective", seeds["lab"]["objective"].strip())
    added = sum(db.add_lead(conn, s["target"], s["reason"], s.get("priority", 50)) for s in seeds["lead"])
    print(f"objective set, {added} new leads ({len(seeds['lead']) - added} already present)")


def cmd_status(conn, a):
    def count(q):
        return conn.execute(f"SELECT count(*) FROM {q}").fetchone()[0]

    open_leads = count("leads WHERE status='open'")
    print(f"ticks {count('ticks WHERE finished=1')}  findings {count('findings')}  "
          f"rated {count('findings WHERE rating IS NOT NULL')}  open leads {open_leads}")
    print("\n## scratchpad\n" + (db.get_meta(conn, "scratchpad") or "(empty)"))
    print("\n## top of frontier")
    for lead in db.frontier(conn, 10):
        print(f"  #{lead['id']} p{lead['priority']}  {lead['target']}")


def cmd_tick(conn, a):
    from langgraph.checkpoint.sqlite import SqliteSaver

    from lab.graph import build_graph, run_tick
    from lab.models import chat_model
    from lab.tools import default_tools

    ckpt = SqliteSaver(sqlite3.connect(a.subject.ckpt, check_same_thread=False))
    out = run_tick(conn, build_graph(conn, chat_model(MODEL), default_tools(), checkpointer=ckpt))
    inv = out["investigation"]
    where = out["lead"]["target"] if out["lead"] else "explore: " + out["plan"]["explore_query"]
    print(f"[{a.subject.name}] tick {out['tick']}: {where}")
    print(f"  {inv['summary']}")
    for f in inv["findings"]:
        print(f"  + {f['title']}  {f['url']}")
    print(f"  {len(inv['new_leads'])} new leads proposed")


def cmd_findings(conn, a):
    for r in conn.execute("SELECT * FROM findings ORDER BY id DESC LIMIT ?", (a.n,)):
        print(f"#{r['id']} [{r['rating'] or '-'}] {r['title']}\n   {r['url']}\n   {r['why_valuable']}\n")


def cmd_rate(conn, a):
    conn.execute("UPDATE findings SET rating=? WHERE id=?", (a.score, a.id))
    conn.commit()


def cmd_lead(conn, a):
    print("added" if db.add_lead(conn, a.target, a.reason, a.priority) else "already present")


def list_subjects():
    for s in subjects():
        print(f"  {s.name:<14} {'seeded' if s.db.exists() else 'not seeded (run: lab -s ' + s.name + ' seed)'}")


def main():
    ap = argparse.ArgumentParser(prog="lab")
    ap.add_argument("-s", "--subject", default=os.environ.get("LAB_SUBJECT", "guix"),
                    help="research subject, one of seeds/*.toml (default: $LAB_SUBJECT or guix)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("subjects")
    sub.add_parser("seed")
    sub.add_parser("status")
    sub.add_parser("tick")
    sub.add_parser("findings").add_argument("-n", type=int, default=20)
    r = sub.add_parser("rate")
    r.add_argument("id", type=int)
    r.add_argument("score", type=int, choices=range(1, 6))
    lp = sub.add_parser("lead")
    lp.add_argument("target")
    lp.add_argument("reason")
    lp.add_argument("priority", type=int, nargs="?", default=70)
    a = ap.parse_args()
    if a.cmd == "subjects":
        return list_subjects()
    a.subject = Subject(a.subject)
    if not a.subject.seeds.exists():
        sys.exit(f"unknown subject {a.subject.name!r}: no {a.subject.seeds}. Known: "
                 + (", ".join(s.name for s in subjects()) or "(none)"))
    DATA_DIR.mkdir(exist_ok=True)
    globals()[f"cmd_{a.cmd}"](db.connect(a.subject.db), a)
