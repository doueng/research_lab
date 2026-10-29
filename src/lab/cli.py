"""lab seed | status | tick | findings | rate ID N | lead TARGET REASON [PRIORITY]"""

import argparse
import os
import sqlite3
import tomllib

from dotenv import find_dotenv, load_dotenv

from lab import db

load_dotenv(find_dotenv(usecwd=True))  # .env from the working directory up; already-set env vars win
DB = os.environ.get("LAB_DB", "lab.db")
MODEL = os.environ.get("LAB_MODEL", "codex:gpt-6-sol")


def cmd_seed(conn, a):
    with open(a.file, "rb") as fh:
        seeds = tomllib.load(fh)
    db.set_meta(conn, "objective", seeds["lab"]["objective"].strip())
    added = sum(db.add_lead(conn, l["target"], l["reason"], l.get("priority", 50)) for l in seeds["lead"])
    print(f"objective set, {added} new leads ({len(seeds['lead']) - added} already present)")


def cmd_status(conn, a):
    n = lambda q: conn.execute(q).fetchone()[0]
    open_q = "SELECT count(*) FROM leads WHERE status='open'"
    print(f"ticks {n('SELECT count(*) FROM ticks WHERE finished=1')}  "
          f"findings {n('SELECT count(*) FROM findings')}  "
          f"rated {n('SELECT count(*) FROM findings WHERE rating IS NOT NULL')}  "
          f"open leads {n(open_q)}")
    print("\n## scratchpad\n" + (db.get_meta(conn, "scratchpad") or "(empty)"))
    print("\n## top of frontier")
    for l in db.frontier(conn, 10):
        print(f"  #{l['id']} p{l['priority']}  {l['target']}")


def cmd_tick(conn, a):
    from langgraph.checkpoint.sqlite import SqliteSaver

    from lab.graph import build_graph, run_tick
    from lab.models import chat_model
    from lab.tools import default_tools

    ckpt = SqliteSaver(sqlite3.connect(DB.removesuffix(".db") + ".ckpt.db", check_same_thread=False))
    out = run_tick(conn, build_graph(conn, chat_model(MODEL), default_tools(), checkpointer=ckpt))
    inv = out["investigation"]
    print(f"tick {out['tick']}: {out['lead']['target'] if out['lead'] else 'explore: ' + out['plan']['explore_query']}")
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


def main():
    ap = argparse.ArgumentParser(prog="lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed").add_argument("file", nargs="?", default="seeds.toml")
    sub.add_parser("status")
    sub.add_parser("tick")
    sub.add_parser("findings").add_argument("-n", type=int, default=20)
    r = sub.add_parser("rate")
    r.add_argument("id", type=int)
    r.add_argument("score", type=int, choices=range(1, 6))
    l = sub.add_parser("lead")
    l.add_argument("target")
    l.add_argument("reason")
    l.add_argument("priority", type=int, nargs="?", default=70)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](db.connect(DB), a)
