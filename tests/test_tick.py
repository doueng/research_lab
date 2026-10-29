"""Deterministic tests: a scripted fake chat model plus stub tools, no network or API key."""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from lab import db
from lab.graph import build_graph, run_tick


class ScriptedModel(FakeMessagesListChatModel):
    """Returns scripted messages in order; ignores bound tools (the script already names them)."""

    def bind_tools(self, tools, **kw):
        return self


def call(name, args, i):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


@tool
def web_search(query: str) -> str:
    """stub"""
    return "Shepherd timers thread\nhttps://lists.gnu.org/x/timers.html\nargument about timers"


@tool
def fetch_url(url: str) -> str:
    """stub"""
    return f"<untrusted_page url='{url}'>page text</untrusted_page>"


def tick_script(lead_id, url, new_lead):
    finding = {"url": url, "title": "T " + url, "summary": "s", "why_valuable": "w", "tags": ["shepherd"]}
    ghost = dict(finding, url="https://never-fetched.example")
    return [
        call("Plan", {"lead_id": lead_id, "rationale": "top lead"}, 1),
        call("fetch_url", {"url": url}, 2),
        call("Investigation", {"findings": [finding, ghost], "summary": f"read {url}",
                                "new_leads": [{"target": new_lead, "reason": "r", "priority": 90}]}, 3),
        call("Reflection", {"scratchpad": f"after {url}", "rescore": []}, 4),
    ]


def seeded(tmp_path):
    conn = db.connect(tmp_path / "lab.db")
    db.set_meta(conn, "objective", "learn guix")
    db.add_lead(conn, "https://seed-a", "seed", 80)
    db.add_lead(conn, "https://seed-b", "seed", 40)
    return conn


def test_two_ticks_continue_across_fresh_graphs(tmp_path):
    conn = seeded(tmp_path)
    tools = [web_search, fetch_url]

    m1 = ScriptedModel(responses=tick_script(1, "https://seed-a", "https://lead-from-tick-1"))
    out1 = run_tick(conn, build_graph(conn, m1, tools, InMemorySaver()))
    assert out1["lead"]["id"] == 1

    # Fresh connection and graph, as if `lab tick` ran again in a new process.
    conn = db.connect(tmp_path / "lab.db")
    assert db.get_meta(conn, "scratchpad") == "after https://seed-a"
    front = [l["target"] for l in db.frontier(conn)]
    assert front[0] == "https://lead-from-tick-1" and "https://seed-a" not in front

    # Tick 2 reports the same URL again: the dataroom must not duplicate it.
    m2 = ScriptedModel(responses=tick_script(3, "https://seed-a", "https://seed-b"))
    out2 = run_tick(conn, build_graph(conn, m2, tools, InMemorySaver()))
    assert out2["tick"] == 2
    urls = [r[0] for r in conn.execute("SELECT url FROM findings")]
    assert urls == ["https://seed-a"]  # deduped, and the never-fetched URL was filtered out
    assert conn.execute("SELECT count(*) FROM leads").fetchone()[0] == 3  # seed-b was not re-added
    assert conn.execute("SELECT count(*) FROM ticks WHERE finished=1").fetchone()[0] == 2


def test_crashed_tick_resumes_from_checkpoint(tmp_path):
    conn = seeded(tmp_path)
    saver = InMemorySaver()
    script = tick_script(1, "https://seed-a", "https://n")

    # The reflect node's model call is missing, so the tick crashes after record.
    crashing = ScriptedModel(responses=script[:3] + [AIMessage("no tool call")])
    try:
        run_tick(conn, build_graph(conn, crashing, [web_search, fetch_url], saver))
    except Exception:
        pass
    assert conn.execute("SELECT finished FROM ticks").fetchone()[0] == 0

    # Resume: only reflect should run, so only one scripted response is needed.
    resumed = ScriptedModel(responses=[script[3]])
    out = run_tick(conn, build_graph(conn, resumed, [web_search, fetch_url], saver))
    assert out["tick"] == 1
    assert conn.execute("SELECT count(*) FROM findings").fetchone()[0] == 1
    assert db.get_meta(conn, "scratchpad") == "after https://seed-a"
