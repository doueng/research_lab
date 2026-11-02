# Research Lab

**A stateful research agent built with LangChain, LangGraph and LangSmith.**

Research Lab hunts for *rare tokens*: sources that are dense with information and seen by few people, such as
mailing-list threads, commit discussions, practitioner blogs and personal config repos. It works on one lead per
run. Each run (a *tick*, or heartbeat) plans, investigates with a tool-using agent, records what it found, and
then rewrites its own working memory and re-ranks its leads for the next tick.

I built it to learn the LangChain stack by using it for real. Every major feature maps to a LangChain, LangGraph or
LangSmith concept, and the code stays small enough (about 530 lines of Python) to read in one sitting.

<p align="center"><img src="docs/img/architecture.svg" alt="Architecture: lab CLI drives a LangGraph StateGraph whose investigate node is a LangChain create_agent with Tavily and fetch_url tools; state lives in SQLite (dataroom and checkpoints); runs are traced to LangSmith" width="100%"></p>

## Highlights

| Area | What the project shows | Where |
|---|---|---|
| **LangGraph** | `StateGraph` over a `TypedDict` state, five nodes, compiled with a `SqliteSaver` checkpointer. One thread per tick, crash resume through `get_state().next` | `src/lab/graph.py` |
| **LangChain agents** | `create_agent` with tools, `ToolCallLimitMiddleware` for a hard tool budget, and `ToolStrategy` for a Pydantic-validated final answer | `src/lab/graph.py` |
| **Structured output** | `with_structured_output` with Pydantic schemas (`Plan`, `Investigation`, `Reflection`) at every LLM boundary. No free-text parsing | `src/lab/graph.py` |
| **Tools** | `TavilySearch` from `langchain-tavily`, plus a custom `@tool fetch_url` that returns page text wrapped as untrusted data | `src/lab/tools.py` |
| **Models** | `init_chat_model` for any provider, plus `ChatCodex`, a `ChatOpenAI` subclass that adapts the request payload for the ChatGPT Codex backend | `src/lab/models.py` |
| **LangSmith** | Every tick is traced to the `research-lab` project through environment variables only. Human ratings are already stored as ground truth for planned dataset-based evaluations | `.env.example`, `lab rate` |
| **Testing** | Deterministic tests with LangChain's `FakeMessagesListChatModel` and an `InMemorySaver`. No network and no API keys | `tests/` |

## How a tick works

<p align="center"><img src="docs/img/tick.svg" alt="One tick: load, plan, investigate, record, reflect, with reflect feeding the next tick, and a LangGraph checkpointer saving state after every node" width="100%"></p>

A tick is one run of a compiled LangGraph `StateGraph`:

| Node | Kind | What it does |
|---|---|---|
| `load` | Python | Reads the objective, the scratchpad and the top 15 open leads from the SQLite dataroom |
| `plan` | LLM, `with_structured_output(Plan)` | Picks exactly one lead, or about one tick in five proposes a new direction to explore. Falls back to the top lead if the model picks nothing usable |
| `investigate` | LangChain agent | Goes deep on that one lead with search and fetch tools (see below) |
| `record` | Python | Writes findings and new leads with `INSERT OR IGNORE`, so replaying the node can never create duplicates |
| `reflect` | LLM, `with_structured_output(Reflection)` | Rewrites the bounded scratchpad (max 2500 chars) and re-scores open leads. This is the *self-editing* part |

**Durable execution.** The graph is compiled with a `SqliteSaver` checkpointer, and each tick runs on its own thread
(`thread_id = "tick-<n>"`). If a tick crashes, for example on an expired token or a malformed model reply, the next
`lab tick` checks `graph.get_state(config).next` and calls `graph.invoke(None, config)` to resume at the failed node.
Nodes that already ran, including their paid LLM calls, are not repeated.

**Two kinds of memory, on purpose.** The agent keeps a free-form scratchpad and a structured, prioritised frontier
of leads. Which one actually drives better research is an open question that the planned LangSmith evaluation is
meant to answer.

## The investigator agent

<p align="center"><img src="docs/img/investigator.svg" alt="The investigate node runs a LangChain create_agent: a model and tools loop capped by ToolCallLimitMiddleware, returning a ToolStrategy(Investigation) structured response, followed by guardrails" width="100%"></p>

The `investigate` node runs a LangChain agent built once per graph:

```python
investigator = create_agent(
    model,
    tools,                                                   # TavilySearch, fetch_url
    middleware=[ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS)],
    response_format=ToolStrategy(Investigation),             # findings, new_leads, summary
)
```

Design choices, each driven by something that went wrong or could go wrong:

- **Bounded cost.** `ToolCallLimitMiddleware` caps each tick at 12 tool calls.
- **Recovering a missing answer.** Live runs showed that after the middleware's budget message the model can stop
  with plain text and never call the `Investigation` tool. The node then makes one extra
  `with_structured_output(Investigation)` call over the conversation so far. A regression test covers this case.
- **No invented sources.** A finding is kept only if its URL appears in a real tool result from this tick. The
  `ToolStrategy` echo message is excluded, so the model cannot cite its own answer as evidence.
- **Prompt-injection hygiene.** Fetched pages are wrapped in `<untrusted_page>` tags, and the system prompt tells the
  agent never to follow instructions found in tool output. The tools are read-only (no shell, no file writes).

## Observability and evaluation with LangSmith

<p align="center"><img src="docs/img/langsmith.svg" alt="Simplified LangSmith trace tree of one tick, what tracing answers, and the planned evaluation loop from human ratings to a LangSmith dataset and experiments" width="100%"></p>

**Tracing (working).** Setting `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT=research-lab` in
`.env` is enough. LangChain and LangGraph pick these up, so each `lab tick` appears as one trace with the graph nodes,
the nested agent graph, every model call and every tool call, including tokens, latency and errors. I use the traces
to see why `plan` chose a lead, how the agent spent its tool budget, and where a crashed tick failed.

**Evaluation (planned, the next milestone).** The data for it is already being collected. `lab rate ID 1-5` stores my
rating on each finding. The plan:

1. Export rated findings into a LangSmith dataset (`Client.create_dataset`, `create_examples`).
2. Run `evaluate()` on three variants of the agent: scratchpad only, frontier only, and both.
3. Compare them in the LangSmith experiments view, with my ratings as the reference and an LLM-as-judge evaluator for
   "is this source rare and high-signal".

## Setup

<p align="center"><img src="docs/img/setup.svg" alt="Setup in six steps: Guix toolchain, Python env with uv, .env credentials, seed a subject, run a tick, inspect and steer" width="100%"></p>

### 1. Toolchain and Python environment

`manifest.scm` supplies Python 3.12, uv, gcc-toolchain, sqlite, git and ruff. `dev/env.sh` builds that profile, runs
`uv sync --frozen` into `.venv` using only the Guix Python, and activates it. Enter it once with `direnv allow`, or
from bash with `source dev/env.sh`. Python dependencies are pinned in `uv.lock`, since Guix does not package LangChain.

Without Guix, any Python 3.12 with uv should work: `uv sync --frozen && source .venv/bin/activate`.

### 2. Credentials

`cp .env.example .env` and fill it in. `.env` is gitignored and loaded by `lab`. Variables already set in the
environment win.

| Variable | Purpose |
|---|---|
| `LAB_MODEL` | `codex:<model>` uses a ChatGPT subscription through the Codex backend. Any other spec, for example `openai:gpt-...`, goes to `init_chat_model` and needs that provider's API key |
| `CODEX_AUTH_FILE` | Login file for the Codex backend. Read-only: the token is never copied or refreshed by the lab |
| `TAVILY_API_KEY` | Web search for the agent |
| `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` | Tracing to LangSmith |

### 3. Run it

```sh
lab subjects                   # list seeds/*.toml and which are seeded
lab -s guix seed               # load seeds/guix.toml into data/guix.db (idempotent)
lab -s guix tick               # one research iteration on exactly one lead
lab status                     # scratchpad + top of the frontier
lab findings                   # what has been saved
lab rate 3 5                   # your 1-5 rating: ground truth for evaluations
lab lead URL "why" 90          # steer it by hand
sqlite3 data/guix.db           # each subject's dataroom is plain SQLite
pytest -q                      # deterministic tests, no API keys needed
```

Every command takes `-s/--subject` (default `$LAB_SUBJECT`, else `guix`). If a tick crashes, run `lab tick` again and
it resumes from the last checkpoint in `data/<subject>.ckpt.db`.

## Research subjects

A subject is a seeds file, `seeds/<name>.toml`, with an objective and a list of starting leads. Subjects never share
state: each has its own objective, scratchpad, frontier, findings, tick numbering and checkpoint database.
Shipped subjects are `guix` (the default), `memory`, `hypertrophy` and `london`. To add one, copy an existing seeds
file and run `lab -s <name> seed`.

```toml
[lab]
objective = """
Build deep, practical understanding of GNU Guix as a daily-driver OS ...
Prefer rare, high-signal sources: guix-devel threads, commit reviews, personal config repos ...
"""

[[lead]]
target = "https://lists.gnu.org/archive/html/guix-devel/"
reason = "guix-devel archive. Design arguments and RFCs live here, not in the manual."
priority = 80
```

## Testing

`tests/test_tick.py` drives the real graph with a scripted `FakeMessagesListChatModel` whose replies are tool calls
for `Plan`, `fetch_url`, `Investigation` and `Reflection`, plus stub tools and an `InMemorySaver`. The tests cover:

- two ticks continuing across fresh graphs and connections, with findings and leads deduplicated
- a fabricated URL being filtered out of the findings
- a tick that crashes before `reflect`, then resumes from the checkpoint and runs only `reflect`
- an agent that ends without a structured response, recovered by the fallback call

`tests/test_subjects.py` runs the real CLI entry point. It checks that subjects get separate datarooms, that the
default subject comes from `LAB_SUBJECT`, that unknown subjects are refused, and that every shipped seeds file loads.

## Project layout

```
src/lab/graph.py    LangGraph StateGraph, Pydantic schemas, investigator agent, run_tick
src/lab/tools.py    fetch_url tool and the Tavily search tool
src/lab/models.py   LAB_MODEL -> chat model (init_chat_model or ChatCodex)
src/lab/db.py       SQLite dataroom: meta, leads, findings, ticks
src/lab/cli.py      the lab command
seeds/              one TOML file per research subject
tests/              deterministic tests with fake models
dev/                Guix dev environment and an LSP check
docs/img/           the diagrams in this README
```

## Editor and linting

The Python LSP is `pylsp` from Guix Home. After `direnv allow`, Emacs envrc-mode exports this project's `VIRTUAL_ENV`,
so pylsp resolves imports from `.venv`. Linting is ruff, configured in `pyproject.toml`. In Emacs, lsp-mode's ruff
client starts because ruff is on `PATH`, and `.dir-locals.el` turns off pylsp's own linters.
`python dev/lsp_check.py` checks go-to-definition and diagnostics the way the editor sees them.

## Roadmap

1. **LangSmith evaluation.** Rated findings become a dataset, then compare scratchpad-only, frontier-only and both.
2. **Human in the loop.** `interrupt()` after `plan`, so I can approve or redirect the lead before tokens are spent.
3. **Personal context.** A read-only tool over my own Guix config, so leads become relevant to my setup.
4. **RAG over the dataroom.** Embeddings and a retriever over saved findings.
5. **Deep agents.** Rebuild the investigator with `create_deep_agent` and compare it against `create_agent` in LangSmith.
