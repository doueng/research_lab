# Research Lab

A seeded, stateful research agent that hunts for *rare tokens* (high-signal, little-seen sources).
It can research several subjects, each defined by a seeds file (`seeds/<name>.toml`: objective plus starting leads).
Shipped: `guix` (the default), `memory`, `hypertrophy`, `london`. The real goal is learning LangChain and LangGraph.
It is based on the "self-editing heartbeats with seeds" idea, but runs are manual: one `lab tick` = one heartbeat.

## Dev environment (Guix)

`manifest.scm` supplies python 3.12, uv, gcc-toolchain, sqlite and git. `dev/env.sh` builds that profile,
runs `uv sync --frozen` into `.venv` (using only the Guix python) and activates it. Enter it with `direnv allow`
once, or from bash with `source dev/env.sh`. Python deps live in `uv.lock`, since Guix does not package LangChain.

## Editor / LSP

Python LSP is `pylsp` from Guix Home. With `direnv allow` done, Emacs envrc-mode exports this project's
`VIRTUAL_ENV`, so pylsp resolves imports from `.venv`. Linter settings live in `setup.cfg`.
`python dev/lsp_check.py` (inside the dev env) checks go-to-definition and diagnostics the way the editor sees them.

## Credentials (.env)

`cp .env.example .env` and fill it in. `.env` is gitignored and loaded by `lab` (already-set env vars win).

- `LAB_MODEL=codex:gpt-6-sol` uses your ChatGPT subscription through the Codex backend. The token is read
  from Pi's login file (`CODEX_AUTH_FILE`) and never copied, because it expires and Pi rotates it. If it has
  expired, start `pi` once. Other specs (e.g. `openai:gpt-...`) go to `init_chat_model` and need API keys.
- `TAVILY_API_KEY` for web search.
- `LANGSMITH_*` for tracing. Each tick shows up as a `LangGraph` trace in the `research-lab` project.

## Use

Inside the dev environment (`.venv` active):

```sh
lab subjects                   # list seeds/*.toml and which are seeded
lab -s memory seed             # load seeds/memory.toml into data/memory.db (idempotent)
lab -s memory tick             # every command takes -s/--subject (default $LAB_SUBJECT or guix)
lab seed                       # load seeds/guix.toml
lab tick                       # one research iteration on exactly one lead
lab status                     # scratchpad + frontier
lab findings                   # what has been saved
lab rate 3 5                   # your 1-5 rating: the ground truth for future evals
lab lead URL "why" 90          # steer it by hand
sqlite3 data/guix.db           # each subject's dataroom is plain SQLite
pytest -q                      # deterministic tests, no API key needed
```

Subjects never share state: each has its own objective, scratchpad, frontier, findings and tick numbering.
To add one, write `seeds/<name>.toml` (copy an existing one) and run `lab -s <name> seed`.

If a tick crashes, the next `lab tick` resumes it from the last checkpointed node (data/<subject>.ckpt.db).

## How a tick works (src/lab/graph.py)

`load -> plan -> investigate -> record -> reflect`

| Node | LangChain / LangGraph concept |
|---|---|
| plan | `with_structured_output(Plan)` picks ONE lead (the article's "bespoke action") |
| investigate | `create_agent` + tools, `ToolCallLimitMiddleware`, `ToolStrategy(Investigation)` structured response |
| record | plain-Python node, idempotent SQLite writes (safe to replay) |
| reflect | rewrites the bounded scratchpad and re-scores leads (the "self-editing" part) |
| whole graph | `StateGraph`, one checkpointer thread per tick, crash resume via `get_state().next` |

Memory is kept in two forms on purpose: a free-form scratchpad (the article's approach) and a
structured frontier of leads. Comparing them is a planned evaluation, not a settled choice.

## Next lessons (in rough order)
1. Turn on LangSmith tracing (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`) and read a tick's trace.
2. `interrupt()` after plan so you approve or redirect the lead (human-in-the-loop).
3. A read-only tool over your own `~/guix` config so leads become relevant to your setup.
4. Rated findings -> LangSmith dataset -> evaluate scratchpad-only vs frontier-only vs both.
5. Embeddings + retriever over saved findings (RAG over the dataroom).
6. Rebuild the investigator with `create_deep_agent` and compare.
