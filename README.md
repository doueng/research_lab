# Research Lab

A seeded, stateful research agent that hunts for *rare tokens* (high-signal, little-seen sources).
The research subject is GNU Guix and Linux configs built on it. The real goal is learning LangChain and LangGraph.
It is based on the "self-editing heartbeats with seeds" idea, but runs are manual: one `lab tick` = one heartbeat.

## Use

```sh
export ANTHROPIC_API_KEY=...            # or LAB_MODEL=openai:gpt-... with langchain-openai added
uv run lab seed                         # load seeds.toml (idempotent)
uv run lab tick                         # one research iteration on exactly one lead
uv run lab status                       # scratchpad + frontier
uv run lab findings                     # what has been saved
uv run lab rate 3 5                     # your 1-5 rating: the ground truth for future evals
uv run lab lead URL "why" 90            # steer it by hand
sqlite3 lab.db                          # the dataroom is plain SQLite
uv run pytest -q                        # deterministic tests, no API key needed
```

If a tick crashes, the next `lab tick` resumes it from the last checkpointed node (lab.ckpt.db).

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
