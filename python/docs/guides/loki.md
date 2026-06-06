# Loki — the global yggdrasil agent

**Loki** is one agent that adapts to wherever it runs. It detects the
backends it can reach, acts as a **token / credential provider** for them
(chiefly Databricks — when a session is present Loki hands its
authenticated client to whatever it drives), and dispatches pluggable
**behaviors**. Drive it from code (`yggdrasil.loki.Loki`) or the terminal
(`ygg loki`).

```python
from yggdrasil.loki import Loki

loki = Loki.current()
loki.card()              # identity + reachable backends + behaviors
loki.databricks          # the live DatabricksClient, or None
loki.run("genie", question="revenue by region last month")
```

## Capability detection

On wake-up Loki sniffs its environment **offline** (no network, never
raises) and reports the backends it can lean on:

| Backend | Detected from |
|---|---|
| `databricks` | the Databricks runtime, `DATABRICKS_*` env vars, `~/.databrickscfg`, or a session remembered by `ygg databricks configure` |
| `node` | a configured yggdrasil node home (`~/.node` / `YGG_NODE_HOME`) |
| `local` | always available |

```python
loki.backends()          # [Backend('databricks', True, {...}), ...]
loki.has("databricks")   # bool
```

## Token provider

When a Databricks session is detected, `loki.databricks` is the
authenticated client — Loki reasons against Databricks service endpoints
(SQL, Genie, jobs, serving) under the agent's resolved credentials.
`loki.token_info()` returns a **non-secret** summary (host, auth type,
catalog/schema); the token itself stays inside the client.

## Behaviors

A `LokiBehavior` is one discoverable, environment-aware action. Behaviors
declare a backend they `require` (so a Databricks-only behavior stays dark
on a bare shell) and register into a global catalog:

```python
from yggdrasil.loki import LokiBehavior, register

@register
class Hello(LokiBehavior):
    name = "hello"
    description = "say hi"
    def run(self, agent, *, who="world", **_):
        return f"hello {who} from {agent.user}@{agent.host}"
```

```python
loki.behaviors()                 # the catalog
loki.run("hello", who="loki")    # dispatch (guards availability first)
```

Built-in behaviors:

- **`genie`** — asks a Genie space a question (autonomously picking the first
  reachable space when none is named), returning the rows as tabular.
- **`code-project`** — reasons out a small Python program from a spec, writes
  the project, and **runs it**, returning the program's output:

  ```python
  ans = loki.run("code-project", spec="print the first 10 primes", name="primes")
  ans.text                 # the program's stdout
  ans.meta["project"]      # where the project was written
  ```

## Reasoning engines (`TokenEngine`)

A `TokenEngine` is the LLM contract Loki reasons on — the seam between the
agent and whatever model backs it. Each declares an `EngineType`
(`openai` / `claude` / `databricks` / `loki_agent`). Built in:

| Engine | Backend | Credentials |
|---|---|---|
| `ClaudeEngine` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `OpenAIEngine` | OpenAI Chat Completions | `OPENAI_API_KEY` |
| `DatabricksServingEngine` | a Databricks serving endpoint | the Databricks session (no extra key) |
| `LokiAgentEngine` | another Loki agent (delegation) | — |

```python
loki.engines()                       # all known, call .available() to filter
loki.engine()                        # the best available (preference order)
loki.engine("claude")                # a specific one
r = loki.reason("summarize today's failed jobs", system="be terse")
r.text                               # the reply (an AgentResponse)
```

`Loki.ENGINE_PREFERENCE` picks the engine when none is named (`claude` →
`openai` → `databricks` for the global agent). Implement `TokenEngine` to
add another backend.

### Models adapt to complexity (`TokenModel`)

`TokenModel` is the model catalog, grouped by `Provider` and `Complexity`
(`low`/`medium`/`high`). An engine adapts the model it uses to the task:

```python
loki.reason("classify this ticket", complexity="low")    # → a cheap/fast model
loki.reason("design a migration plan", complexity="high")  # → the most capable
```

The catalog is extensible — `register_model(...)` / `unregister_model(...)`
add or drop models; `select_model(provider, complexity)` picks the fit.

### Tabular answers (`AgentResponse`)

`reason()` and behaviors return an `AgentResponse` — narrative `text` plus,
when the result is tabular-like, an optional `tabular` frame (Arrow /
Polars / pandas). The `genie` behavior attaches the rows it computed:

```python
ans = loki.run("genie", question="top 10 customers by revenue")
ans.text                # Genie's narrative
ans.is_tabular          # True when it answered with SQL
ans.tabular             # the rows (Polars), or ans.to_polars()
```

## Replication — local parallel agents

A Loki replicates by **forking copies of itself into separate local
processes**, each running a behavior in parallel; it monitors them and
collects their `AgentResponse`s. This is on-machine parallelism, not remote
jobs.

```python
rep = loki.spawn("genie", question="failures today")   # a child process
rep.status                                              # running → done/failed
rep.result(timeout=60)                                  # → AgentResponse

reps = loki.map("genie", questions, arg="question")     # fan-out
answers = loki.gather(reps)                             # collect all
```

Forked children inherit the live agent — its registered behaviors, detected
backends, and engine — so a replica is a true copy of the current agent.

## DatabricksLoki — the specialized agent

`yggdrasil.databricks.loki.DatabricksLoki` is a Loki that lives on
Databricks at most:

- **Detects its workspace only from the `ygg databricks configure` session**
  (the remembered profile/host) — never from `DATABRICKS_*` env vars or hard
  parameters.
- **Reasons through a Databricks serving endpoint** by default
  (`ENGINE_PREFERENCE = ("databricks", "claude", "openai")`; override the
  endpoint with `serving_endpoint=` or `YGG_LOKI_SERVING_ENDPOINT`).
- **Replicates locally** like every Loki — `spawn` / `map` / `gather` fork
  child agent processes on the current machine.

```python
from yggdrasil.databricks.loki import DatabricksLoki

loki = DatabricksLoki.current()
loki.databricks                      # client from the configure profile, or None
loki.reason("which jobs failed in the last hour?")
loki.gather(loki.map("genie", questions, arg="question"))
```

## CLI

```bash
ygg loki                  # status: identity + backends + engines + behaviors
ygg loki capabilities     # the detected backends and their signals
ygg loki engines          # the reasoning engines and which are available
ygg loki behaviors        # the registered behavior catalog
ygg loki token --probe    # the Databricks credentials Loki provides
ygg loki reason "summarize failed jobs" --system "be terse"
ygg loki run genie --kwarg question='"top customers"' --json
```

`run` JSON-decodes each `--kwarg` value, so quote strings (`'"..."'`) and
pass numbers/booleans/objects directly.
