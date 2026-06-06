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

The built-in **`genie`** behavior is the reference for the *token-provider*
pattern: it guards on the `databricks` backend, then asks a Genie space a
question (autonomously picking the first reachable space when none is
named). The **`agent`** behavior (below) is the reference for *autonomy*.
Replication, inter-agent messaging, HTTP ingestion and serving land on this
abstraction next.

## Acting autonomously (`loki.act` / `ygg loki do`)

`reason()` is one shot. `act()` is the agent **on its own**: a
reason → act → observe loop where Loki's engine plans against a tool
catalog, emits **one JSON tool call per turn**, and Loki runs it and feeds
the observation back — discovering the project and modifying files itself —
until it declares it's done (or hits the step budget).

```python
result = loki.act(
    "find the failing assertion in tests/ and fix it",
    root=".",            # the working tree the agent is confined to
    max_steps=12,        # tool-call budget
    read_only=False,     # set True for discovery only (no writes)
    allow_shell=False,   # opt in to give it a shell tool too
)
result["files_changed"]  # ['tests/test_foo.py', ...]
result["answer"]         # the agent's summary of what it did
result["steps"]          # full transcript: thought / tool / args / observation
```

### Tools — the agent's hands

The toolbox (`yggdrasil.loki.filesystem_toolbox`) is **confined to `root`**:
a path that resolves outside it is refused, so an autonomous run can touch
the project it was pointed at and nothing above it. Mutating tools record
what they changed.

| Tool | What it does | Group |
|---|---|---|
| `list_dir` | list a directory | discovery |
| `read_file` | read a file (optional line range) | discovery |
| `find` | find files by glob | discovery |
| `grep` | search file contents by regex | discovery |
| `write_file` | create/overwrite a file | write (skipped when `read_only`) |
| `edit_file` | replace one unique occurrence | write (skipped when `read_only`) |
| `run` | run a shell command in `root` | shell (only when `allow_shell`) |

Implement your own `Tool`s and pass a custom `Toolbox` to `act(toolbox=…)`
to give the agent different hands.

```bash
ygg loki tools                         # the catalog (✎ marks mutating tools)
ygg loki do "add a __repr__ to the Backend dataclass"
ygg loki do "audit imports" --read-only
ygg loki do "format the package" --allow-shell --max-steps 20
ygg loki do "tidy the docs" --json     # full transcript as JSON
```

`do` streams each step (the tool call, the agent's one-line thought, the
head of the observation), then prints the files it changed and the agent's
summary.

## Reasoning engines (`TokenEngine`)

A `TokenEngine` is the LLM contract Loki reasons on — the seam between the
agent and whatever model backs it. Three are built in:

| Engine | Backend | Credentials |
|---|---|---|
| `ClaudeEngine` | Anthropic Messages API (`claude-opus-4-8`) | `ANTHROPIC_API_KEY` |
| `OpenAIEngine` | OpenAI Chat Completions | `OPENAI_API_KEY` |
| `DatabricksServingEngine` | a Databricks serving endpoint | the Databricks session (no extra key) |

```python
loki.engines()                       # all three, call .available() to filter
loki.engine()                        # the best available (preference order)
loki.engine("claude")                # a specific one
loki.reason("summarize today's failed jobs", system="be terse")
```

`Loki.ENGINE_PREFERENCE` picks the engine when none is named (`claude` →
`openai` → `databricks` for the global agent). Implement `TokenEngine` to
add another backend.

## DatabricksLoki — the specialized agent

`yggdrasil.databricks.loki.DatabricksLoki` is a Loki that lives on
Databricks at most:

- **Detects its workspace only from the `ygg databricks configure` session**
  (the remembered profile/host) — never from `DATABRICKS_*` env vars or hard
  parameters.
- **Reasons through a Databricks serving endpoint** by default
  (`ENGINE_PREFERENCE = ("databricks", "claude", "openai")`; override the
  endpoint with `serving_endpoint=` or `YGG_LOKI_SERVING_ENDPOINT`).
- **Deploys to Databricks**: `deploy()` upserts a serverless Job that runs the
  agent on the pre-built ygg image through the single `ygg` wheel entry point
  (`ygg loki reason …` / `ygg loki run …` — on the runtime `ygg loki` resolves
  to this `DatabricksLoki`).

```python
from yggdrasil.databricks.loki import DatabricksLoki

loki = DatabricksLoki.current()
loki.databricks                      # client from the configure profile, or None
loki.reason("which jobs failed in the last hour?")
job = loki.deploy(behavior="reason", prompt="nightly health check")
job.run()
```

## CLI

```bash
ygg loki                  # status: identity + backends + engines + behaviors
ygg loki capabilities     # the detected backends and their signals
ygg loki engines          # the reasoning engines and which are available
ygg loki behaviors        # the registered behavior catalog
ygg loki tools            # the tools the autonomous agent acts through
ygg loki token --probe    # the Databricks credentials Loki provides
ygg loki reason "summarize failed jobs" --system "be terse"
ygg loki do "fix the failing test in tests/test_io.py"
ygg loki run genie --kwarg question='"top customers"' --json
```

`run` JSON-decodes each `--kwarg` value, so quote strings (`'"..."'`) and
pass numbers/booleans/objects directly.
