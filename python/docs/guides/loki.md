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
| `ClaudeEngine` | Anthropic Messages API (Haiku ↔ Opus, adaptive) | `ANTHROPIC_API_KEY`, **or** a Claude Code OAuth login (no key) |
| `OpenAIEngine` | OpenAI Chat Completions (mini ↔ 4o, adaptive) | `OPENAI_API_KEY` |
| `DatabricksServingEngine` | a Databricks serving endpoint | the Databricks session (no extra key) |

### Reasoning with Claude **without an API key**

`ClaudeEngine` can authenticate with the same OAuth / subscription token
Claude Code itself logs in with — so on a machine where you're signed into
Claude Code, Loki reasons through Claude with **no separate billed API
key**, which is the cheap path for local testing. It resolves the token, in
order, from `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`, or the Claude
Code credentials file (`~/.claude/.credentials.json` → `claudeAiOauth`). A
real `ANTHROPIC_API_KEY` still wins when both are present. The OAuth request
carries the `oauth-2025-04-20` beta header and leads its system prompt with
the Claude Code identity the grant is scoped to.

```python
ClaudeEngine().available()    # True on a logged-in Claude Code box, no key set
ClaudeEngine().uses_oauth     # True when it will use the subscription token
loki.act("refactor utils.py", root=".")   # the agent loop, reasoning keyless
```

```python
loki.engines()                       # all three, call .available() to filter
loki.engine()                        # the best available (preference order)
loki.engine("claude")                # a specific one
loki.reason("summarize today's failed jobs", system="be terse")
```

`Loki.ENGINE_PREFERENCE` picks the engine when none is named (`claude` →
`openai` → `databricks` for the global agent). Implement `TokenEngine` to
add another backend.

### Adaptive model selection (the default)

Each engine declares a small **tier map** — a `fast` model and a `deep`
(more capable) one — and **by default the model is chosen adaptively**: a
short, light request resolves to the fast model; a long or reasoning-heavy
one (sized on the message content, plus signal words like *refactor*,
*debug*, *design*, *optimize*) resolves to the deep model. This keeps cheap
turns cheap without ever capping the hard ones.

| Engine | `fast` | `deep` |
|---|---|---|
| `ClaudeEngine` | `claude-haiku-4-5` | `claude-opus-4-8` |
| `OpenAIEngine` | `gpt-4o-mini` | `gpt-4o` |
| `DatabricksServingEngine` | (pinned to one workspace endpoint — no fast/deep pair) | — |

Adaptivity is only the **default**, never an override of an explicit
decision:

```python
loki.reason("classify this ticket")                 # short → fast (Haiku)
loki.reason("refactor the planner for correctness")  # signalled → deep (Opus)
loki.reason("anything", tier="deep")                 # force the deep tier
loki.engine("claude").model = "claude-sonnet-4-6"    # a hard pin always wins
loki.act("tidy utils.py", tier="fast")               # pin the whole agent loop
```

In the agent loop, leaving `tier` unset lets each turn adapt on its own —
cheap scouting turns early, the capable model once the transcript (and the
reasoning) grows. `ygg loki reason` and `ygg loki do` take `--tier fast|deep`;
`ygg loki engines` shows each engine's adaptive ceiling (e.g.
`claude-opus-4-8 (adaptive)`). Override `TokenEngine.choose_tier` for a
smarter policy.

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

## Interactive session

On a terminal, bare `ygg loki` opens a **modern interactive session** (a
pipe / CI / Databricks job falls back to a static `status`). Type a request
and Loki **routes it**: it categorizes the problem and picks a solution path
— answer (`reason`), act on files (`act`), or ask Genie — and **isolates**
Databricks problems to the workspace-resident `DatabricksLoki` ("databricks
on databricks") when one is reachable. Each turn streams a live token/USD
KPI line.

```text
⟢ auto ›  how do I size a Databricks SQL warehouse?
  ▹ databricks · matched a Databricks/Unity/warehouse signal  →  databricks-loki (isolated)
  …answer…
  usage  ↑16 ↓111  127 tok  $0.0006  (+67)  49,873 left
```

Slash commands: `/status` `/engines` `/usage` `/tier fast|deep|auto`
`/root <dir>` `/budget [N|+N|off]` `/reset` `/help` `/quit`.

### Token budget

Every session starts with a default **token budget** (50 000 tokens). When
you reach it, Loki stops and asks — raise by one step (25 000), set a custom
cap, turn it off, or stop — so a runaway session never spends without bound:

```text
▲ token budget reached — 50,120 ≥ 50,000
  raise by 25,000 [Enter] · set N · off · stop [s]:
```

## Token monitoring (`ygg loki usage`)

Every engine records its spend into a process-global meter
(`yggdrasil.loki.METER`), keyed by `(engine, model)`. `ygg loki usage` (and
`/usage` in the session) shows the live KPIs — calls, input/output tokens,
total, and **USD** per model plus a global roll-up — priced from
per-model-per-engine defaults in `yggdrasil.loki.usage.PRICING` (Opus 4.8
$5/$25 per 1M, Haiku 4.5 $1/$5, gpt-4o $2.5/$10, …; retune per workspace).

```python
from yggdrasil.loki import METER
METER.set_limit(100_000)          # cap total tokens (None = unlimited)
loki.reason("…")                  # engines record automatically
METER.total().total_tokens        # global tokens; METER.total_cost → USD
engine.usage()                    # this engine's per-model rows
```

## CLI

```bash
ygg loki                  # interactive session on a terminal (else status)
ygg loki status           # identity + backends + engines + behaviors
ygg loki engines          # the reasoning engines and which are available
ygg loki usage            # live token + USD KPIs, per model and global
ygg loki capabilities     # the detected backends and their signals
ygg loki behaviors        # the registered behavior catalog
ygg loki tools            # the tools the autonomous agent acts through
ygg loki token --probe    # the Databricks credentials Loki provides
ygg loki reason "summarize failed jobs" --system "be terse"   # adaptive model
ygg loki reason "prove this refactor is safe" --tier deep      # force the deep tier
ygg loki do "fix the failing test in tests/test_io.py"
ygg loki run genie --kwarg question='"top customers"' --json
```

`run` JSON-decodes each `--kwarg` value, so quote strings (`'"..."'`) and
pass numbers/booleans/objects directly.
