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

## Skills

A `LokiSkill` is one discoverable, environment-aware capability. Skills
declare a backend they `require` (so a Databricks-only skill stays dark on a
bare shell) and register into a global catalog:

```python
from yggdrasil.loki import LokiSkill, register

@register
class Hello(LokiSkill):
    name = "hello"
    description = "say hi"
    def run(self, agent, *, who="world", **_):
        return f"hello {who} from {agent.user}@{agent.host}"
```

```python
loki.skills()                    # the catalog
loki.run("hello", who="loki")    # dispatch (guards availability first)
```

The backend-agnostic catalog is `agent` (autonomy, below), `web`, `tabular` /
`transform` (the data path), `python_project`, `setup`, and `guide` (below).
Backend-specialized skills register only when their backend is reachable —
`genie` / `databricks-*` from `yggdrasil.databricks.loki`, `aws-*` from
`yggdrasil.aws.loki`.

**Preprompts.** A skill that reasons through an engine carries a domain
`preprompt` — a short system prompt tuned to steer the model toward the best
result *and* to lean on yggdrasil's own features for that domain. The
`python_project` skill prompts for code that uses `IO`/`HTTPSession`/`DataType`
(and grounds the request in the matching `guide` recipes); the Databricks fleet
carries a "prefer serverless / Unity Catalog / Arrow / seeded wheel envs"
preprompt that `databricks-serving` sends as the served model's system message;
`web` answers strictly from the fetched page. It's `LokiSkill.preprompt` — set
it on any skill that calls an engine.

### `guide` — the optimized yggdrasil way

Loki knows the project it lives in. The **`guide`** skill is its
*do-it-the-yggdrasil-way* adviser: for a task it matches a curated set of
recipes (`yggdrasil.loki.guides.GUIDES`) — naming the right abstraction (the io
handlers, `HTTPSession`, `Field`/`DataType` casting, the `dbc` accessors,
`Tabular.display`, …), the idiomatic snippet, and the hand-rolled anti-pattern to
avoid — and, with `plan=True`, has the engine synthesize a concrete plan
**grounded only in those features**.

```python
loki.run("guide", task="fetch a CSV from an API and cache it typed")
#   → recipes: http-fetch (HTTPSession → to_polars), tabular-io (IO.from_),
#     schema-cast (Field/DataType)
loki.run("guide", topic="databricks-compute")   # a specific recipe
```

In the session, asking *how* to build something the yggdrasil way routes here
automatically: `best way to … in yggdrasil`, `the idiomatic way to … with ygg`.
It's data over code — add a `Guide` to `GUIDES` and it's instantly discoverable.

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
| `read_table` | parse a local CSV/Parquet/Arrow/XLSX/JSON via the io handlers | discovery |
| `write_file` | create/overwrite a file | write (skipped when `read_only`) |
| `edit_file` | replace one unique occurrence | write (skipped when `read_only`) |
| `run_python` | **write & run Python in `root`** (compute or apply changes) — on by default | write |
| `run` | **run a shell command in `root`** — on by default | write |

The agent **codes in its reasoning by default**: `run_python` lets it write
and execute Python to compute, transform, or apply changes without a shell.
A `confirm` callback gates **destructive ops on non-temporary assets** —
overwriting or editing an existing file outside the system temp dir asks
first (the interactive session prompts `⚠ confirm …? [y/N]`); new files and
scratch/temp files don't. So Loki runs autonomously and only stops the user
for credentials, clarifications, and destructive changes to real assets.

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

### Use case — look up live external information

With `--allow-web` (and `run_python`, on by default) the agent **browses the
internet and computes** on its own. Ask for the EUR/USD move over two weeks:

```bash
ygg loki do "Report how EUR/USD changed over the last 2 weeks. Use web_fetch \
on the Frankfurter API: GET /v1/latest?base=EUR&symbols=USD for the latest, \
then GET /v1/<start>..<end>?base=EUR&symbols=USD with start = end − 14 days." \
  --allow-web
```

Loki runs the loop unattended — `web_fetch` the latest rate, `web_fetch` the
date-range series, `run_python` to compute — and reports, e.g.:

```text
 1 web_fetch(https://api.frankfurter.dev/v1/latest?base=EUR&symbols=USD)   → 200 application/json
 2 web_fetch(https://api.frankfurter.dev/v1/2026-05-22..2026-06-05?…)       → 200 application/json
 3 run_python(start=1.1595; end=1.164; change=end-start; pct=change/start*100; print(...))
 ✓ EUR/USD over the last 2 weeks: start 1.1595, end 1.164, change +0.0045 (+0.39%).
 usage  ↑3,924 ↓479  4,403 tok  $0.0011
```

The same in code: `loki.act(task, allow_web=True)`, or one fetch at a time
with `loki.run("web", url=…)` / `web.read_json(url)`.

## Reasoning engines (`TokenEngine`)

A `TokenEngine` is the LLM contract Loki reasons on — the seam between the
agent and whatever model backs it. Engines are either **remote** (hosted
APIs — capable and metered) or **local** (run on this workstation — free,
private, but bounded by the machine; `TokenEngine.local` says which). Five
are built in:

| Engine | Local? | Backend | Credentials / requirements |
|---|---|---|---|
| `ClaudeEngine` | remote | Anthropic Messages API (Haiku ↔ Opus, adaptive) | `ANTHROPIC_API_KEY`, **or** a Claude Code OAuth login (no key) |
| `OpenAIEngine` | remote | OpenAI Chat Completions (mini ↔ 4o, adaptive) | `OPENAI_API_KEY` |
| `DatabricksServingEngine` | remote | a Databricks serving endpoint | the Databricks session (no extra key) |
| `TransformersEngine` | **local** | an open HuggingFace model via the `transformers` text-generation pipeline (Qwen2.5 0.5B ↔ 1.5B by default) | `transformers` + `torch` installed; `YGG_LOKI_HF_MODEL` / `YGG_LOKI_HF_DEVICE` override |
| `OllamaEngine` | **local** | a model served by a local [Ollama](https://ollama.com) server (`llama3.2:1b` ↔ `llama3.2`) | a running Ollama server (`OLLAMA_HOST`, default `localhost:11434`); `YGG_LOKI_OLLAMA_MODEL` override |

Local engines are **free** (priced at `$0` in the token meter) and keep data
on the box; pick them up by `ollama pull <model>` or installing
`transformers`+`torch`. They're chosen automatically for simple work when the
workstation can run them — see *Resource-aware local vs remote* below.

### A free local brain, **sized to your workstation** (`ygg loki setup`)

A local model is bounded by the box it runs on, so Loki sizes its `bootstrap_model`
to the machine (`yggdrasil.loki.resources` — CPU/RAM/GPU → a size tier): the more
muscle, the larger the default. It climbs a ladder of Qwen2.5 instruct models:

| resource tier | gate | Ollama | HF transformers |
|---|---|---|---|
| `small` | ≥ 8 GB CPU | `qwen2.5:3b` | Qwen2.5-1.5B |
| `medium` | ≥ 16 GB | `qwen2.5:7b` | Qwen2.5-3B |
| `large` | ≥ 32 GB | `qwen2.5:14b` | Qwen2.5-7B |
| `xlarge` | CUDA GPU | `qwen2.5:32b` | Qwen2.5-14B |

(`ygg loki engines` shows the pick, e.g. `qwen2.5:7b (resources: medium)`.) An
explicit pin (`OllamaEngine(model=…)`, `YGG_LOKI_OLLAMA_MODEL`/`YGG_LOKI_HF_MODEL`)
always wins.

**Accelerator auto-detection.** When `YGG_LOKI_HF_DEVICE` is unset the
`transformers` engine loads onto the best device it can find — NVIDIA `cuda`,
**Intel GPU** `xpu` (Arc / integrated Xe), or Apple `mps` — instead of crawling
on the CPU (`yggdrasil.loki.resources.accelerator()`). An **Intel NPU** (AI
Boost) is detected too (`resources.has_npu()`, via OpenVINO) and surfaced in
`ygg loki status` under `compute`, but the HF pipeline can't target it directly
— offload it with `optimum-intel` + `openvino`. The download forces the classic
LFS transfer (`HF_HUB_DISABLE_XET=1`) so a blocked xet CAS endpoint behind a
corporate proxy doesn't abort the weights fetch. `Loki.bootstrap_local()` readies the sized model **lazily** — it
pulls it only when missing (Ollama `POST /api/pull`), notes that HF weights
download on first use, or tells you what to install:

```text
ygg loki setup            # ready the resource-sized free local model
ygg loki setup qwen2.5:14b # ready a specific, heavier local model
```

```python
loki.bootstrap_local()                 # {"engine": "ollama", "ready": True, "model": "<sized>", ...}
loki.run("setup")                      # same, plus "redirects" for heavier setup
OllamaEngine().bootstrap_model         # the model this box gets
OllamaEngine().ensure()                # pull the sized model only if missing
```

The `setup` skill also surfaces **redirects** — what a lightweight model should
hand heavier work to: `ygg databricks configure` to set up Databricks, a larger
model id for more capability, or an API key so heavy reasoning escalates to a
remote engine automatically.

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
loki.engines()                       # all five, call .available() to filter
loki.engine()                        # the best available (preference order)
loki.engine("claude")                # a specific one
loki.select("classify this ticket")  # resource-aware local-vs-remote choice
loki.reason("summarize today's failed jobs", system="be terse")
```

`Loki.ENGINE_PREFERENCE` picks the engine when one is named or as the
fallback order — capable remote APIs first, then the free local engines
(`claude` → `openai` → `databricks` → `ollama` → `transformers` for the
global agent). Implement `TokenEngine` to add another backend.

### Resource-aware local vs remote

When no engine is named, `reason` / `reason_stream` / `act` route through
`Loki.select`, which weighs **task complexity** against **workstation
resources**:

- *Complexity* comes from an explicit `tier` (`deep` → complex) or, failing
  that, the prompt itself — long or reasoning-heavy text (the same signals
  that drive adaptive tiers) counts as complex.
- *Resources* are probed inline: any GPU accelerator (NVIDIA `cuda`, Intel
  `xpu`, or Apple `mps`), or enough CPU + RAM (≥ 4 cores and ≥ 8 GB).

Simple work on a capable box stays **local** (free, private); complex work,
or a thin machine, goes to the best available **remote** API. Either way it
falls back when one side is unreachable — local-only when no API is
configured, remote-only when no local engine is installed.

**Session-sticky base, with a confirm before escalating.** An interactive
session starts on a **base** provider (chosen at startup, e.g. Claude) and
sticks with it. Ordinary work runs on the cheapest capable option — a local
model when the box can host one — and only **heavy** work escalates up to the
capable remote. When that escalation means switching *from a free local model
up to a paid remote one* (e.g. Claude Opus), `select` asks first via a
`confirm(engine, model)` callback; decline and the work stays local. So the
small free model handles the light/setup tasks and you're only ever billed for
the big model when you say yes.

```python
loki.select("hi there", base="claude")             # light + capable box → ollama (local, free)
loki.select("refactor the planner", base="claude", # heavy → asks, then claude (remote)
            confirm=lambda eng, model: True)
loki.select("anything", tier="deep")               # forced complex → remote
```

In the REPL this is automatic: light prompts run free/local, and a heavy one
prints `⤴ escalate … switch up to claude … (remote, paid)? [Y/n]` before
spending. `/engine` sets the session base; `/setup` readies the free local
model.

### Adaptive model selection (the default)

Each **remote** engine declares a small **tier map** — a `fast` model and a
`deep` (more capable) one — and **by default the model is chosen adaptively**:
a short, light request resolves to the fast model; a long or reasoning-heavy
one (sized on the message content, plus signal words like *refactor*, *debug*,
*design*, *optimize*) resolves to the deep model. This keeps cheap turns cheap
without ever capping the hard ones.

| Engine | `fast` | `deep` |
|---|---|---|
| `ClaudeEngine` | `claude-haiku-4-5` | `claude-opus-4-8` |
| `OpenAIEngine` | `gpt-4o-mini` | `gpt-4o` |
| `DatabricksServingEngine` | (pinned to one workspace endpoint — no fast/deep pair) | — |

**Local** engines don't use the prompt's cost tier — they're bounded by the
machine, so they size to **resources** instead (the `bootstrap_model` ladder
above): the box, not the prompt, picks `TransformersEngine` / `OllamaEngine`'s
model.

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

Replies **stream live** — the chosen engine prints token deltas as they
arrive, not after a pause. At startup Loki **detects the configured engines**
(a Claude key or Claude Code login, a Databricks session, an OpenAI key) and
picks a default for the session; when several are configured it lets you
choose. Switch any time with `/engine claude|databricks|openai|auto`.

```text
  interactive session  · live prompts, streamed replies · /help · /quit
  engine databricks · databricks-meta-llama-3-1-8b-instruct
  ⟢ databricks·auto ›  In one short sentence, what is Apache Arrow?
  ▹ chat · no specialized signal — plain reasoning
  Apache Arrow is an open-source, cross-language in-memory columnar format…   ← streamed
  usage  ↑11 ↓33  44 tok  $0.0000  (+44)  49,956 left
```

Slash commands: `/engine` `/engines` `/status` `/usage` `/tier fast|deep|auto`
`/root <dir>` `/budget [N|+N|off]` `/reset` `/help` `/quit`. (A Databricks
serving engine self-heals to a deployed endpoint if its configured one is
missing.)

### Cost budget

Every session starts with a default **cost budget of $1** (USD spend, not
tokens — a fixed cap across models of very different per-token prices).
The cap is checked **between actions** (never mid-action), so a running
turn always finishes; when the spend crosses it Loki stops and asks — raise
by one **$1** step, set a custom cap, turn it off, or stop:

```text
▲ cost budget reached — $1.0042 ≥ $1.00
  raise by $1.00 [Enter] · set $N · off · stop [s]:
```

`/budget $5` sets it, `/budget +$2` raises it, `/budget off` removes the cap.

## On the internet (`yggdrasil.loki.web`)

Loki reaches the web through yggdrasil's own stack — every request rides
`HTTPSession` (pooling, retry, response cache) and every tabular body is
parsed by the **io handlers** (`HTTPResponse.to_polars()` auto-detects CSV /
JSON / Parquet / Arrow / XLSX). So "look it up" and "parse that table" are
the same abstractions the rest of yggdrasil runs on.

```python
from yggdrasil.loki import web

web.read_text("https://example.com")        # browse → readable text + links
web.read_table("https://…/iris.csv")          # → polars DataFrame (any format)
web.read_json("https://api.example.com/x")     # → decoded JSON
web.read_image("https://…/logo.png")           # → bytes + dims + content-type
```

### Interacting with a page — fill forms, click, type

Reading a page is a plain HTTP fetch; *operating* one — typing into fields,
ticking boxes, clicking buttons, submitting a form and reading what the page
becomes — drives a real **headless browser** (`web.Browser`, Playwright /
Chromium, imported lazily). Playwright (and its Chromium binary) **auto-install
on first use** — `web.ensure_browser()` pulls them on demand — unless you
disable that (see *Auto-installing optional deps* below):

```python
from yggdrasil.loki import web

# Fill a form and submit it, then read the resulting page.
web.fill_form("https://site/login",
              {"#user": "me", "#pass": "secret"},
              submit="button[type=submit]", wait_for="#dashboard")

# Or drive a page through an explicit sequence of interactions.
web.interact("https://shop/search", [
    {"type": ["#q", "wireless headphones"]},
    {"press": ["#q", "Enter"]},
    {"wait_for": ".results"},
    {"check": "#in-stock"},
    {"click": ".results a:first-child"},
])

# Low-level: a chainable browser session.
with web.Browser() as b:
    b.goto("https://site").fill("#email", "a@b.com").click("#go")
    print(b.url, b.title(), b.text())
```

The **`web` skill** exposes these too — `loki.run("web", url=…, action="form",
fields={…}, submit="#go")` and `action="interact", steps=[…]`. When the browser
isn't installed the skill returns an install hint rather than failing.

The **`web` behavior** also wraps the read paths (`loki.run("web", url=…,
question=…)`); the autonomous loop gets `web_fetch` / `web_table` / `web_image`
tools with `act(..., allow_web=True)` (or `ygg loki do --allow-web`), and local
tabular files parse through the same io layer via the always-on `read_table`
tool. In the interactive session a URL routes itself:

```text
⟢ auto ›  fetch https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
  ▹ web · a URL / web-fetch request — uses the HTTP session + io handlers
  🌐 https://…/iris.csv
  ▦ (150, 5) · sepal_length, sepal_width, petal_length, petal_width, species
  ┌──────────────┬─────────────┬─ …
```

## Data path — detect, fetch tabular, cache, propose

Loki keeps a **global read on the request**: `loki.classify_data(text)` flags
whether it's *data*- or *time-series*-shaped (rates, prices, history, "over
the last N weeks", a `.csv`/`.parquet`, …). When such a request carries a
source URL, the router sends it to the **data path** (`category: "data"`,
action `tabular`) instead of a plain page fetch — the `tabular` behavior:

1. **fetches it into a polars frame** through `web.read_table` — i.e. the io
   tabular handlers (`HTTPResponse.to_polars()` auto-detects CSV / JSON /
   Parquet / Arrow / XLSX); the io layer owns JSON→frame, not a bespoke
   normalizer;
2. **caches an optimized Parquet copy** in the session `cache/` via the io
   abstraction (`IO.from_(path).write_polars_frame(df)`);
3. **proposes next steps** — reuse the cache, store elsewhere
   (Parquet / Arrow / CSV / Delta), or load it into Databricks.

The frame preview it returns is **`Tabular.display()`** (see *Showing data*
below) — an aligned sample of the rows, no hand-rolled serialization.

```text
⟢ get EUR/USD exchange rates over the last 2 weeks from https://api.frankfurter.dev/…
  ▹ data · data/timeseries source → fetch as a cached tabular frame
  ▦ 11 × 3 · date, symbol, value
  …polars preview…
  ✎ cached …/cache/frankfurter-b03cb0.parquet  (Parquet, via io)
  next steps — reuse · store · load
    › reuse the cache:  loki.run('tabular', cache='…/frankfurter-b03cb0.parquet')
    › store as Parquet/Arrow/CSV/Delta:  loki.run('tabular', cache='…', store='out.parquet')
    › read it back anywhere:  IO.from_('…').to_polars()
```

```python
res = loki.run("tabular", url="https://…/rates.json")   # fetch → frame → cache
loki.run("tabular", cache=res["cached_to"], store="prices.parquet")   # reuse + store
```

## Showing data — `Tabular.display()`

Every row set in yggdrasil — a Databricks statement result, an `IO` leaf, a
Genie answer — **is a `Tabular`**, so there's nothing to serialize by hand.
`Tabular.display(n)` renders an aligned first-`n`-rows preview (header + values,
padded, long cells truncated), reading only enough Arrow batches to fill `n`
rows before stopping — cheap even on a large source:

```python
print(dbc.sql.execute("SELECT * FROM samples.nyctaxi.trips").display())
print(IO.from_("data.parquet").display(5))
```

Headers carry a **short data type** (`col:i64` / `:str` / `:ts` / `:list` …),
columns are separated by `│`, and nested values (lists / structs) are compacted
so the output never balloons:

```text
tpep_pickup_datetime:ts   │ trip_distance:f64 │ fare_amount:f64 │ pickup_zip:i32
──────────────────────────┼───────────────────┼─────────────────┼───────────────
2016-02-14 16:52:13+00:00 │ 4.94              │ 19.0            │ 10282
2016-02-04 18:44:19+00:00 │ 0.28              │ 3.5             │ 10110
… (first 10 rows)
```

For other representations, ask the same object: `to_pylist()` (records, e.g. for
`--json`), `to_polars()`/`to_arrow()` (frame / Arrow), and `from_`/`select`/
`filter`/`cast` to wrap and transform first. The Loki skills hand back the raw
`Tabular` (not a pre-baked string), and the CLI shows it with `display()`.

## Auto-installing optional deps (`YGG_LOKI_AUTO_INSTALL`)

Loki reaches for heavy optional packages only when a feature needs one — an
engine SDK (`anthropic` / `openai`), a local-model runtime (`transformers` +
`torch`), a headless browser (`playwright`), the MCP server (`mcp`). Rather than
fail, it **installs the missing package into the running interpreter on first
use** so it persists in the env Loki runs in. `yggdrasil.loki.runtime.load` is a
default-on wrapper over the project's one import-or-install guard
(`yggdrasil.lazy_imports._lazy_import` → `PyEnv.runtime_import_module`, anchored
on `sys.executable`); the only thing Loki changes is the default (`install=True`
instead of the project-wide `False`). Probes (`engine.available()`,
`web.browser_available()`) never install — only the code path that actually
needs the package does. Set `YGG_LOKI_AUTO_INSTALL=0` to turn this off, and a
missing package raises the normal `ImportError` instead.

All HTTP, meanwhile, is centralized on `HTTPSession` — `web.*` and the Ollama
engine drive their requests through it (pooling, retry budget, response
parsing), with a zero-retry probe session for liveness checks so a not-running
local server fails fast instead of stalling on the retry budget.

## Sessions & memory

Each interactive run gets an **isolated session workspace** under
`~/.loki/session/<id>/` (`workspace/` + `memory/` + `cache/`). The agent is
rooted at `workspace/`, so files it writes land there — out of your cwd and
safe to wipe. On start, Loki **auto-purges** old sessions (keep the 20 newest,
drop anything older than 14 days) so the tree never grows without bound.
Session trees count as scratch, so the destructive-confirm gate stays quiet
inside them.

A session carries **self-compressing working memory** (`yggdrasil.loki.LokiMemory`):
recent turns stay verbatim, and when context grows past a threshold the older
turns are folded into a compact **synthesis** (summarized by the fast engine,
for clean LLM reuse) and the raw turns dropped. This keeps a long, continuous
development session token-efficient and precise. `/memory` shows the
synthesis; `/sessions` lists the workspaces.

## MCP — Loki as a server, and Databricks MCP

`ygg loki mcp` runs Loki as a **Model Context Protocol** server over stdio, so
any MCP client (Claude Desktop, an editor) can drive the whole agent — tools
`reason`, `skills`, `run` (dispatch any skill: `databricks-sql`, `aws-s3`,
`genie`, …), `web`, `guide` (the optimized yggdrasil way), `tabular` (read any
source → cached frame), `engines`, `usage` (token KPIs), `setup` (a free local
model), and `capabilities`. Requires the optional `mcp` package (auto-installed
on first use).

```jsonc
// an MCP client config entry
{ "loki": { "command": "ygg", "args": ["loki", "mcp"] } }
```

The other direction: the **`databricks-mcp`** skill connects *out* to a
workspace's managed MCP servers — Unity Catalog functions, Genie, or vector
search — authenticating with the agent's Databricks credentials:

```bash
ygg loki run databricks-mcp --kwarg kind='"functions"' \
  --kwarg catalog='"main"' --kwarg schema='"sales"'       # list UC-function tools
ygg loki run databricks-mcp --kwarg kind='"genie"' --kwarg space='"01ef…"' \
  --kwarg tool='"ask"' --kwarg args='{"question":"top customers"}'
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
ygg loki mcp              # run Loki as an MCP server (stdio) for MCP clients
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
