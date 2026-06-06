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

The built-in **`genie`** behavior is the reference: it guards on the
`databricks` backend, then asks a Genie space a question (autonomously
picking the first reachable space when none is named). Replication,
inter-agent messaging, HTTP ingestion and serving land on this abstraction
next.

## CLI

```bash
ygg loki                  # status: identity + reachable backends + behaviors
ygg loki capabilities     # the detected backends and their signals
ygg loki behaviors        # the registered behavior catalog
ygg loki token --probe    # the Databricks credentials Loki provides
ygg loki run genie --kwarg question='"top customers"' --json
```

`run` JSON-decodes each `--kwarg` value, so quote strings (`'"..."'`) and
pass numbers/booleans/objects directly.
