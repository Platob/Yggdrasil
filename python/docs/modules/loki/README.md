# Loki — the global agent

`yggdrasil.loki` is one agent that adapts to wherever it runs: it detects the
backends it can reach (offline, never raises), acts as a **token / credential
provider** (chiefly Databricks), and dispatches pluggable **skills**. Drive it
from code (`Loki`) or the terminal (`ygg loki`).

For the full walkthrough — capability detection, the token provider, skills,
engines, tools, and the CLI — see the [Loki guide](../../guides/loki.md). Full
signatures live in the auto-generated [API reference](../../reference/yggdrasil/loki/).

```python
from yggdrasil.loki import Loki

loki = Loki.current()
loki.card()              # identity + reachable backends + skills
loki.databricks          # the live DatabricksClient, or None
loki.run("genie", question="revenue by region last month")
```

## Surface

| Symbol | Role |
|---|---|
| [`Loki`][yggdrasil.loki.Loki] | The agent — `current()`, `card()`, `run()`, `databricks`, `backends()` |
| [`Backend`][yggdrasil.loki.Backend] / [`detect`][yggdrasil.loki.detect] | Offline capability detection (`databricks`, `node`, `local`) |
| [`LokiSkill`][yggdrasil.loki.LokiSkill] / [`register`][yggdrasil.loki.register] | Discoverable, backend-aware capabilities + the registration decorator |
| [`TokenEngine`][yggdrasil.loki.TokenEngine] / [`Completion`][yggdrasil.loki.Completion] | The reasoning-engine contract (Claude, OpenAI, Ollama, transformers, local) |
| [`Tool`][yggdrasil.loki.Tool] / [`Toolbox`][yggdrasil.loki.Toolbox] / [`filesystem_toolbox`][yggdrasil.loki.filesystem_toolbox] | Callable tools exposed to an engine |
| [`LokiSession`][yggdrasil.loki.LokiSession] / [`LokiMemory`][yggdrasil.loki.LokiMemory] | Conversation session + persistent memory |
| [`TokenMeter`][yggdrasil.loki.TokenMeter] / [`price_for`][yggdrasil.loki.price_for] | Token accounting + model pricing (`METER`) |

## Register a skill

```python
from yggdrasil.loki import LokiSkill, register

@register
class Hello(LokiSkill):
    name = "hello"
    description = "say hi"
    def run(self, agent, *, who="world", **_):
        return f"hello {who} from {agent.user}@{agent.host}"
```

## CLI

`ygg loki` is a thin shell over `Loki` — `status` / `capabilities` / `skills` /
`engines` / `tools` / `reason` / `do` / `token` / `run`.
