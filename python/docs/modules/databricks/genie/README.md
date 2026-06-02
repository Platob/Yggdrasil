# yggdrasil.databricks.genie

Conversational analytics over Databricks **Genie** — ask questions in
plain English, get back a natural-language answer, the SQL Genie
generated, and the result materialised as Arrow / Polars / pandas. Plus a
self-driving **agent** that turns a single goal into a multi-turn
investigation, and two CLIs (`ygg databricks genie …` and `ygg-genie`).

Reached through the single client entrypoint:

```python
from yggdrasil.databricks import DatabricksClient

genie = DatabricksClient().genie
```

## One-liner

```python
from dataclasses import replace
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
client.genie.defaults = replace(client.genie.defaults, space_id="01ef…")

answer = client.genie.ask("top 5 customers by revenue this year")
print(answer.text)          # natural-language summary
print(answer.sql)           # the SQL Genie generated
df = answer.to_polars()     # the result as a polars DataFrame
```

## Concepts

| Resource | What it is |
|---|---|
| `Genie` | The service (`client.genie`). Resolves spaces, lists them, one-shot `ask`, builds agents. Carries `GenieDefaults`. |
| `GenieSpace` | A curated Genie room scoped to a set of tables + instructions. Start conversations, ask, list conversations. |
| `GenieConversation` | A live thread. `ask()` posts a follow-up turn and waits. |
| `GenieAnswer` | One message: `.text`, `.sql`, `.questions`, lifecycle state, and the query result (`.to_arrow()` / `.to_polars()` / `.to_pandas()` / `.rows()`). |
| `GenieAgent` / `AgentRun` | The autonomous driver and its transcript. |

## Spaces

A space is the unit you talk to — every question runs against the tables
and instructions curated in that space.

```python
genie = DatabricksClient().genie

# Discover spaces visible to you
for space in genie.list_spaces():
    print(space.space_id, space.title)

# Find one by title (case-insensitive)
space = genie.find_space(title="Renewable Energy Site Management")

# Or address it by id
space = genie.space("01f133f54ffd1442bc1ee8ecd6cba1c7")
print(space.title)          # cached infos — one get_space round-trip
print(space.warehouse_id)   # the warehouse Genie runs queries on
print(space.explore_url)    # deep link into the Databricks UI
```

## Ask a question

`ask()` is the one-shot path — it starts a fresh conversation and returns
just the answer.

```python
answer = space.ask("How many renewable sites are there?")

answer.text          # "There are 967 renewable sites…"
answer.status        # "COMPLETED"
answer.is_complete   # True
answer.failed        # False
answer.has_query     # True — Genie ran a SQL query
answer.sql           # "SELECT COUNT(DISTINCT id) …"
answer.description    # Genie's one-line description of the query
```

When Genie can't answer concretely it offers **suggested follow-up
questions** instead:

```python
answer = space.ask("show me sales")
if not answer.has_query:
    print(answer.questions)   # ("by region?", "by month?", …)
```

## Materialise the result

A query-backed answer carries an inline result. It projects to the
project's usual surfaces, with typed casts applied from the result
manifest (unknown / complex types stay as strings to preserve the bytes):

```python
answer = space.ask("top 3 sites by installed capacity")

answer.to_arrow()    # pyarrow.Table  (or None for a text-only answer)
answer.to_polars()   # polars.DataFrame
answer.to_pandas()   # pandas.DataFrame
answer.rows()        # list[dict] — e.g. [{"name": "...", "capacity": 400000.0}, …]
answer.row_count     # rows reported by the result manifest

# The raw SDK statement response is available too, if you need it
answer.statement_response
```

## Multi-turn conversations

Keep the thread alive to ask follow-ups in context:

```python
conv, first = space.start_conversation("revenue by month")
print(first.to_polars())

nxt = conv.ask("now just for EMEA")          # follow-up in the same thread
print(nxt.sql)

for msg in conv.messages():                   # replay the thread
    print(msg.text)
```

## Feedback

```python
answer.thumbs_up(comment="spot on")
answer.thumbs_down(comment="wrong table")
```

## The autonomous agent

`GenieAgent` turns a single goal into a multi-turn investigation. It opens
a conversation and, whenever Genie answers with clarifying *suggested
questions* instead of a concrete data answer, it **picks one and keeps
going on its own** — until it lands a query-backed answer, hits a failure,
runs out of suggestions, or exhausts its turn budget.

```python
run = client.genie.agent(space_id="01ef…").run("why did Q3 revenue dip?")

print(run.summary())     # full transcript of every turn
print(run.text)          # the agent's final word
print(run.sql)           # SQL behind the final data answer
df = run.to_polars()     # the final result as a DataFrame
run.rows()               # … or as rows

# Inspect the path it took
for turn in run.turns:
    tag = "agent" if turn.autonomous else "you"
    print(tag, turn.question)
```

Tunables:

```python
agent = client.genie.agent(
    space_id="01ef…",
    max_turns=6,              # how far it will drive (default 4)
    follow_suggestions=True,  # set False to stop after the first answer
)
run = agent.run("break down churn by plan")
```

`AgentRun` resolves the final answer for you:

- `run.answer` — the last answer in the run.
- `run.data_answer` — the most recent answer that actually ran a query.
- `run.text` / `run.sql` / `run.to_arrow()` / `run.to_polars()` /
  `run.to_pandas()` / `run.rows()` — convenience over `data_answer`.

## Defaults

Set the defaults once on the service and every call inherits them:

```python
from dataclasses import replace

client.genie.defaults = replace(
    client.genie.defaults,
    space_id="01ef…",          # default space for ask() / space() / agent()
    warehouse_id=None,          # override the warehouse used for results
    max_result_rows=5000,       # cap rows pulled into Arrow
)
```

`GenieDefaults` also carries the polling budget (`wait`,
default 20 min / 2 s) used while Genie embeds the question, plans a query,
runs it, and summarises. Override per call: `space.ask("…", wait=60)`.

## CLI — `ygg databricks genie`

The dispatcher sub-command:

```bash
# List spaces
ygg databricks genie spaces

# One-shot ask (space id or $YGG_GENIE_SPACE)
ygg databricks genie ask "How many renewable sites are there?" --space 01f133…

# Let the agent drive an investigation
ygg databricks genie agent "top 3 sites by capacity" --space 01f133… --max-turns 3

# Interactive conversation
ygg databricks genie repl --space 01f133…
```

Example `ask` output:

```text
There are **967** renewable sites according to the data provided.

SQL:
SELECT COUNT(DISTINCT `id`) AS num_renewable_sites
FROM `trading_tgp_dev`.`src_site_register`.`raw_ren_sites`
WHERE `id` IS NOT NULL;

num_renewable_sites
967
```

## CLI — `ygg-genie` (standalone agent)

A dedicated console script focused on the autonomous agent. The space id
falls back to `$YGG_GENIE_SPACE`.

```bash
# Agent mode (default): drive a multi-turn investigation, print the transcript
ygg-genie --space 01ef… "why did Q3 revenue dip?"

# One-shot ask (no autonomous follow-ups)
ygg-genie --space 01ef… --ask "top 5 customers by revenue"

# Deeper investigation
ygg-genie --space 01ef… --max-turns 6 "break down churn by plan"

# Interactive REPL — /agent <goal>, /new, /quit
YGG_GENIE_SPACE=01ef… ygg-genie
```

Both CLIs accept the shared Databricks client flags (`--host` / `--token`
/ `--profile` / …) and fall back to the standard `DATABRICKS_*`
environment variables.

## API reference

- [`yggdrasil.databricks.genie.service`](../../../reference/yggdrasil/databricks/genie/service.md)
- [`yggdrasil.databricks.genie.resources`](../../../reference/yggdrasil/databricks/genie/resources.md)
- [`yggdrasil.databricks.genie.agent`](../../../reference/yggdrasil/databricks/genie/agent.md)
