# Skill: ask Genie questions from Python

## When to use

The user asks to query Genie, run a natural-language question against
a Genie space, or materialise Genie answers as DataFrames.

## One-shot question

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
answer = dbc.genie.ask("01ef…", "How many orders shipped last week?")

answer.text                          # natural-language reply
answer.sql                           # generated SQL
answer.statement_result.to_polars()  # materialise as DataFrame
```

## Multi-turn conversation

```python
space = dbc.genie.space("01ef…")
conv = space.start_conversation("List top customers this month")
follow_up = conv.ask("Break that down by product category")
```

## Discover spaces

```python
space = dbc.genie.find_space(name="Sales analytics")
for s in dbc.genie.list_spaces():
    print(s.space_id, s.name)
```

## Don'ts

- Don't re-execute `answer.sql` via `dbc.sql.execute(...)` — the
  result is already in `answer.statement_result`.
- Don't poll Genie endpoints directly — `dbc.genie.ask` handles the
  state machine.
