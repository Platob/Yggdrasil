# Skill: ask Genie questions and consume answers programmatically

## When to use

The user asks to "ask Genie", "chat with a Genie space", "run a
Genie conversation from Python", "fetch the SQL Genie generated",
"materialise the result of a Genie question as a DataFrame", or to
list / discover Genie spaces in the workspace.

## Primary surface

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
genie = dbc.genie                          # Genie service
```

The main entry points on `dbc.genie`:

| Call | Returns | For |
| --- | --- | --- |
| `dbc.genie.ask(space_id, "question")` | `GenieAnswer` | one-shot question |
| `dbc.genie.space(space_id)` | `GenieSpace` | resource singleton |
| `dbc.genie.find_space(name="…")` | `GenieSpace \| None` | discover by name |
| `dbc.genie.list_spaces()` | iterator of `GenieSpace` | enumerate |
| `dbc.genie.conversation(space_id=…, conversation_id=…)` | `GenieConversation` | resume a thread |

## One-shot question

```python
answer = dbc.genie.ask(
    space_id="01ef…",
    "How many orders did we ship last week, by region?",
)

answer.text                # Genie's natural-language reply
answer.sql                 # the generated SQL string
answer.statement_result    # StatementResult — full DataTable surface
answer.statement_result.to_polars()
```

`GenieAnswer` carries both sides: the natural-language reply *and*
the underlying `StatementResult`. Materialise via the standard
`to_arrow_table()` / `to_polars()` / `to_pandas()` / `to_spark()`
methods — same surface as `dbc.sql.execute(...)`.

## Multi-turn conversation

```python
space = dbc.genie.space("01ef…")
conv = space.start_conversation("List the top customers this month")
follow_up = conv.ask("Now break that down by product category")

for msg in conv.messages:
    msg.text, msg.statement_result
```

Use `GenieConversation` when follow-ups should reuse Genie's prior
context (filters, joins, named entities). Each turn produces a
`GenieAnswer` with its own `StatementResult`.

## Discovery

```python
space = dbc.genie.find_space(name="Sales analytics")
if space is None:
    raise LookupError("Sales analytics Genie space not found")

for s in dbc.genie.list_spaces():
    print(s.space_id, s.name)
```

`find_space` returns `None` on miss (forgiving on input); raise
yourself if absence is an error in your context.

## Pin a target schema on Genie results

`GenieAnswer.statement_result` is a `StatementResult`, so the same
`CastOptions(target_field=...)` trick works:

```python
from yggdrasil.data.cast.options import CastOptions

target = my_schema.to_arrow()
answer = dbc.genie.ask(
    space_id="01ef…",
    "Show me orders by region",
    options=CastOptions(target_field=target),
)
answer.statement_result.to_arrow_table()    # already conforming
```

## Don'ts

- Don't poll the Genie REST endpoints directly — `dbc.genie.ask` /
  `GenieConversation.ask` already handle the long-running message
  state machine with the right backoff.
- Don't re-execute `answer.sql` via `dbc.sql.execute(...)` to "get
  rows" — `answer.statement_result` already holds them.
- Don't pickle a `GenieConversation` across processes hoping Genie
  resumes server-side — the conversation id is what's picklable; on
  the other side call `dbc.genie.conversation(space_id=…,
  conversation_id=…)` to rehydrate.
- Don't `.to_pylist()` the answer rows to format a chart input —
  build a Polars / pandas frame and pass it to the plot library.
