# Skill: drive Databricks Genie from code

## When to use

The user asks to list / open / create a Genie space, ask an AI/BI **Genie**
space a question from a notebook or job, or pull a Genie answer's rows into
a DataFrame — all in Python, never a CLI.

## Spaces

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()

dbc.genie.spaces()                 # list[GenieSpace]
space = dbc.genie["01ef…space_id"]  # a handle (metadata fetched lazily)
space = dbc.genie.find("Sales analytics")   # by exact title → GenieSpace | None

space.title; space.description; space.warehouse_id
```

Lifecycle (code-oriented manipulation):

```python
space = dbc.genie.create_space(
    warehouse_id="wh123",
    serialized_space="{…}",        # the space definition JSON
    title="Sales analytics",
)
space.update(title="Sales (FY26)")
space.trash()                      # delete (alias: space.delete())
```

## Ask a question

```python
answer = space.ask("top 10 customers by revenue last quarter")
# or, one-shot from the service:
answer = dbc.genie.ask("01ef…space_id", "revenue by region last month")

answer.text          # Genie's natural-language reply
answer.query         # the SQL Genie generated (or None)
answer.statement_id  # the executed statement id (or None)
```

Follow up in the same conversation:

```python
follow = space.follow_up(answer.conversation_id, "now split by product")
```

## Get the rows

Genie runs its SQL on the **space's own warehouse**; ygg re-attaches to the
finished statement and reads it the usual way — no re-run, same
`StatementResult` surface as `dbc.sql`:

```python
answer.to_polars()        # polars.DataFrame
answer.to_arrow_table()   # pyarrow.Table
answer.to_pandas()        # pandas.DataFrame
answer.to_pylist()        # list[dict] — small results only

# or the StatementResult itself, for streaming / config:
result = answer.result()
```

If the turn produced no query (`answer.query is None`), `result()` raises —
check `answer.text` for the narrative reply instead.

## Don'ts

- Don't shell out to a CLI — `dbc.genie` is pure Python and works on
  serverless.
- Don't re-run the SQL by hand to get rows — `answer.to_polars()` reads the
  statement Genie already executed.
- Don't pass a warehouse for the result read — the space's warehouse is
  used automatically (the engine default if the space has none).
- Don't `exists()`-check a space before `ask` — open the handle and ask.
