# yggdrasil.databricks.genie

Genie conversational analytics — ask natural-language questions against a Databricks Genie space and get back SQL-backed answers.

**Requires:** `ygg[databricks]` and a Genie space set up in the workspace.

---

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

answer = DatabricksClient().genie.ask("<space-id>", "Top 10 customers by revenue")
print(answer)
```

---

## 1) Connect

```python
from yggdrasil.databricks import DatabricksClient

# PAT token
genie = DatabricksClient(host="https://<workspace>", token="<token>").genie

# Environment-driven (DATABRICKS_HOST / DATABRICKS_TOKEN)
genie = DatabricksClient().genie
```

---

## 2) One-shot question

```python
from yggdrasil.databricks import DatabricksClient

genie = DatabricksClient().genie
space_id = "<space-id>"

# Ask and get the answer as text
answer = genie.ask(space_id, "Weekly cloud cost trend for the last 3 months")
print(answer)

# The underlying SQL query Genie generated (when available)
result = genie.ask(space_id, "Revenue by region this month", return_query=True)
print(result.query)   # the SQL
print(result.answer)  # the text answer
```

---

## 3) Multi-turn conversation

```python
from yggdrasil.databricks import DatabricksClient

genie = DatabricksClient().genie
space_id = "<space-id>"

# Start a conversation
conv = genie.start_conversation(space_id)
cid  = conv.conversation_id

# First question
msg1 = genie.create_message(space_id, cid, "Revenue by month for the last 12 months")
out1 = genie.get_message(space_id, cid, msg1.id)
print(out1.content)

# Follow-up in the same conversation
msg2 = genie.create_message(space_id, cid, "Now break that down by product category")
out2 = genie.get_message(space_id, cid, msg2.id)
print(out2.content)

# Execute the generated SQL and get an Arrow table
result_tbl = genie.execute_message_query(space_id, cid, msg2.id)
print(result_tbl.to_arrow_table())
```

---

## 4) Execute the SQL a message generated

```python
from yggdrasil.databricks import DatabricksClient
import polars as pl

genie = DatabricksClient().genie
space_id = "<space-id>"

conv = genie.start_conversation(space_id)
msg  = genie.create_message(space_id, conv.conversation_id, "Top 5 SKUs by margin")
stmt = genie.execute_message_query(space_id, conv.conversation_id, msg.id)

# Convert to any engine
arrow_tbl = stmt.to_arrow_table()
df_polars  = stmt.to_polars()
df_pandas  = stmt.to_pandas()

print(df_polars.head(5))
```

---

## 5) Space management

```python
from yggdrasil.databricks import DatabricksClient

genie    = DatabricksClient().genie
space_id = "<space-id>"

# List all Genie spaces in the workspace
for space in genie.list_spaces():
    print(space.space_id, space.title)

# Create a new space
new_space = genie.create_space(
    title="Sales Analytics",
    description="Q&A over the sales schema in main.sales",
)
print(new_space.space_id)

# Update an existing space
genie.update_space(space_id, title="Sales Analytics (v2)")

# Delete a space
genie.delete_space(space_id)
```

---

## 6) Evaluation runs

Use evaluation runs to benchmark Genie accuracy against a curated question-answer set.

```python
from yggdrasil.databricks import DatabricksClient

genie    = DatabricksClient().genie
space_id = "<space-id>"

# Create an eval run (questions + expected SQL)
questions = [
    {"question": "Total revenue last month", "expected_sql": "SELECT SUM(revenue) ..."},
    {"question": "Top 10 customers",         "expected_sql": "SELECT customer_id, ..."},
]
run = genie.create_eval_run(space_id=space_id, questions=questions)
print(run.eval_run_id, run.status)

# List past eval runs
for r in genie.list_eval_runs(space_id):
    print(r.eval_run_id, r.status, r.accuracy)

# Get detailed results
details = genie.get_eval_run(space_id, run.eval_run_id)
print(details.accuracy, details.results)
```

---

## 7) Full pipeline: Genie → Arrow → Databricks table

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.data.enums import Mode

client   = DatabricksClient()
genie    = client.genie
space_id = "<space-id>"

# Ask a question and materialize the result into a Databricks table
conv = genie.start_conversation(space_id)
msg  = genie.create_message(
    space_id, conv.conversation_id,
    "Revenue by customer for the last 30 days",
)
stmt   = genie.execute_message_query(space_id, conv.conversation_id, msg.id)
tbl    = stmt.to_arrow_table()

# Write result into a curated table
target = client.catalogs["main"]["analytics"]["genie_revenue_by_customer"]
target.write_arrow_table(tbl, mode=Mode.OVERWRITE)
print(f"Wrote {len(tbl)} rows to {target.full_name()}")
```

---

## 8) CLI usage

```bash
# Ask a one-shot question from the command line
ygg-genie \
  --host https://<workspace> \
  --token <token> \
  --space-id <space-id> \
  "Top 10 customers by revenue last month"

# Environment-driven
DATABRICKS_HOST=https://<workspace> DATABRICKS_TOKEN=<token> \
  ygg-genie --space-id <space-id> "Weekly revenue trend"
```

See [cli module docs](../../cli/README.md) for full flag reference.
