# yggdrasil.databricks.genie

Genie conversational analytics — natural-language queries against your Databricks data, backed by your Unity Catalog tables.

---

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

answer = DatabricksClient().genie.ask("<space-id>", "Top 10 customers by revenue this month")
print(answer)
```

---

## 1) Access the service

```python
from yggdrasil.databricks import DatabricksClient

genie = DatabricksClient(
    host="https://<workspace>",
    token="<token>",
).genie
```

---

## 2) Single-shot question

`ask()` starts a conversation, sends the question, waits for the answer, and returns it as a string:

```python
answer = genie.ask("<space-id>", "What is the total revenue by region for Q1 2026?")
print(answer)
```

---

## 3) Conversation lifecycle

```python
space_id = "<space-id>"

# Start a conversation
conv = genie.start_conversation(space_id)
print("Conversation ID:", conv.conversation_id)

# Send messages
msg1 = genie.create_message(space_id, conv.conversation_id, "Revenue by month for last 12 months")
msg2 = genie.create_message(space_id, conv.conversation_id, "Break it down by region")
msg3 = genie.create_message(space_id, conv.conversation_id, "Only show EMEA")

# Retrieve a message result
result = genie.get_message(space_id, conv.conversation_id, msg3.id)
print(result)
```

---

## 4) Execute the underlying SQL query

Genie generates SQL behind the scenes. Retrieve and run it directly for Arrow/pandas output:

```python
space_id = "<space-id>"
conv = genie.start_conversation(space_id)
msg = genie.create_message(space_id, conv.conversation_id, "Daily active users last 30 days")

# Execute the query that Genie generated
stmt = genie.execute_message_query(space_id, conv.conversation_id, msg.id)
print(stmt.to_pandas())   # DataFrame of the result
print(stmt.to_arrow_table())
```

---

## 5) List and manage spaces

```python
# List all Genie spaces you have access to
for space in genie.list_spaces():
    print(space.space_id, space.title)

# Create a space (requires admin privilege)
new_space = genie.create_space(
    title="Sales Analytics",
    description="Natural-language access to sales data",
)
print(new_space.space_id)

# Update title/description
genie.update_space(new_space.space_id, title="Sales Analytics v2")

# Delete
genie.delete_space(new_space.space_id)
```

---

## 6) Evaluation runs (batch accuracy testing)

```python
# Create an eval run to test a set of benchmark questions
run = genie.create_eval_run(
    space_id="<space-id>",
    questions=["Total revenue Q1 2026", "Top 5 products by units sold"],
)
print(run.run_id, run.status)

# List eval runs for a space
for r in genie.list_eval_runs("<space-id>"):
    print(r.run_id, r.status, r.created_at)

# Get run details / results
details = genie.get_eval_run("<space-id>", run.run_id)
for q_result in details.question_results:
    print(q_result.question, q_result.accuracy)
```

---

## 7) Complex pattern: Genie → Arrow → Databricks table

Pull a Genie answer as structured data and persist it:

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient()
genie = c.genie
space_id = "<space-id>"

# Ask and get structured result
conv = genie.start_conversation(space_id)
msg = genie.create_message(space_id, conv.conversation_id, "Weekly revenue last 8 weeks by region")
stmt = genie.execute_message_query(space_id, conv.conversation_id, msg.id)

# Persist to a Delta table
arrow_tbl = stmt.to_arrow_table()
c.sql.execute(
    f"CREATE TABLE IF NOT EXISTS main.genie_exports.weekly_revenue USING DELTA"
)
c.sql.insert_into("main.genie_exports.weekly_revenue", arrow_tbl, mode="overwrite")
print(f"Saved {arrow_tbl.num_rows} rows")
```

---

## 8) Polling loop for async messages

```python
import time
from yggdrasil.databricks import DatabricksClient

genie = DatabricksClient().genie
space_id = "<space-id>"

conv = genie.start_conversation(space_id)
msg = genie.create_message(space_id, conv.conversation_id, "Complex multi-join aggregation")

for attempt in range(30):
    result = genie.get_message(space_id, conv.conversation_id, msg.id)
    if result.status in ("COMPLETED", "FAILED"):
        break
    time.sleep(2)

print(result.status, result.content)
```
