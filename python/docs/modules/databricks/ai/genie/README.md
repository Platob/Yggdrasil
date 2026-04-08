# yggdrasil.databricks.ai.genie

Genie conversational analytics service wrapper.

## Recommended one-liner

```python
from yggdrasil.databricks import DatabricksClient

print(DatabricksClient().genie.ask("<space-id>", "Top 10 customers by revenue"))
```

## Features and examples

```python
from yggdrasil.databricks import DatabricksClient

genie = DatabricksClient(host="https://<workspace>", token="<token>").genie
```

- Ask a question directly: `answer = genie.ask("<space-id>", "Weekly cloud cost trend")`
- Conversation lifecycle: `conv = genie.start_conversation("<space-id>")`
- Send message: `msg = genie.create_message("<space-id>", conv.conversation_id, "Break down by team")`
- Fetch message/query outputs: `genie.get_message("<space-id>", conv.conversation_id, msg.id)`
- Execute attached SQL query: `genie.execute_message_query("<space-id>", conv.conversation_id, msg.id)`
- Space management: `genie.list_spaces()` / `genie.create_space(...)` / `genie.update_space(...)` / `genie.delete_space(...)`
- Eval runs: `run = genie.create_eval_run(space_id="<space-id>", ...)`; `genie.list_eval_runs("<space-id>")`

## Extended example: conversation loop

```python
from yggdrasil.databricks import DatabricksClient

genie = DatabricksClient(host="https://<workspace>", token="<token>").genie
space_id = "<space-id>"

conversation = genie.start_conversation(space_id)
first = genie.create_message(space_id, conversation.conversation_id, "Revenue by month for last 12 months")
second = genie.create_message(space_id, conversation.conversation_id, "Now split by region")

print(first.id, second.id)
print(genie.get_message(space_id, conversation.conversation_id, second.id))
```
