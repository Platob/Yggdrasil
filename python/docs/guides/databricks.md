# Databricks Guide

Yggdrasil includes a broad Databricks surface area:

- workspace management,
- SQL engines and statement results,
- jobs, compute, IAM, secrets,
- account-level services.

## SQL example

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient(host="https://<workspace>", token="<token>")
stmt = client.sql.execute("SELECT 1 AS value")
print(stmt.to_arrow_table())
```
