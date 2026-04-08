# yggdrasil.databricks.account

Account-level service entry point.

## Recommended one-liner

```python
from yggdrasil.databricks import DatabricksClient

account_client = DatabricksClient(account_id="<account-id>", host="https://accounts.cloud.databricks.com")
```

## Usage notes

`Accounts` is currently a thin service wrapper. Most account-level workflows are accessed through other services (for example IAM) once the client is initialized with `account_id`.

Examples:

- Account-scoped IAM groups: `DatabricksClient(account_id="<account-id>").iam.groups.list()`
- Account-scoped IAM users: `DatabricksClient(account_id="<account-id>").iam.users.list()`

## Extended example: account-scoped IAM browsing

```python
from yggdrasil.databricks import DatabricksClient

account = DatabricksClient(
    host="https://accounts.cloud.databricks.com",
    account_id="<account-id>",
    token="<token>",
)

print(next(account.iam.users.list(limit=1), None))
print(next(account.iam.groups.list(limit=1), None))
```
