# yggdrasil.databricks.iam

Identity and access management helpers for users and groups (workspace or account scope).

## Recommended one-liner

```python
from yggdrasil.databricks import DatabricksClient

print(DatabricksClient().iam.users.current_user)
```

## Features and examples

```python
from yggdrasil.databricks import DatabricksClient

iam = DatabricksClient(host="https://<workspace>", token="<token>").iam
```

### Users

- Create user: `iam.users.create("analyst@company.com")`
- List users: `list(iam.users.list(limit=20))`
- Current user: `iam.users.current_user`
- Reset local user cache: `iam.users.reset_local_cache()`

### Groups

- Create group with default current-user membership: `grp = iam.groups.create("data-engineering")`
- List groups: `list(iam.groups.list(name="data-engineering", limit=5))`
- Delete group by object or id: `iam.groups.delete(grp)` or `iam.groups.delete_group("<group-id>")`

For account-level operations, initialize `DatabricksClient(account_id="<account-id>", ...)` so IAM calls target the account APIs.

## Extended example: create group, add members via create flow, cleanup

```python
from yggdrasil.databricks import DatabricksClient

iam = DatabricksClient(host="https://<workspace>", token="<token>").iam

user = iam.users.create("docs_user@company.com")
group = iam.groups.create("docs-group", members=[user])

print(group.id, group.name)
print([g.name for g in iam.groups.list(name="docs-group", limit=10)])

iam.groups.delete(group)
```
