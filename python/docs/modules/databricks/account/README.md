# yggdrasil.databricks.account

Account-level Databricks API — multi-workspace governance, account-scoped IAM, and metastore management.

---

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

account = DatabricksClient(
    host="https://accounts.cloud.databricks.com",
    account_id="<account-id>",
    token="<token>",
)
```

---

## 1) Account client vs workspace client

```python
from yggdrasil.databricks import DatabricksClient

# Workspace-scoped (most operations)
ws = DatabricksClient(host="https://<workspace>", token="<token>")

# Account-scoped (cross-workspace governance)
account = DatabricksClient(
    host="https://accounts.cloud.databricks.com",
    account_id="<account-id>",
    token="<token>",
)
```

The `account_id` is the UUID from **Account Settings → Account ID** in the Databricks console.

---

## 2) Account-scoped IAM

```python
# List all account-level users
for user in account.iam.users.list(limit=100):
    print(user.user_name, user.id)

# List account-level groups
for group in account.iam.groups.list(limit=50):
    print(group.display_name)

# Create an account-level group
grp = account.iam.groups.create("platform-admins")

# Add users to the group
user = account.iam.users.create("sre@company.com")
account.iam.groups.add_member(group=grp, member=user)
```

---

## 3) List metastores

```python
# List Unity Catalog metastores registered to the account
for metastore in account.metastores.list():
    print(metastore.name, metastore.metastore_id, metastore.region)
```

---

## 4) List workspaces

```python
# Enumerate all workspaces in the account
for ws_info in account.workspaces.list():
    print(ws_info.workspace_name, ws_info.workspace_id, ws_info.workspace_status)
```

---

## 5) Audit: cross-account group membership report

```python
from yggdrasil.databricks import DatabricksClient

account = DatabricksClient(
    host="https://accounts.cloud.databricks.com",
    account_id="<account-id>",
    token="<token>",
)

report = []
for group in account.iam.groups.list(limit=200):
    members = group.members or []
    report.append({
        "group": group.display_name,
        "member_count": len(members),
        "members": [m.display for m in members],
    })

for row in sorted(report, key=lambda r: -r["member_count"])[:10]:
    print(row["group"], "→", row["member_count"], "members")
```

---

## 6) Multi-workspace user provisioning

```python
from yggdrasil.databricks import DatabricksClient

def provision_user_across_workspaces(
    account_id: str,
    account_token: str,
    workspace_urls: list[str],
    workspace_token: str,
    email: str,
    group_name: str,
) -> None:
    account = DatabricksClient(
        host="https://accounts.cloud.databricks.com",
        account_id=account_id,
        token=account_token,
    )
    # Create at account level first
    user = account.iam.users.create(email)

    # Then add to each workspace
    for url in workspace_urls:
        ws_iam = DatabricksClient(host=url, token=workspace_token).iam
        try:
            ws_user = ws_iam.users.create(email)
            group = next(ws_iam.groups.list(name=group_name), None)
            if group:
                ws_iam.groups.add_member(group=group, member=ws_user)
                print(f"Added {email} to {group_name} on {url}")
        except Exception as e:
            print(f"Skipped {url}: {e}")

provision_user_across_workspaces(
    account_id="<account-id>",
    account_token="<account-token>",
    workspace_urls=["https://prod.azuredatabricks.net", "https://dev.azuredatabricks.net"],
    workspace_token="<workspace-token>",
    email="new.engineer@company.com",
    group_name="data-platform",
)
```
