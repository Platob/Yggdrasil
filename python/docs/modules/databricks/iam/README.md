# yggdrasil.databricks.iam

Identity and access management — users and groups for workspace-scoped and account-scoped operations.

---

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

print(DatabricksClient().iam.users.current_user)
```

---

## 1) Access the service

```python
from yggdrasil.databricks import DatabricksClient

# Workspace-scoped
iam = DatabricksClient(host="https://<workspace>", token="<token>").iam

# Account-scoped (all operations apply to the account level)
account_iam = DatabricksClient(
    host="https://accounts.cloud.databricks.com",
    account_id="<account-id>",
    token="<token>",
).iam
```

---

## 2) Current user

```python
me = iam.users.current_user
print(me.user_name, me.display_name, me.id)
```

---

## 3) User management

```python
# Create
user = iam.users.create("analyst@company.com")
print(user.id, user.user_name)

# List (paginated generator)
for u in iam.users.list(limit=50):
    print(u.user_name)

# Filter by email
for u in iam.users.list(filter="userName eq 'analyst@company.com'"):
    print(u.id, u.user_name)

# Refresh local user cache (cache miss fetches from API)
iam.users.reset_local_cache()
```

---

## 4) Group management

```python
# Create group
group = iam.groups.create("data-engineering")
print(group.id, group.display_name)

# Create with initial members
users = [iam.users.create(f"eng{i}@company.com") for i in range(3)]
group = iam.groups.create("ml-team", members=users)

# List groups
for g in iam.groups.list(name="data-engineering", limit=5):
    print(g.id, g.display_name)

# Delete
iam.groups.delete(group)
iam.groups.delete_group("<group-id>")   # by string id
```

---

## 5) Add members to an existing group

```python
iam_svc = iam.groups

# Add a single user to an existing group
group = next(iam_svc.list(name="data-engineering"), None)
if group:
    user = iam.users.create("new.engineer@company.com")
    iam_svc.add_member(group=group, member=user)
```

---

## 6) Account-level group sync

Use the account-scoped client to manage groups that span multiple workspaces:

```python
from yggdrasil.databricks import DatabricksClient

account = DatabricksClient(
    host="https://accounts.cloud.databricks.com",
    account_id="<account-id>",
    token="<account-token>",
)

# List account-level groups
for g in account.iam.groups.list(limit=20):
    print(g.display_name)

# List account users
for u in account.iam.users.list(limit=20):
    print(u.user_name)
```

---

## 7) Onboard a team (end-to-end)

```python
from yggdrasil.databricks import DatabricksClient

def onboard_team(workspace_url: str, token: str, team_name: str, emails: list[str]):
    iam = DatabricksClient(host=workspace_url, token=token).iam
    users = [iam.users.create(email) for email in emails]
    group = iam.groups.create(team_name, members=users)
    print(f"Created group '{group.display_name}' with {len(users)} members")
    return group

group = onboard_team(
    workspace_url="https://<workspace>",
    token="<token>",
    team_name="analytics-platform",
    emails=["alice@company.com", "bob@company.com", "carol@company.com"],
)
```

---

## 8) Audit: find all admin groups

```python
from yggdrasil.databricks import DatabricksClient

iam = DatabricksClient().iam

admin_groups = [
    g for g in iam.groups.list(limit=200)
    if "admin" in g.display_name.lower()
]

for g in admin_groups:
    print(g.id, g.display_name)
```
