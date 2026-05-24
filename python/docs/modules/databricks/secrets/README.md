# yggdrasil.databricks.secrets

Databricks Secrets API — scope and secret management with dict-style access, auto-scope creation, and rotation helpers.

---

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

DatabricksClient().secrets.create_secret("demo-scope/demo-key", "demo-value")
```

---

## 1) Access the service

```python
from yggdrasil.databricks import DatabricksClient

secrets = DatabricksClient(
    host="https://<workspace>",
    token="<token>",
).secrets
```

---

## 2) Create scope and secret (auto-creates scope if missing)

```python
# Auto-creates the scope when it doesn't exist (requires MANAGE privilege)
secrets.create_scope("my-service")
secrets.create_secret(key="api-token", value="s3cr3t-v1", scope="my-service")

# Slash-separated shorthand — same as above
secrets.create_secret("my-service/api-token", "s3cr3t-v1")
```

---

## 3) Dict-style access

```python
# Put (create or rotate)
secrets["my-service/api-token"] = "s3cr3t-v2"

# Delete
del secrets["my-service/api-token"]

# Key existence check (does not reveal the secret value)
print("my-service/api-token" in secrets)
```

---

## 4) Scope and secret objects

```python
scope = secrets.scope("my-service")
print(scope.name, scope.backend_type)

secret_obj = secrets.secret("my-service/api-token")
print(secret_obj.key, secret_obj.scope)

# List all secrets in a scope
for s in secrets.list_secrets(scope="my-service"):
    print(s.key, s.last_updated_timestamp)
```

---

## 5) Delete scope and its secrets

```python
# Delete one secret
secrets.delete_secret(key="api-token", scope="my-service")

# Delete the entire scope (removes all secrets in it)
secrets.delete_scope("my-service")
```

---

## 6) Rotate a secret

```python
def rotate_secret(scope: str, key: str, new_value: str) -> None:
    from yggdrasil.databricks import DatabricksClient
    svc = DatabricksClient().secrets
    svc[f"{scope}/{key}"] = new_value
    print(f"Rotated {scope}/{key}")

rotate_secret("my-service", "api-token", "s3cr3t-v3")
```

---

## 7) Bootstrap a service's credential set

```python
from yggdrasil.databricks import DatabricksClient

def bootstrap_secrets(workspace_url: str, token: str, service: str, creds: dict) -> None:
    """Create scope + all secrets for a new service integration."""
    svc = DatabricksClient(host=workspace_url, token=token).secrets
    svc.create_scope(service)
    for key, value in creds.items():
        svc.create_secret(key=key, value=value, scope=service)
        print(f"  Created {service}/{key}")

bootstrap_secrets(
    workspace_url="https://<workspace>",
    token="<token>",
    service="vendor-api",
    creds={
        "api-key": "abc123",
        "api-secret": "xyz987",
        "endpoint": "https://vendor.example.com",
    },
)
```

---

## 8) Use secrets in a Databricks notebook

```python
from yggdrasil.databricks.jobs import get_dbutils

dbutils = get_dbutils()

if dbutils:
    api_key = dbutils.secrets.get(scope="vendor-api", key="api-key")
    api_secret = dbutils.secrets.get(scope="vendor-api", key="api-secret")
    print("Loaded credentials, key len:", len(api_key))
```

---

## 9) End-to-end: create, use, rotate, delete

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient(host="https://<workspace>", token="<token>")
s = c.secrets

# Bootstrap
s.create_scope("docs-demo")
s["docs-demo/token"] = "initial-v1"

# Verify existence (key only — never returns the value via the API)
print("docs-demo/token" in s)   # True

# Rotate
s["docs-demo/token"] = "rotated-v2"

# Cleanup
s.delete_secret(key="token", scope="docs-demo")
s.delete_scope("docs-demo")
```
