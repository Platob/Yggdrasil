# yggdrasil.databricks.secrets

Databricks Secrets API helpers with scope/secret objects and dict-style shortcuts.

## Recommended one-liner

```python
from yggdrasil.databricks import DatabricksClient

DatabricksClient().secrets.create_secret("demo-scope/demo-key", "demo-value")
```

## Features and examples

```python
from yggdrasil.databricks import DatabricksClient

secrets = DatabricksClient(host="https://<workspace>", token="<token>").secrets
```

- Create scope: `secrets.create_scope("demo-scope")`
- Create secret in scope: `secrets.create_secret("api-key", "<value>", scope="demo-scope")`
- Parse scope/secret handles: `scope = secrets.scope("demo-scope")`; `secret = secrets.secret("demo-scope/api-key")`
- Delete secret: `secrets.delete_secret("api-key", scope="demo-scope")`
- Dict-style put/get/delete: `secrets["demo-scope/api-key"] = "new-value"`; `del secrets["demo-scope/api-key"]`

`create_secret` auto-creates the scope when missing (when permitted), which keeps onboarding scripts short.

## Extended example: bootstrap scope and rotate key

```python
from yggdrasil.databricks import DatabricksClient

secrets = DatabricksClient(host="https://<workspace>", token="<token>").secrets
scope = "demo-service"
key = "api-token"

secrets.create_scope(scope)
secrets.create_secret(key=key, value="v1-token", scope=scope)

# rotate
secrets[f"{scope}/{key}"] = "v2-token"
print(secrets.secret(f"{scope}/{key}").key)

secrets.delete_secret(key=key, scope=scope)
```
