# yggdrasil.databricks

Databricks integration layer: client configuration, SQL execution, compute helpers, paths/filesystems, IAM, secrets, and job config parsing.

## Main submodules

- [workspaces](workspaces/README.md)
- [sql](sql/README.md)
- [compute](compute/README.md)
- [compute.remote](compute/remote/README.md)
- [jobs](jobs/README.md)

## Quick start

```python
from yggdrasil.databricks.workspaces import Workspace
from yggdrasil.databricks.sql import SQLEngine

client = Workspace(host="https://<workspace>", token="<token>")
engine = SQLEngine(client=client)

stmt = engine.execute("SELECT 1 AS ok")
print(stmt.to_arrow_table())
```
