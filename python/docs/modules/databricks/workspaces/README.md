# yggdrasil.databricks.workspaces

Workspace-oriented client setup plus path helpers.

## Recommended one-liner

```python
from yggdrasil.databricks import DatabricksClient

print(DatabricksClient().workspace)
```

## Client initialization patterns

```python
from yggdrasil.databricks import DatabricksClient
```

- Environment-based auth: `c = DatabricksClient()`
- PAT auth: `c = DatabricksClient(host="https://<workspace>", token="<token>")`
- OAuth client auth: `c = DatabricksClient(host="https://<workspace>", client_id="...", client_secret="...")`
- Reuse global singleton: `c = DatabricksClient.current()`
- Parse URL form: `c = DatabricksClient.parse("dbks://:<token>@<workspace-host>/?cluster_id=<id>")`

## Workspace and path helpers

- Workspace resource: `ws = c.workspace`
- Build DBFS path: `p = c.dbfs_path("dbfs:/tmp/example.json")`
- Temp managed path: `tmp = c.tmp_path(extension="json", max_lifetime=3600)`
- Cleanup stale temp files: `c.clean_tmp_folder()`
- Connect/close lifecycle: `c.connect(); c.close()`

## Related docs

- [fs](../fs/README.md) for `DatabricksPath`, `DBFSPath`, `VolumePath`, file I/O helpers.
- [sql](../sql/README.md) for SQL service usage via `DatabricksClient().sql`.

## Extended example: temporary file workflow

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient(host="https://<workspace>", token="<token>")
tmp = c.tmp_path(extension="json", max_lifetime=1800)

tmp.write_text('{"step": "created"}')
print(tmp.exists(), tmp.read_text())

archive = tmp.with_suffix(".bak.json")
tmp.rename(archive)
print(archive.exists())

archive.remove()
```
