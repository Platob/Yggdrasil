# yggdrasil.databricks.fs

Filesystem abstractions for DBFS, Workspace files, Volumes, and table-backed paths.

## Recommended one-liner

```python
from yggdrasil.databricks import DatabricksClient

DatabricksClient().dbfs_path("dbfs:/tmp/hello.txt").write_text("hello")
```

## Path features

```python
from yggdrasil.databricks import DatabricksClient

p = DatabricksClient().dbfs_path("dbfs:/tmp/example.parquet")
```

- Introspection: `p.exists()`, `p.is_file()`, `p.is_dir()`, `p.stat()`
- Read/write bytes/text: `p.write_bytes(b"x")`; `p.read_text()`
- Directory operations: `p.parent.mkdir(parents=True)`; `list(p.parent.ls())`
- Move/copy/delete: `p.rename("dbfs:/tmp/new.parquet")`; `p.remove()`
- Path transforms: `p.with_suffix(".json")`; `p.relative_to("dbfs:/tmp")`
- Volume parsing: `DatabricksClient().dbfs_path("/Volumes/main/default/raw/data.parquet")`

## SQL-aware volume/table helpers

- Resolve SQL triple: `catalog, schema, name = p.sql_volume_or_table_parts()`
- Access SQL engine from path: `p.sql_engine.execute("SELECT 1")`

## Extended example: copy between DBFS and Volume

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient(host="https://<workspace>", token="<token>")
src = c.dbfs_path("dbfs:/tmp/demo/source.txt")
dst = c.dbfs_path("/Volumes/main/default/tmp/source.txt")

src.parent.mkdir(parents=True, exist_ok=True)
src.write_text("hello from dbfs")

src.copy_to(dst)
print(dst.read_text())

src.remove()
dst.remove()
```
