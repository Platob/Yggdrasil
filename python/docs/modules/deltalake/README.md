# deltalake interop

The `deltalake` Python package (Rust-backed Delta Lake reader/writer) is an **optional dependency**:

```bash
pip install 'ygg[delta]'   # installs deltalake>=1.0
```

## Yggdrasil's built-in Delta support

Yggdrasil has its own pure-Python Delta implementation at [`yggdrasil.delta`](../delta/README.md) — no Rust, no JVM, works on any `Path` (local, S3, DBFS). Use it for:

- Reading/writing Delta tables from Python without Spark
- Spark-distributed reads via `mapInArrow` (no driver collect)
- Full protocol support: V1/V2 checkpoints, deletion vectors, partition pruning
- Interop with Databricks SQL engine

## When to use `deltalake` instead

The `deltalake` package is useful when you need:

- Features not yet in yggdrasil (e.g., OPTIMIZE/ZORDER, VACUUM)
- Rust-native performance for very large single-file reads
- Compatibility testing against the reference Delta implementation

## Interop between yggdrasil and deltalake

Tables written by either engine are readable by the other:

```python
from yggdrasil.delta import DeltaFolder
import deltalake
import pyarrow as pa

# Write with yggdrasil, read with deltalake
folder = DeltaFolder(path="/tmp/table")
folder.write_arrow_table(pa.table({"id": [1, 2, 3]}))
dt = deltalake.DeltaTable("/tmp/table")
print(dt.to_pyarrow_table())

# Write with deltalake, read with yggdrasil
deltalake.write_deltalake("/tmp/table2", pa.table({"id": [4, 5]}))
folder2 = DeltaFolder(path="/tmp/table2")
print(folder2.read_arrow_table())
```

See the [delta module docs](../delta/README.md) for the complete API reference.
