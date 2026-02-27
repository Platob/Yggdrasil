# Yggdrasil (Python)

Schema-aware utilities for moving data between Python objects, Arrow, Polars, pandas, Spark, and Databricks. Define types once — cast everywhere.

## Install

```bash
pip install ygg                           # core (Arrow, requests, pyutils)
pip install "ygg[polars]"                # + Polars
pip install "ygg[pandas]"                # + pandas
pip install "ygg[spark]"                 # + PySpark
pip install "ygg[databricks]"            # + Databricks SDK
```

From source (dev):

```bash
cd python/
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quickstart

### Infer Arrow schema from a dataclass

```python
from dataclasses import dataclass
from yggdrasil.dataclasses import dataclass_to_arrow_field

@dataclass
class Order:
    id: int
    amount: float
    country: str | None = None

field = dataclass_to_arrow_field(Order)
print(field.type)           # struct<id: int64, amount: double, country: string>
schema = field.type.to_schema()
```

### Cast any table to an Arrow schema

```python
import pyarrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions

target = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("amount", pa.float64()),
])
raw = pa.table({"id": ["1", "2"], "amount": ["10.5", "20.0"]})
out = cast_arrow_tabular(raw, CastOptions(target_field=target))
```

### Retry + parallel

```python
from yggdrasil.pyutils import retry, parallelize

@retry(tries=5, delay=0.5, backoff=2.0)
def fetch(url: str) -> bytes: ...

@parallelize(max_workers=8)
def process(item: str) -> dict:
    return {"result": item.upper()}

results = list(process(["a", "b", "c"]))
```

### Databricks — SQL with typed results

```python
from yggdrasil.databricks.workspaces import Workspace
from yggdrasil.databricks.sql import SQLEngine

ws = Workspace(host="https://<workspace>", token="<pat>").connect()
engine = SQLEngine(catalog_name="main", schema_name="analytics", workspace=ws)

result = engine.execute("SELECT id, amount FROM transactions LIMIT 100")
df = result.to_pandas()
arrow_table = result.to_arrow_table()
```

## Module map

| Module | Key exports |
|---|---|
| `yggdrasil.arrow` | `arrow_field_from_hint` |
| `yggdrasil.arrow.cast` | `cast_arrow_tabular`, `cast_arrow_array` |
| `yggdrasil.data.cast` | `CastOptions`, `convert`, `register_converter` |
| `yggdrasil.dataclasses` | `dataclass_to_arrow_field` |
| `yggdrasil.pandas.cast` | `cast_pandas_dataframe` |
| `yggdrasil.polars.cast` | `cast_polars_dataframe`, `cast_polars_lazyframe` |
| `yggdrasil.spark.cast` | `cast_spark_dataframe` |
| `yggdrasil.pyutils` | `retry`, `parallelize` |
| `yggdrasil.concurrent` | `JobPoolExecutor`, `Job` |
| `yggdrasil.requests` | `YGGSession` |
| `yggdrasil.io` | `BytesIO`, `Codec`, `MediaType` |
| `yggdrasil.deltalake` | `DeltaTable` |
| `yggdrasil.databricks` | `Workspace`, `SQLEngine`, `Cluster`, `NotebookConfig` |

## Docs

[Module reference →](docs/modules/README.md)

## Test

```bash
cd python/
pytest
ruff check .
mypy
```
