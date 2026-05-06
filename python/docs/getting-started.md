# Getting Started

A copy-pasteable walkthrough from install to a working example for every major layer of the library.

## 1. Install

```bash
pip install ygg
```

Add what you need:

```bash
pip install "ygg[data]"        # pandas + numpy + sqlglot
pip install "ygg[bigdata]"     # pyspark + delta-spark
pip install "ygg[databricks]"  # databricks-sdk
pip install "ygg[api]"         # fastapi + uvicorn + pydantic
pip install "ygg[http]"        # urllib3 + xxhash
```

Editable dev install:

```bash
git clone https://github.com/Platob/Yggdrasil.git
cd Yggdrasil/python
uv venv .venv && source .venv/bin/activate
uv pip install -e .[dev]
```

## 2. First conversion

```python
from yggdrasil.data.cast.registry import convert

convert("42", int)              # 42
convert("3.14", float)          # 3.14
convert("yes", bool)            # True
convert("2024-06-01", "date")   # datetime.date(2024, 6, 1)
```

## 3. Dict → dataclass

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class Order:
    id: int
    amount: float
    paid: bool = False

convert({"id": "7", "amount": "99.50", "paid": "yes"}, Order)
# Order(id=7, amount=99.5, paid=True)
```

## 4. Arrow schema contract

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast.options import CastOptions

raw = pa.table({"id": ["1", "2"], "score": ["9.1", "8.7"]})
target = pa.schema([
    pa.field("id",    pa.int64(),   nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])
out = cast_arrow_tabular(raw, CastOptions(target_field=target, strict_match_names=True))
```

## 5. Engine bridges

Use the `lib.py` guards so base installs without an engine still work:

```python
from yggdrasil.polars.lib import polars
from yggdrasil.pandas.lib import pandas
```

Polars cast:

```python
import yggdrasil.arrow as pa
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.polars.cast import cast_polars_dataframe
from yggdrasil.polars.lib import polars

df = polars.DataFrame({"id": ["1"], "value": ["4.2"]})
target = pa.schema([pa.field("id", pa.int64()), pa.field("value", pa.float64())])
out = cast_polars_dataframe(df, CastOptions(target_field=target))
```

## 6. Register a custom converter

```python
from decimal import Decimal
from yggdrasil.data.cast.registry import convert, register_converter

@register_converter(str, Decimal)
def _str_to_decimal(value: str, options=None) -> Decimal:
    return Decimal(value.replace(",", "."))

convert("19,95", Decimal)   # Decimal('19.95')
```

## 7. HTTP

```python
from yggdrasil.io.http_ import HTTPSession

http = HTTPSession()
print(http.get("https://httpbin.org/get").json())

req = http.prepare_request("POST", "https://httpbin.org/post",
                           json={"event": "ping"})
print(http.send(req).status)
```

Batch:

```python
from yggdrasil.io import SendManyConfig

reqs = [http.prepare_request("GET", "https://httpbin.org/get",
                             params={"page": i}) for i in range(1, 6)]
responses = list(http.send_many(reqs, send_config=SendManyConfig(max_workers=3)))
```

## 8. Databricks SQL

```python
from yggdrasil.databricks import DatabricksClient

stmt = DatabricksClient().sql.execute("SELECT current_user() AS me")
print(stmt.to_polars())
```

Service shortcuts:

| What | Call |
|---|---|
| SQL execution | `DatabricksClient().sql.execute("...")` |
| Unity Catalog | `DatabricksClient().catalogs["main"]["default"]["orders"]` |
| Compute | `DatabricksClient().compute.clusters.all_purpose_cluster(name="etl")` |
| DBFS / Volume | `DatabricksClient().dbfs_path("/Volumes/main/...").write_text("ok")` |
| Secrets | `DatabricksClient().secrets["scope/key"] = "value"` |
| Genie | `DatabricksClient().genie.ask("<space-id>", "weekly revenue")` |

See the [Databricks guide](guides/databricks.md) for full workflows.

## 9. Where next

- [Architecture](guides/architecture.md) — the cast registry and dispatch order.
- [Casting guide](guides/casting.md) — scalar, tabular, and engine-specific casting.
- [IO & HTTP](guides/io-http.md) — buffers, URLs, sessions, batch dispatch.
- [Databricks](guides/databricks.md) — SQL, compute, files, secrets, IAM, Genie.
- [Module walkthrough](modules.md) — curated index of focused module pages.
- [API Reference](api/index.md) — generated from the `yggdrasil` source tree.
