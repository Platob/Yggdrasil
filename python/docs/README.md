# Yggdrasil Python documentation

Hands-on, copy-paste guide that walks from the smallest possible cast up through dataframe engines, HTTP, and Databricks workflows.

The published site lives at **https://platob.github.io/Yggdrasil/** and is built from the same files you see here.

## Quick links

- [Getting Started](getting-started.md) — install + a working example for every layer.
- [Architecture](guides/architecture.md) — cast registry, dispatch order, options.
- [Casting](guides/casting.md) — scalars, tabular, engine bridges.
- [IO & HTTP](guides/io-http.md) — buffers, URLs, sessions, batch dispatch.
- [Databricks](guides/databricks.md) — SQL, compute, files, secrets, IAM, Genie.
- [Development](guides/development.md) — tests, lint, docs, Rust extension.
- [Module index](modules.md) and [module pages](modules/README.md).
- [API Reference](api/index.md) — auto-generated.

## 1. Install

```bash
pip install ygg
pip install "ygg[data]"        # pandas + numpy + sqlglot
pip install "ygg[bigdata]"     # pyspark + delta-spark
pip install "ygg[databricks]"  # databricks-sdk
pip install "ygg[api]"         # fastapi + uvicorn + pydantic
pip install "ygg[http]"        # urllib3 + xxhash
```

## 2. Cast a scalar

```python
from yggdrasil.data.cast.registry import convert

convert("42", int)              # 42
convert("yes", bool)            # True
convert("2024-06-01", "date")   # datetime.date(2024, 6, 1)
```

## 3. Dict → typed dataclass

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class Order:
    id: int
    amount: float
    paid: bool = False

convert({"id": "7", "amount": "99.5", "paid": "yes"}, Order)
```

## 4. Infer Arrow schema from Python hints

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow import arrow_field_from_hint

pa.schema([
    arrow_field_from_hint(int,                 name="id"),
    arrow_field_from_hint(list[str],           name="tags"),
    arrow_field_from_hint(dict[str, float],    name="metrics"),
])
```

## 5. Cast tabular Arrow data

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast.options import CastOptions

raw = pa.table({"id": ["1", "2"], "score": ["3.14", "2.71"]})
target = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])
out = cast_arrow_tabular(raw, CastOptions(target_field=target, strict_match_names=True))
```

## 6. Engine-specific casting

```python
import yggdrasil.arrow as pa
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.polars.cast import cast_polars_dataframe
from yggdrasil.polars.lib import polars

df = polars.DataFrame({"id": ["1"], "value": ["4.2"]})
target = pa.schema([pa.field("id", pa.int64()), pa.field("value", pa.float64())])
out = cast_polars_dataframe(df, CastOptions(target_field=target))
```

`yggdrasil.pandas.cast` and `yggdrasil.spark.cast` follow the same shape.

## 7. HTTP with `HTTPSession`

```python
from yggdrasil.http_ import HTTPSession

http = HTTPSession()
print(http.get("https://httpbin.org/get").json())
print(http.post("https://httpbin.org/post", json={"x": 1}).status)
```

Prepared requests + batch:

```python
from yggdrasil.io import SendManyConfig

reqs = [http.prepare_request("GET", "https://httpbin.org/get",
                             params={"page": p}) for p in range(1, 6)]
responses = list(http.send_many(reqs, send_config=SendManyConfig(max_workers=3)))
```

Buffers and URLs:

```python
from yggdrasil.io import BytesIO, URL

with BytesIO() as buf:
    buf.write(b"hello")
    buf.seek(0)
    print(buf.compression, buf.media_type)

print(URL.from_str("https://example.com/?q=1").host)
```

## 8. Databricks SQL

```python
from yggdrasil.databricks import DatabricksClient

stmt = DatabricksClient().sql.execute("SELECT current_user() AS me")
print(stmt.to_polars())
```

Read/write across formats — see the [SQL module page](modules/databricks/sql/README.md) for the full pattern (write from pyarrow, pandas, polars, pyspark, then read back into any engine).

## 9. Reuse `CastOptions.check`

```python
from yggdrasil.data.cast.options import CastOptions

def normalize_options(options=None, *, target_field=None) -> CastOptions:
    return CastOptions.check(options, target_field=target_field, strict_match_names=True)
```

## 10. Module map

- **Casting:** `yggdrasil.data.cast`, `yggdrasil.arrow`, `yggdrasil.dataclasses`
- **Engines:** `yggdrasil.polars`, `yggdrasil.pandas`, `yggdrasil.spark`
- **IO / HTTP:** `yggdrasil.io`, `yggdrasil.http_`
- **Databricks:** `yggdrasil.databricks.*`
- **Utilities:** `yggdrasil.pyutils`, `yggdrasil.concurrent`, `yggdrasil.environ`, `yggdrasil.pickle`, `yggdrasil.mongo`, `yggdrasil.mongoengine`, `yggdrasil.fastapi`

For per-module pages, see [`modules/`](modules/README.md) and [`modules.md`](modules.md).
