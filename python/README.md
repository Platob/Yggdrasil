# `ygg` — Yggdrasil for Python

Schema-aware data interchange for Python teams that move data between Python types, Arrow, Polars, pandas, Spark, and Databricks. One conversion registry, one schema contract, optional dependencies.

- **PyPI:** [`ygg`](https://pypi.org/project/ygg/) · **Import:** `yggdrasil`
- **Docs:** https://platob.github.io/Yggdrasil/
- **Source:** [`python/src/yggdrasil/`](src/yggdrasil/)

```bash
pip install ygg
```

---

## Why pick this up

- Stop hand-writing brittle casting code between app models, dataframes, and warehouse schemas.
- Treat Arrow schema as the contract surface so every tool agrees on field names, nullability, and metadata.
- Use one conversion registry instead of separate utilities per engine.
- Install only what you need beyond the core. Most integrations are optional extras.

---

## Install with the right extras

```bash
pip install ygg                   # core: pyarrow + polars + xxhash + orjson
pip install "ygg[data]"           # reserved for tabular extras
pip install "ygg[bigdata]"        # pyspark + delta-spark
pip install "ygg[delta]"          # deltalake
pip install "ygg[databricks]"     # databricks-sdk
pip install "ygg[api]"            # fastapi + uvicorn + pydantic
pip install "ygg[http]"           # urllib3 + xxhash
pip install "ygg[pickle]"         # cloudpickle + dill + zstandard + xxhash + blake3
pip install "ygg[mongo]"          # mongoengine
pip install "ygg[postgres]"       # psycopg + adbc-driver-postgresql
pip install "ygg[kafka]"          # confluent-kafka
pip install "ygg[dev]"            # everything for local development
```

Editable dev install:

```bash
cd python
uv venv --seed .venv && source .venv/bin/activate
uv pip install -e .[dev]
```

---

## Progressive examples

### 1. Cast scalars

```python
from yggdrasil.data.cast.registry import convert

convert("42", int)              # 42
convert("3.14", float)          # 3.14
convert("yes", bool)            # True
convert("2024-06-01", "date")   # datetime.date(2024, 6, 1)
```

### 2. Dict → typed dataclass

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class User:
    id: int
    email: str
    active: bool = True

convert({"id": "7", "email": "ada@example.com", "active": "false"}, User)
# User(id=7, email='ada@example.com', active=False)
```

### 3. Register a custom converter

```python
from decimal import Decimal
from yggdrasil.data.cast.registry import convert, register_converter

@register_converter(str, Decimal)
def _str_to_decimal(value: str, options=None) -> Decimal:
    return Decimal(value.replace(",", "."))

convert("19,95", Decimal)   # Decimal('19.95')
```

### 4. Infer Arrow fields from Python type hints

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow import arrow_field_from_hint

pa.schema([
    arrow_field_from_hint(int,                 name="id"),
    arrow_field_from_hint(list[str],           name="tags"),
    arrow_field_from_hint(dict[str, float],    name="metrics"),
])
```

### 5. Cast an Arrow table to a target schema

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
print(out.schema)
```

### 6. Convert across engines (Polars / pandas / Spark)

Always import optional engines through their `lib.py` guard:

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

df = polars.DataFrame({"id": ["1", "2"], "value": ["4.2", "5.1"]})
target = pa.schema([pa.field("id", pa.int64()), pa.field("value", pa.float64())])
out = cast_polars_dataframe(df, CastOptions(target_field=target))
```

Arrow ↔ Polars round-trip:

```python
from yggdrasil.polars.cast import (
    arrow_table_to_polars_dataframe,
    polars_dataframe_to_arrow_table,
)

pl_df = arrow_table_to_polars_dataframe(arrow_table)
roundtrip = polars_dataframe_to_arrow_table(pl_df)
```

### 7. Dataclass → Arrow struct field

```python
from dataclasses import dataclass
from yggdrasil.dataclasses import dataclass_to_arrow_field

@dataclass
class Position:
    symbol: str
    quantity: float

field = dataclass_to_arrow_field(Position)
print(field)
```

### 8. HTTP: simple to advanced

```python
from yggdrasil.http_ import HTTPSession

http = HTTPSession()
print(http.get("https://httpbin.org/get").json())
print(http.post("https://httpbin.org/post", json={"name": "alice"}).status)
```

Prepared request + send:

```python
req = http.prepare_request("POST", "https://httpbin.org/post",
                           json={"event": "order_created", "id": 123})
resp = http.send(req)
print(resp.status, resp.json()["json"])
```

Parallel batch dispatch:

```python
from yggdrasil.io import SendManyConfig

reqs = [http.prepare_request("GET", "https://httpbin.org/get", params={"page": i})
        for i in range(1, 11)]
responses = list(http.send_many(reqs, send_config=SendManyConfig(max_workers=5)))
print([r.status for r in responses])
```

Tabular response → engine of your choice:

```python
resp = http.get("https://api.example.com/v1/orders?format=arrow")
table  = resp.to_arrow_table()
pdf    = resp.to_pandas()
plf    = resp.to_polars()
```

### 9. Buffers and URLs

```python
from yggdrasil.io import BytesIO, URL

with BytesIO() as buf:           # spill-to-disk byte buffer with media detection
    buf.write(b"hello")
    buf.seek(0)
    print(buf.media_type, buf.compression)

u = URL.from_str("https://example.com/a/b?q=1")
print(u.host, u.path)
print(u.with_query_items({"q": 2, "lang": "en"}).to_string())
```

### 10. Databricks SQL: read/write across formats

```python
from yggdrasil.databricks import DatabricksClient

c = DatabricksClient(host="https://<workspace>", token="<token>")

c.sql.execute("""
CREATE TABLE IF NOT EXISTS main.default.demo (id BIGINT, name STRING) USING DELTA
""")
c.sql.insert_into("main.default.demo",
                  [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}])

stmt = c.sql.execute("SELECT * FROM main.default.demo ORDER BY id")
print(stmt.to_arrow_table())
print(stmt.to_pandas())
print(stmt.to_polars())
```

`DatabricksClient` also covers Unity Catalog (`c.catalogs["main"]["default"]["orders"]`), Compute (`c.compute.clusters.all_purpose_cluster(...)`), DBFS/Volumes (`c.dbfs_path("/Volumes/...").write_text(...)`), Secrets (`c.secrets["scope/key"] = "..."`), IAM, and Genie. See [docs/guides/databricks.md](docs/guides/databricks.md).

### 11. Typed Databricks job widgets

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class IngestConfig(NotebookConfig):
    catalog: str = "main"
    schema: str = "ingest"
    table: str = "events"
    dry_run: bool = False

cfg = IngestConfig.from_environment()   # in a job run
# cfg = IngestConfig.init_widgets()     # in a local notebook
```

### 12. Retries, parallelism, jobs

```python
from yggdrasil.pyutils import retry, parallelize
from yggdrasil.concurrent import Job, JobPoolExecutor

@retry(tries=3, delay=0.2, backoff=2)
def flaky(x: int) -> int:
    return x

@parallelize(max_workers=4)
def square(x: int) -> int:
    return x * x

list(square(range(6)))   # [0, 1, 4, 9, 16, 25]

# Bounded streaming jobs
jobs = [Job.make(lambda x=x: x * x) for x in range(20)]
with JobPoolExecutor(max_workers=4, max_in_flight=8) as pool:
    for result in pool.as_completed(jobs):
        print(result.value)
```

### 13. Reuse `CastOptions.check`

```python
from yggdrasil.data.cast.options import CastOptions

def normalize_options(options=None, *, target_field=None) -> CastOptions:
    return CastOptions.check(options, target_field=target_field, strict_match_names=True)
```

---

## Modules at a glance

| Module | Purpose |
|---|---|
| `yggdrasil.data` | Cast registry, `CastOptions`, `DataType`, `Field`/`Schema`, `DataTable`, normalized enums |
| `yggdrasil.arrow` | Arrow type inference, casting helpers (`cast_arrow_tabular`, `cast_arrow_record_batch_reader`) |
| `yggdrasil.dataclasses` | `dataclass_to_arrow_field`, `WaitingConfig`, `Expiring`, `ExpiringDict` |
| `yggdrasil.polars` / `yggdrasil.pandas` / `yggdrasil.spark` | Engine bridges (`cast.py`, `lib.py`, `tests.py` TestCase bases) |
| `yggdrasil.io` | `BytesIO`, `URL`, `SendConfig`/`SendManyConfig`, codecs, media types |
| `yggdrasil.http_` | `HTTPSession` (preferred), `PreparedRequest`, `Response` |
| `yggdrasil.databricks` | `DatabricksClient` + `sql`/`compute`/`workspaces`/`fs`/`iam`/`secrets`/`jobs`/`account`/`genie` |
| `yggdrasil.fastapi` | FastAPI service powering the Power Query connector |
| `yggdrasil.pyutils` / `yggdrasil.concurrent` | `retry`, `parallelize`, `Job`, `JobPoolExecutor` |
| `yggdrasil.pickle` / `blake3` / `xxhash` | Optional serialization + hashing |
| `yggdrasil.mongo` / `mongoengine` | Mongo helpers |
| `yggdrasil.fxrates` | FX-rate helpers |

For per-module pages, see [`docs/modules/`](docs/modules/) and the navigable [docs site](https://platob.github.io/Yggdrasil/).

---

## Testing

Tests that touch a dataframe or Arrow object subclass the matching engine `TestCase` from `yggdrasil.<engine>.tests`:

```python
from yggdrasil.arrow.tests import ArrowTestCase

class TestX(ArrowTestCase):
    def test_table(self):
        t = self.table({"id": [1, 2]})
        self.assertSchemaEqual(t.schema, self.pa.schema([self.pa.field("id", self.pa.int64())]))
```

This handles optional-dependency skipping, per-test tmp dirs, Arrow interop, and frame/schema assertions.

```bash
pytest                                                   # full suite
pytest tests/test_yggdrasil/test_data/                   # one area
pytest tests/test_yggdrasil/test_data/test_registry.py   # one file
ruff check
black .
```

`pytest-asyncio` is in `strict` mode — async tests must use the explicit marker. The `integration` marker is skipped unless `DATABRICKS_HOST` is set.

---

## Documentation locally

```bash
cd python
mkdocs serve     # http://127.0.0.1:8000
mkdocs build     # static site → python/site/
```

The published site is deployed by [`.github/workflows/docs.yml`](../.github/workflows/docs.yml) on every push to `main` that touches `python/docs/**`, `python/src/**`, `mkdocs.yml`, or the workflow itself.

---

## License

[Apache-2.0](LICENSE).
