# Yggdrasil Python docs

Clean guide for `yggdrasil`.
Short phrases. Fast start. Clear module map.

## 1) Install

```bash
cd python
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

Install only what you need:

```bash
uv pip install -e .[polars]
uv pip install -e .[pandas]
uv pip install -e .[spark]
uv pip install -e .[databricks]
uv pip install -e .[api]
```

---

## 2) Easiest use case: cast one value

```python
from yggdrasil.data.cast.registry import convert

age = convert("42", int)
active = convert("true", bool)
print(age, active)
```

---

## 3) Cast dict -> dataclass

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class User:
    id: int
    email: str
    active: bool

raw = {"id": "7", "email": "a@x.com", "active": "false"}
user = convert(raw, User)
print(user)
```

---

## 4) Build Arrow schema from type hints

```python
from dataclasses import dataclass
from yggdrasil.arrow import arrow_field_from_hint

@dataclass
class Event:
    ts: str
    value: float

field = arrow_field_from_hint(Event, name="event")
print(field)
```

---

## 5) Cast table with schema options

```python
import pyarrow as pa
from yggdrasil.data.cast import CastOptions
from yggdrasil.arrow.cast import cast_arrow_tabular

raw = pa.table({"id": ["1", "2"], "score": ["1.2", "2.3"]})
target = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])

out = cast_arrow_tabular(raw, CastOptions(target_field=target, strict_match_names=True))
print(out.schema)
```

---

## 6) Optional engines (safe imports)

Always import optional engines from `lib.py` guards.

```python
from yggdrasil.polars.lib import polars
from yggdrasil.pandas.lib import pandas
import pyspark.sql as pyspark_sql
```

Polars cast example:

```python
from yggdrasil.polars.cast import cast_polars_dataframe
from yggdrasil.data.cast import CastOptions

pl_df = polars.DataFrame({"id": ["1"], "name": ["Ada"]})
# opts can be pa.Schema, pa.Field, pa.DataType, dict, or CastOptions
opts = CastOptions.check_arg(None)
out = cast_polars_dataframe(pl_df, opts)
```

---

## 7) IO and HTTP sessions

Preferred HTTP client:

```python
from yggdrasil.io.http_.session import HTTPSession

session = HTTPSession()
response = session.get("https://example.com")
print(response.status)
```

Buffer + media type detection:

```python
from yggdrasil.io import BytesIO

with BytesIO() as buf:
    buf.write(b"hello")
    buf.seek(0)
    print(buf.media_type)
```

Legacy retry session:

```python
from yggdrasil.requests import YGGSession

s = YGGSession(num_retry=3)
print(s.get("https://example.com", timeout=10).status_code)
```

---

## 8) Databricks SQL workflow

```python
from yggdrasil.databricks.workspaces import Workspace
from yggdrasil.databricks.sql import SQLEngine

ws = Workspace(host="https://<workspace>", token="<token>")
engine = SQLEngine(workspace=ws)
result = engine.execute("SELECT 1 AS value")
print(result)
```

---

## 9) Full module and submodule map

### Core data and schema
- `yggdrasil.data.cast` - converter registry, dispatch, `CastOptions`.
- `yggdrasil.data.enums` - timezone, currency, geozone helpers.
- `yggdrasil.arrow` - Python type hint -> Arrow field/schema.
- `yggdrasil.dataclasses` - dataclass -> Arrow field helpers.

### Dataframe engines
- `yggdrasil.polars` - Polars converters, extensions, guarded import.
- `yggdrasil.pandas` - pandas converters, extensions, guarded import.
- `yggdrasil.spark` - Spark converters, extensions, guarded import.

### IO and transport
- `yggdrasil.io` - URL, request/response models, buffers, codecs.
- `yggdrasil.io.http_` - `HTTPSession` (preferred modern HTTP client).
- `yggdrasil.io.buffer` - Arrow IPC, Parquet, JSON, ZIP buffer readers.
- `yggdrasil.requests` - `YGGSession` and `MSALSession` (legacy/simple HTTP).

### Databricks
- `yggdrasil.databricks` - root package.
- `yggdrasil.databricks.workspaces` - workspace client + paths/files.
- `yggdrasil.databricks.sql` - SQL engine and statement results.
- `yggdrasil.databricks.compute` - cluster and remote compute helpers.
- `yggdrasil.databricks.jobs` - typed notebook parameter contracts.
- `yggdrasil.databricks.account` / `iam` / `secrets` / `fs` - account APIs.

### AI and API
- `yggdrasil.ai` - OpenAI-backed sessions and SQL generation sessions.
- `yggdrasil.fastapi` - optional REST API routers/services/schemas.

### Utilities
- `yggdrasil.pyutils` - `retry`, `parallelize`, misc helpers.
- `yggdrasil.concurrent` - bounded job executor.
- `yggdrasil.environ` - runtime import/install helpers.
- `yggdrasil.mongo` and `yggdrasil.mongoengine` - MongoDB integrations.
- `yggdrasil.pickle` - custom serialization and serde helpers.
- `yggdrasil.fxrates` - FX-related helpers.
- `yggdrasil.blake3` / `yggdrasil.xxhash` - hash wrappers.

---

## 11) Learning path (easy -> advanced)

1. `data.cast.registry.convert`
2. `arrow_field_from_hint`
3. `CastOptions` + `arrow.cast`
4. Polars/pandas/Spark casts
5. `io.http_.HTTPSession` + `io.buffer`
6. Databricks SQL + jobs
7. AI SQL generation
8. FastAPI and platform integrations

