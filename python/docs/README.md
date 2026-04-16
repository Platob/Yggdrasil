# Yggdrasil Python documentation

This guide is intentionally ordered from the easiest feature to the most advanced implementation patterns used in the current codebase.

## 1) Setup

```bash
cd python
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

Optional extras:

```bash
uv pip install -e .[polars]
uv pip install -e .[pandas]
uv pip install -e .[spark]
uv pip install -e .[databricks]
uv pip install -e .[api]
```

---

## 2) Easiest: convert a scalar

```python
from yggdrasil.data.cast.registry import convert

print(convert("42", int))
print(convert("true", bool))
```

---

## 3) Convert dictionaries into typed dataclasses

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class Order:
    id: int
    amount: float

order = convert({"id": "10", "amount": "99.5"}, Order)
print(order)
```

---

## 4) Infer Arrow fields from Python type hints

```python
from yggdrasil.arrow import arrow_field_from_hint

field = arrow_field_from_hint(dict[str, list[int]], name="payload")
print(field)
```

---

## 5) Cast tabular Arrow data with `CastOptions`

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions

raw = pa.table({"id": ["1", "2"], "score": ["3.14", "2.71"]})

target = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])

opts = CastOptions(target_field=target, strict_match_names=True)
out = cast_arrow_tabular(raw, opts)
print(out.schema)
```

---

## 6) Engine-specific casting (Polars / pandas / Spark)

Always import optional dependencies through their `lib.py` guard modules:

```python
from yggdrasil.polars.lib import polars
from yggdrasil.pandas.lib import pandas
```

Polars cast example:

```python
import yggdrasil.arrow as pa
from yggdrasil.polars.cast import cast_polars_dataframe
from yggdrasil.data.cast import CastOptions
from yggdrasil.polars.lib import polars

source_df = polars.DataFrame({"id": ["1"], "value": ["4.2"]})
target = pa.schema([pa.field("id", pa.int64()), pa.field("value", pa.float64())])
out_df = cast_polars_dataframe(source_df, CastOptions(target_field=target))
```

---

## 7) IO and HTTP subsystem

Preferred client for new HTTP behavior:

```python
from yggdrasil.io.http_ import HTTPSession

session = HTTPSession()
resp = session.get("https://example.com")
print(resp.status)
```

Buffer / media detection:

```python
from yggdrasil.io import BytesIO

with BytesIO() as buf:
    buf.write(b"hello")
    buf.seek(0)
    print(buf.compression)
    print(buf.media_type)
```

Legacy retry-only session:

```python
from yggdrasil.requests import YGGSession

legacy = YGGSession(num_retry=3)
print(legacy.get("https://example.com", timeout=10).status_code)
```

---

## 8) Databricks SQL workflow

```python
from yggdrasil.databricks import DatabricksClient

stmt = DatabricksClient(host="https://<workspace>", token="<token>").sql.execute("SELECT current_timestamp() AS ts")
print(stmt.to_arrow_table())
```

---

## 9) Advanced: reuse `CastOptions.check_arg` in custom helpers

```python
from yggdrasil.data.cast import CastOptions


def normalize_options(options=None, target_field=None) -> CastOptions:
    return CastOptions.check(options, target_field=target_field, strict_match_names=True)
```

---

## 10) Module map

- Core casting: `yggdrasil.data.cast`, `yggdrasil.arrow`, `yggdrasil.dataclasses`
- Engines: `yggdrasil.polars`, `yggdrasil.pandas`, `yggdrasil.spark`
- IO / HTTP: `yggdrasil.io`, `yggdrasil.io.http_`, `yggdrasil.requests`
- Databricks: `yggdrasil.databricks.*`
- Utilities: `yggdrasil.pyutils`, `yggdrasil.concurrent`, `yggdrasil.environ`, `yggdrasil.pickle`, `yggdrasil.mongo`, `yggdrasil.mongoengine`

For per-module pages, see `python/docs/modules/` and `python/docs/modules.md`.
