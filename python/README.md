# Yggdrasil (Python package)

Yggdrasil (`ygg` on PyPI, `yggdrasil` in imports) is a schema-aware data interchange library. It centers on an Arrow-first conversion registry that can cast values across Python types, Arrow, Polars, pandas, Spark, and Databricks-oriented workflows.

## Install

```bash
cd python
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

Install optional integrations only when needed:

```bash
uv pip install -e .[polars]
uv pip install -e .[pandas]
uv pip install -e .[spark]
uv pip install -e .[databricks]
uv pip install -e .[api]
uv pip install -e .[pickle]
```

## Progressive examples (easy → advanced)

### 1) Cast a scalar value

```python
from yggdrasil.data.cast.registry import convert

age = convert("42", int)
active = convert("true", bool)
print(age, active)  # 42 True
```

### 2) Cast a dictionary into a dataclass

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class User:
    id: int
    email: str
    active: bool = True

payload = {"id": "7", "email": "ada@example.com", "active": "false"}
user = convert(payload, User)
print(user)  # User(id=7, email='ada@example.com', active=False)
```

### 3) Infer Arrow schema from Python hints

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow import arrow_field_from_hint

field = arrow_field_from_hint(list[int], name="scores")
schema = pa.schema([field])
print(schema)
```

### 4) Cast Arrow table with explicit `CastOptions`

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions

raw = pa.table({"id": ["1", "2"], "score": ["9.1", "8.7"]})
target = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])

out = cast_arrow_tabular(raw, CastOptions(target_field=target, strict_match_names=True))
print(out.schema)
```

### 5) Use lazy optional dependency guards (`lib.py` pattern)

```python
from yggdrasil.polars.lib import polars
from yggdrasil.pandas.lib import pandas

pl_df = polars.DataFrame({"id": [1, 2]})
pd_df = pandas.DataFrame({"id": [1, 2]})
```

### 6) Use IO buffers + HTTP session

```python
from yggdrasil.io import BytesIO
from yggdrasil.io.http_ import HTTPSession

with BytesIO() as buf:
    buf.write(b"hello")
    buf.seek(0)
    print(buf.media_type)

session = HTTPSession()
response = session.get("https://example.com")
print(response.status)
```

Batch + prepared requests:

```python
from yggdrasil.io.http_ import HTTPSession

http = HTTPSession()
req = http.prepare_request("POST", "https://httpbin.org/post", json={"x": 1})
resp = http.send(req)
print(resp.status, resp.json().get("json"))
```

### 7) Databricks SQL execution (Arrow-first results)

```python
from yggdrasil.databricks import DatabricksClient

stmt = DatabricksClient(host="https://<workspace>", token="<token>").sql.execute("SELECT 1 AS value")
print(stmt.to_arrow_table())
```

### 8) Utility decorators for retries and concurrency

```python
from yggdrasil.pyutils import retry, parallelize

@retry(tries=3, delay=0.1, backoff=2)
def flaky(x: int) -> int:
    return x

@parallelize(max_workers=4)
def square(x: int) -> int:
    return x * x

print(list(square(range(5))))
```

## Documentation map

- `docs/README.md` – full Python docs walkthrough.
- `docs/modules.md` – concise module index.
- `docs/modules/*` – focused module pages for core APIs.

## Documentation website

- GitHub Pages docs (MkDocs): https://platob.github.io/Yggdrasil/
- Source docs: `python/docs/`

## Development checks

```bash
cd python
pytest
ruff check
black .
```
