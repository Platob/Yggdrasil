# Developer starter templates

These templates demonstrate common integration points for the Yggdrasil Python
package. Copy/paste a snippet and adapt it to your environment.

## 1) Enhanced dataclasses for data contracts

Use the custom `@dataclass` decorator to add serialization helpers, safe
initialization, and Arrow-aware schemas to your models.

```python
from yggdrasil.dataclasses import dataclass

@dataclass
class Customer:
    id: int
    name: str
    active: bool = True

# Construct objects with type-safe coercion
payload = {"id": "101", "name": "Ada"}
customer = Customer.from_dict(payload)  # converts types and applies defaults

# Round-trip support
row = customer.to_dict()
tuple_view = customer.to_tuple()
clone = Customer.from_tuple(tuple_view)

# Generate Arrow fields for downstream engines
id_field = Customer.arrow_field("id")
print(id_field)
```

## 2) Resilient HTTP calls with retry

Wrap HTTP access in `YGGSession` to get connection retries out of the box.

```python
from yggdrasil.requests import YGGSession

with YGGSession(num_retry=3) as session:
    session.headers["User-Agent"] = "yggdrasil-client/0.1"
    response = session.get("https://example.com/api")
    response.raise_for_status()
    data = response.json()
```

## 3) Azure AD client credentials with automatic token refresh

Authenticate API calls using `MSALAuth` and `MSALSession`. Configuration can
come from environment variables (`AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`,
`AZURE_TENANT_ID`, `AZURE_SCOPES`) or constructor arguments.

```python
from yggdrasil.requests import MSALAuth

auth = MSALAuth(scopes=["api://my-app/.default"])
session = auth.requests_session()

response = session.get("https://resource.example.com/data")
response.raise_for_status()
print(response.json())
```

## 4) Guarding optional dependencies

Use the dependency guard decorators to give users clear error messages when a
library is missing. Decorators can be used with or without parentheses.

```python
from yggdrasil.libs import require_polars, require_pyspark

@require_polars
def build_polars_dataframe(record_batch):
    import polars as pl
    return pl.DataFrame(record_batch)

@require_pyspark(active_session=True)
def read_spark_table(name: str):
    from pyspark.sql import SparkSession
    return SparkSession.getActiveSession().table(name)
```

## 5) Converting Arrow schemas to Polars or Spark

Leverage the conversion utilities to keep data type handling consistent across
processing engines.

```python
import pyarrow as pa
from yggdrasil.libs import polarslib, sparklib

arrow_type = pa.struct([
    pa.field("timestamp", pa.timestamp("us", tz="UTC")),
    pa.field("payload", pa.large_string()),
])

# Polars dtype objects
pl_struct = polarslib.arrow_type_to_polars_type(arrow_type)

# Spark StructType fields
spark_struct = sparklib.arrow_type_to_spark_type(arrow_type)
```

## 6) Auto-install helpers for truly dynamic environments

If your integration runs in an environment where dependencies may be missing,
wrap entry points with `check_modules` to automatically install and retry when
imports fail.

```python
from yggdrasil.libs.modules import check_modules

@check_modules("polars")
def ensure_polars_available():
    import polars as pl
    return pl.DataFrame({"x": [1, 2, 3]})
```
