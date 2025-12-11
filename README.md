# Yggdrasil

Utilities for schema-aware data interchange, conversions, and platform integrations. The repository currently focuses on the Python package in `python/`.

## Contents
- [Python package](python/README.md): Installation, quickstart, and module documentation for the `yggdrasil` Python utilities.
- [Module docs](python/docs): Per-module guides (e.g., dataclasses, types, pyutils, requests, libs, databricks, ser). Use the
  directory index in [`python/docs/modules/README.md`](python/docs/modules/README.md) for direct links to each module page.
- [Tests](python/tests): Pytest suite for validating conversions, dataclasses, requests, and platform helpers.

## Getting started
Change into the Python project and follow its README for setup and usage:

```bash
cd python
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

Then explore the quickstart examples in [`python/README.md`](python/README.md).

## Publishing configuration
The GitHub Actions workflow at [`.github/workflows/publish.yml`](.github/workflows/publish.yml) publishes the Python package on pushes to the `main` branch. To authorize uploads:

1. In your PyPI account, create an **API token** scoped to the project (Account settings → API tokens → *Add API token*).
2. In the GitHub repository settings, add a new **Actions secret** named `PYPI_API_TOKEN` with the token value. Use `__token__` as the username is already configured in the workflow.
3. Push to `main` (or trigger the workflow manually). The workflow will build from `python/` and upload to PyPI using the stored secret.

## Type conversion highlights
The `yggdrasil.types.cast.convert` entry point connects multiple dataframe ecosystems with optional Arrow-aware casting hints and convenient scalar parsing:

```python
from yggdrasil.types.cast import convert, CastOptions
import pandas as pd
import polars as pl
import pyarrow as pa
import datetime
from pyspark.sql import SparkSession, types as T

# Polars -> pandas
pl_df = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})
pandas_df = convert(pl_df, pd.DataFrame)

# Spark -> pandas while enforcing a schema via Arrow metadata
spark = SparkSession.builder.master("local[1]").getOrCreate()
spark_df = spark.createDataFrame([(1, "a"), (2, "b")], schema=T.StructType([
    T.StructField("id", T.LongType(), nullable=False),
    T.StructField("value", T.StringType(), nullable=True),
]))

schema_hint = CastOptions.check_arg(
    target_field=pa.field("row", pa.struct([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("value", pa.string()),
    ]))
)
spark_pdf = convert(spark_df, pd.DataFrame, options=schema_hint)

# Arrow field <-> Spark schema for lightweight casting hints
arrow_field = pa.field(
    "payload",
    pa.struct([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("ts", pa.timestamp("us", tz="UTC")),
    ]),
)
spark_struct = convert(arrow_field, T.StructField)
spark_schema = T.StructType([spark_struct])
roundtripped_arrow = convert(spark_schema, pa.Schema)

# Scalar parsing and container casting helpers
ts = convert("2024-01-02T03:04:05Z", datetime.datetime)
values = convert(["1", "2", "3"], list[int])
```

These conversions rely on optional dependencies (Polars, pandas, PySpark, PyArrow); install the relevant extras before running the examples.
