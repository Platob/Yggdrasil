# Yggdrasil

**Schema-aware data interchange for Python.** One conversion registry that moves values cleanly between Python types, dataclasses, Arrow, Polars, pandas, Spark, Databricks, and the wire — without losing schema, nullability, or metadata.

- **PyPI:** [`ygg`](https://pypi.org/project/ygg/) · **Import:** `yggdrasil`
- **Source:** [github.com/Platob/Yggdrasil](https://github.com/Platob/Yggdrasil)
- **License:** [Apache-2.0](https://github.com/Platob/Yggdrasil/blob/main/LICENSE)

```bash
pip install ygg
```

---

## Why people pick this up

- Stop writing brittle hand-rolled casts between dicts, dataframes, and warehouse schemas.
- Treat **Arrow schema** as the contract: names, order, nullability, metadata, nested structure are preserved across boundaries.
- Use **one converter registry** instead of separate utilities per engine.
- **Optional dependencies** — pull `pandas` / `polars` / `spark` / `databricks` only when you need them.
- **Rust fast path, Python canonical** — `yggrs` accelerates hot paths; everything works without it.

---

## 60-second tour

=== "Cast scalars"

    ```python
    from yggdrasil.data.cast.registry import convert

    convert("42", int)              # 42
    convert("yes", bool)            # True
    convert("2024-06-01", "date")   # datetime.date(2024, 6, 1)
    ```

=== "Dict → dataclass"

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

=== "Arrow schema contract"

    ```python
    import yggdrasil.arrow as pa
    from yggdrasil.arrow.cast import cast_arrow_tabular
    from yggdrasil.data.cast.options import CastOptions

    raw = pa.table({"id": ["1", "2"], "score": ["9.1", "8.7"]})
    target = pa.schema([
        pa.field("id",    pa.int64(),   nullable=False),
        pa.field("score", pa.float64(), nullable=False),
    ])
    out = cast_arrow_tabular(raw, CastOptions(target_field=target))
    ```

=== "Databricks SQL → any engine"

    ```python
    from yggdrasil.databricks import DatabricksClient

    stmt = DatabricksClient().sql.execute("SELECT * FROM main.default.orders LIMIT 100")
    stmt.to_arrow_table()
    stmt.to_pandas()
    stmt.to_polars()
    stmt.to_spark()
    ```

---

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting Started](getting-started.md)**
  Install, first conversions, a working example for every layer.

- :material-file-tree: **[Architecture](guides/architecture.md)**
  Cast registry, dispatch order, `CastOptions`, optional-dep guards.

- :material-table-arrow-up: **[Casting guide](guides/casting.md)**
  Scalar conversion, schema-aware tabular cast, engine bridges.

- :material-web: **[IO & HTTP](guides/io-http.md)**
  `BytesIO`, `URL`, `HTTPSession`, batch dispatch, response conversions.

- :material-database: **[Databricks](guides/databricks.md)**
  SQL, Unity Catalog, Compute, DBFS/Volumes, Secrets, IAM, Genie.

- :material-tools: **[Development](guides/development.md)**
  Tests, lint, docs, Rust extension, optional dependencies.

- :material-bookshelf: **[Module walkthrough](modules.md)**
  Curated index of focused module pages.

- :material-api: **[API Reference](api/index.md)**
  Auto-generated from the `yggdrasil` source tree.

</div>

---

## Install patterns

```bash
pip install ygg                   # core: pyarrow + polars + yggrs
pip install "ygg[data]"           # pandas, numpy, sqlglot
pip install "ygg[bigdata]"        # pyspark, delta-spark
pip install "ygg[databricks]"     # databricks-sdk
pip install "ygg[api]"            # fastapi, uvicorn, pydantic
pip install "ygg[http]"           # urllib3, xxhash
pip install "ygg[pickle]"         # cloudpickle, dill, zstandard, blake3
pip install "ygg[mongo]"          # mongoengine
pip install "ygg[postgres]"       # psycopg, adbc-driver-postgresql
pip install "ygg[kafka]"          # confluent-kafka
pip install "ygg[delta]"          # deltalake
```

The only hard runtime deps are `pyarrow>=20`, `polars>=1.3`, and the matching `yggrs` wheel.
