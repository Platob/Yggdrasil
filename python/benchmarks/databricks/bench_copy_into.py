"""Live A/B benchmark: ``COPY INTO`` vs ``INSERT INTO … SELECT`` from a Volume.

Measures the *warehouse* wall-clock time of the two load shapes the
volume-insert path can take — the part :mod:`bench_databricks_insert_staging`
declares out of scope (it benches the in-process Parquet write only):

* **single file** — ``COPY INTO t FROM (SELECT … FROM '<file>')`` vs
  ``INSERT INTO t (cols) SELECT … FROM parquet.`<file>```
* **batch of N files** — one ``COPY INTO … FILES = (…)`` over a shared dir
  vs ``INSERT INTO t (cols) SELECT … UNION ALL …`` across the N files

Methodology (so the numbers are honest):

* the warehouse is warmed with a trivial query before timing;
* every trial stages **fresh, unique** Parquet files, so ``COPY INTO``'s
  file-level idempotency never lets it skip work it should do;
* loads append into a Delta table (append cost is ~independent of current
  size), and we report the **median** of ``--repeat`` trials.

Requires a live workspace (``DATABRICKS_HOST`` + creds). Usage::

    PYTHONPATH=src python benchmarks/databricks/bench_copy_into.py
    PYTHONPATH=src python benchmarks/databricks/bench_copy_into.py \\
        --rows 250000 --files 8 --repeat 5

Captured on a live SQL warehouse (rows/file=250000, repeat=6, median,
load time only — staging excluded; lower is better)::

                COPY INTO   INSERT SEL   delta
    single file   2446 ms     2149 ms    +14%   ← COPY's per-load overhead loses
    batch x8      3311 ms     4956 ms    -33%   ← one bulk load beats UNION ALL

So the insert path uses ``COPY INTO`` only for multi-file append batches
and keeps ``INSERT … SELECT`` for a single staged file.
"""
from __future__ import annotations

import argparse
import secrets
import statistics
import time

import numpy as np
import pyarrow as pa

from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.table.insert import make_sql_copy_into
from yggdrasil.databricks.sql.sql_utils import quote_ident
from yggdrasil.enums import Mode


def _dataset(rows: int) -> pa.Table:
    rng = np.random.default_rng(7)
    return pa.table({
        "id": pa.array(np.arange(rows, dtype=np.int64)),
        "name": pa.array(np.array([f"name-{i}" for i in range(rows)], dtype=object)),
        "amount": pa.array(rng.normal(0.0, 1.0, size=rows)),
        "qty": pa.array(rng.integers(0, 1000, size=rows, dtype=np.int32)),
        "active": pa.array(rng.integers(0, 2, size=rows, dtype=np.int8).astype(bool)),
    })


def _insert_union_sql(loc: str, columns: list[str], paths: list[str]) -> str:
    cols = ", ".join(quote_ident(c) for c in columns)
    selects = [f"SELECT {cols} FROM parquet.`{p}`" for p in paths]
    return f"INSERT INTO {loc} ({cols})\n" + "\nUNION ALL\n".join(selects)


def _median_ms(stage_fn, sql_fn, exec_fn, repeat: int) -> float:
    """Median wall time of ``exec_fn(sql)`` only — staging is excluded so the
    number isolates the warehouse load, not the parquet write + upload."""
    samples = []
    for _ in range(repeat):
        paths = stage_fn()
        sql = sql_fn(paths)
        t0 = time.perf_counter()
        exec_fn(sql)
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=250_000, help="Rows per staged file.")
    ap.add_argument("--files", type=int, default=8, help="Files in the batch case.")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--schema", default=None)
    args = ap.parse_args()

    client = DatabricksClient()
    catalog = args.catalog or client.catalog_name or "trading_tgp_dev"
    schema = args.schema or client.schema_name or "ygg_integration"
    from yggdrasil.data import Field, Schema

    fields = Schema.from_fields([
        Field("id", "int64"), Field("name", "string"), Field("amount", "double"),
        Field("qty", "int32"), Field("active", "bool"),
    ])
    columns = [f.name for f in fields.fields]
    data = _dataset(args.rows)

    suffix = secrets.token_hex(3)
    t_copy = client.tables.table(f"{catalog}.{schema}.bench_copy_{suffix}")
    t_ins = client.tables.table(f"{catalog}.{schema}.bench_ins_{suffix}")
    created = []
    try:
        for t in (t_copy, t_ins):
            t.ensure_created(fields)
            created.append(t)

        client.sql.execute("SELECT 1")  # warm the warehouse

        def stage(n: int) -> list[str]:
            """Stage *n* fresh unique Parquet files in one dir; return paths."""
            vp = t_copy.insert_volume_path(t_copy, temporary=False)
            base_dir = vp.parent
            paths = []
            for _ in range(n):
                f = base_dir / f"bench-{secrets.token_hex(8)}.parquet"
                f.write_table(data, mode=Mode.OVERWRITE)
                paths.append(f.full_path())
            return paths

        loc_copy = t_copy.full_name(safe=True)
        loc_ins = t_ins.full_name(safe=True)

        run = lambda sql: client.sql.execute(sql, raise_error=True)  # noqa: E731
        copy_sql = lambda loc: (lambda p: make_sql_copy_into(loc, columns, paths=p))  # noqa: E731
        ins_sql = lambda loc: (lambda p: _insert_union_sql(loc, columns, p))  # noqa: E731

        print(f"# rows/file={args.rows}  files={args.files}  repeat={args.repeat}")
        print(f"# {'case':>16s}  {'COPY INTO':>12s}  {'INSERT SEL':>12s}  {'delta':>8s}")

        c1 = _median_ms(lambda: stage(1), copy_sql(loc_copy), run, args.repeat)
        i1 = _median_ms(lambda: stage(1), ins_sql(loc_ins), run, args.repeat)
        _row("single file", c1, i1)

        cN = _median_ms(lambda: stage(args.files), copy_sql(loc_copy), run, args.repeat)
        iN = _median_ms(lambda: stage(args.files), ins_sql(loc_ins), run, args.repeat)
        _row(f"batch x{args.files}", cN, iN)
    finally:
        for t in created:
            try:
                t.delete(missing_ok=True)
            except Exception:
                pass


def _row(case: str, copy_ms: float, insert_ms: float) -> None:
    delta = (copy_ms - insert_ms) / insert_ms * 100 if insert_ms else 0.0
    print(f"  {case:>16s}  {copy_ms:>9.0f} ms  {insert_ms:>9.0f} ms  {delta:>+6.0f}%")


if __name__ == "__main__":
    main()
