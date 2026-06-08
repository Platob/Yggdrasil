"""Benchmark: SQL parsing + plan execution throughput.

Measures:
- SQL parsing speed (tokens/s, queries/s)
- Plan execution against ArrowTabular and Folder
- from_sql → to_sql round-trip cost
- Function registry lookup speed
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import shutil
import time
from typing import Callable

import pyarrow as pa


def _time_one(
    label: str, fn: Callable[[], None], *, repeat: int = 5, inner: int = 1000,
) -> dict:
    for _ in range(min(inner, 200)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    best = min(samples)
    med = statistics.median(samples)
    mean = statistics.fmean(samples)
    unit = "µs" if best < 0.001 else "ms" if best < 1 else "s"
    scale = 1e6 if unit == "µs" else 1e3 if unit == "ms" else 1
    print(
        f"  {label:40s}  best={best * scale:8.1f}{unit}  "
        f"median={med * scale:8.1f}{unit}  mean={mean * scale:8.1f}{unit}"
    )
    return {"label": label, "best": best, "median": med, "mean": mean}


def bench_sql_parsing(repeat: int = 5) -> None:
    from yggdrasil.saga.plan.sql_parser import SQLQueryParser, _tokenize_query
    from yggdrasil.saga.plan.databricks import DatabricksSQLParser
    from yggdrasil.enums import Dialect

    print("\n=== SQL Parsing ===")

    simple = "SELECT id, name FROM users WHERE id > 10 LIMIT 100"
    medium = (
        "SELECT u.id, u.name, o.amount "
        "FROM users u "
        "INNER JOIN orders o ON u.id = o.user_id "
        "WHERE u.region = 'US' AND o.amount > 50 "
        "ORDER BY o.amount DESC LIMIT 10"
    )
    complex_ = (
        "WITH top_users AS ("
        "  SELECT region, COUNT(*) AS cnt, SUM(score) AS total "
        "  FROM users WHERE score > 50 GROUP BY region HAVING COUNT(*) > 2"
        ") "
        "SELECT t.region, t.cnt, t.total, "
        "ROW_NUMBER() OVER (ORDER BY t.total DESC) AS rn "
        "FROM top_users t "
        "ORDER BY t.total DESC LIMIT 5"
    )
    databricks = (
        "SELECT DATE_TRUNC('month', ts) AS month, "
        "COALESCE(name, 'unknown') AS name, "
        "EXTRACT(YEAR FROM ts) AS yr, "
        "CASE WHEN score > 90 THEN 'A' WHEN score > 80 THEN 'B' ELSE 'C' END AS grade "
        "FROM users "
        "WHERE ts > CURRENT_TIMESTAMP() - INTERVAL '30' DAY "
        "AND UPPER(region) = 'US'"
    )

    _time_one("tokenize (simple)", lambda: _tokenize_query(simple, Dialect.DATABRICKS),
              repeat=repeat, inner=5000)
    _time_one("tokenize (complex)", lambda: _tokenize_query(complex_, Dialect.DATABRICKS),
              repeat=repeat, inner=2000)
    _time_one("parse (simple)", lambda: SQLQueryParser(simple, Dialect.ANSI).parse(),
              repeat=repeat, inner=2000)
    _time_one("parse (medium join)", lambda: SQLQueryParser(medium, Dialect.ANSI).parse(),
              repeat=repeat, inner=1000)
    _time_one("parse (CTE+window+groupby)",
              lambda: SQLQueryParser(complex_, Dialect.ANSI).parse(),
              repeat=repeat, inner=500)
    _time_one("parse databricks (functions)",
              lambda: DatabricksSQLParser(databricks, Dialect.DATABRICKS).parse(),
              repeat=repeat, inner=500)


def bench_plan_execution(repeat: int = 5) -> None:
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.saga.plan import parse_sql

    print("\n=== Plan Execution (ArrowTabular) ===")

    rows = 10_000
    table = pa.table({
        "id": list(range(rows)),
        "name": [f"user_{i}" for i in range(rows)],
        "region": ["US" if i % 2 == 0 else "EU" for i in range(rows)],
        "score": [50 + (i * 7) % 51 for i in range(rows)],
    })
    source = ArrowTabular(table)
    tables = {"users": source}

    _time_one("SELECT * (10k rows)",
              lambda: parse_sql("SELECT * FROM users").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=200)
    _time_one("WHERE filter (10k→~5k)",
              lambda: parse_sql("SELECT * FROM users WHERE score > 75").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=200)
    _time_one("SELECT+WHERE+LIMIT",
              lambda: parse_sql("SELECT id, name FROM users WHERE score > 80 LIMIT 100").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=200)
    _time_one("ORDER BY+LIMIT",
              lambda: parse_sql("SELECT * FROM users ORDER BY score DESC LIMIT 10").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=100)
    _time_one("GROUP BY COUNT",
              lambda: parse_sql("SELECT region, COUNT(*) AS cnt FROM users GROUP BY region").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=200)
    _time_one("GROUP BY SUM+AVG",
              lambda: parse_sql("SELECT region, SUM(score) AS total, AVG(score) AS avg FROM users GROUP BY region").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=200)


def bench_folder_execution(repeat: int = 5) -> None:
    from yggdrasil.path.local_path import LocalPath
    from yggdrasil.path.folder import Folder
    from yggdrasil.saga.plan import parse_sql

    print("\n=== Plan Execution (Folder, disk-backed) ===")

    tmpdir = tempfile.mkdtemp()
    try:
        rows = 10_000
        table = pa.table({
            "id": list(range(rows)),
            "name": [f"user_{i}" for i in range(rows)],
            "region": ["US" if i % 2 == 0 else "EU" for i in range(rows)],
            "score": [50 + (i * 7) % 51 for i in range(rows)],
        })
        folder = Folder(path=LocalPath(tmpdir))
        folder.write_table(table)
        tables = {"users": folder}

        _time_one("Folder SELECT * (10k)",
                  lambda: parse_sql("SELECT * FROM users").execute(tables=tables).read_arrow_table(),
                  repeat=repeat, inner=50)
        _time_one("Folder WHERE (10k→~5k)",
                  lambda: parse_sql("SELECT * FROM users WHERE score > 75").execute(tables=tables).read_arrow_table(),
                  repeat=repeat, inner=50)
        _time_one("Folder GROUP BY",
                  lambda: parse_sql("SELECT region, COUNT(*) AS cnt FROM users GROUP BY region").execute(tables=tables).read_arrow_table(),
                  repeat=repeat, inner=50)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_roundtrip(repeat: int = 5) -> None:
    from yggdrasil.saga.plan import parse_sql

    print("\n=== SQL Round-trip (parse → emit → re-parse) ===")

    queries = [
        "SELECT id, name FROM users WHERE id > 10 LIMIT 100",
        "SELECT * FROM a INNER JOIN b ON a.id = b.id WHERE a.score > 80",
        "WITH cte AS (SELECT * FROM t WHERE x > 0) SELECT * FROM cte",
        "SELECT region, COUNT(*) FROM users GROUP BY region HAVING COUNT(*) > 5 ORDER BY region",
    ]
    for q in queries:
        label = q[:50] + "..." if len(q) > 50 else q
        def _rt(sql=q):
            node = parse_sql(sql, dialect="databricks")
            emitted = node.to_sql(dialect="databricks")
            parse_sql(emitted, dialect="databricks")
        _time_one(f"RT: {label}", _rt, repeat=repeat, inner=500)


def bench_registry(repeat: int = 5) -> None:
    from yggdrasil.saga.plan.func_registry import BUILTIN_REGISTRY

    print("\n=== Function Registry Lookup ===")

    _time_one("is_known (hit)", lambda: BUILTIN_REGISTRY.is_known("DATE_TRUNC"),
              repeat=repeat, inner=50000)
    _time_one("is_known (miss)", lambda: BUILTIN_REGISTRY.is_known("NOT_A_FUNCTION"),
              repeat=repeat, inner=50000)
    _time_one("get (hit)", lambda: BUILTIN_REGISTRY.get("ROW_NUMBER"),
              repeat=repeat, inner=50000)


def bench_udf_execution(repeat: int = 5) -> None:
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.saga.plan import parse_sql
    from yggdrasil.saga.plan.func_registry import BUILTIN_REGISTRY

    print("\n=== UDF / Arrow Kernel Execution ===")

    rows = 10_000
    table = pa.table({
        "name": [f"user_{i}" for i in range(rows)],
        "score": [50.5 + (i * 7) % 51 for i in range(rows)],
        "val": [float(i) for i in range(rows)],
    })
    source = ArrowTabular(table)
    tables = {"t": source}

    _time_one("UPPER(name) 10k rows",
              lambda: parse_sql("SELECT UPPER(name) AS u FROM t").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=100)
    _time_one("ABS(score) 10k rows",
              lambda: parse_sql("SELECT ABS(score) AS a FROM t").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=100)
    _time_one("SQRT(val) 10k rows",
              lambda: parse_sql("SELECT SQRT(val) AS s FROM t").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=100)
    _time_one("COALESCE(score, val) 10k",
              lambda: parse_sql("SELECT COALESCE(score, val) AS c FROM t").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=100)
    _time_one("id+UPPER+ABS 10k (3 cols)",
              lambda: parse_sql("SELECT UPPER(name) AS u, ABS(score) AS a, SQRT(val) AS s FROM t").execute(tables=tables).read_arrow_table(),
              repeat=repeat, inner=100)

    _time_one("direct pc.utf8_upper 10k",
              lambda: pc.utf8_upper(table.column("name")),
              repeat=repeat, inner=500)
    _time_one("registry.apply_arrow UPPER",
              lambda: BUILTIN_REGISTRY.apply_arrow("UPPER", table.column("name")),
              repeat=repeat, inner=500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    r = args.repeat

    bench_sql_parsing(r)
    bench_plan_execution(r)
    bench_folder_execution(r)
    bench_roundtrip(r)
    bench_registry(r)
    bench_udf_execution(r)


if __name__ == "__main__":
    main()
