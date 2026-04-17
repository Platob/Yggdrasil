from __future__ import annotations

from yggdrasil.environ.importlib import cached_import

try:
    import pandas

    pandas = pandas
except ImportError:
    pandas = None

try:
    import polars

    polars = polars
except ImportError:
    polars = None

try:
    import pyspark.sql as pyspark_sql

    pyspark_sql = pyspark_sql
except ImportError:
    pyspark_sql = None

__all__ = [
    "get_pandas",
    "get_polars",
    "get_spark_sql",
]


def get_polars():
    global polars
    if polars is None:
        polars = cached_import("polars")
    return polars


def get_pandas():
    global pandas
    if pandas is None:
        pandas = cached_import("pandas")
    return pandas


def get_spark_sql():
    global pyspark_sql
    if pyspark_sql is None:
        pyspark_sql = cached_import("pyspark.sql")
    return pyspark_sql