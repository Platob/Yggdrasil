import decimal as dec
import datetime as dt
import pyarrow as pa
import polars as pl
import pandas as pd


__all__ = [
    "PYTHON_TO_ARROW_TYPE_MAP",
    "ArrowTabular",
    "safe_arrow_tabular"
]


PYTHON_TO_ARROW_TYPE_MAP = {
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    str: pa.utf8(),
    bytes: pa.binary(),
    memoryview: pa.binary(),
    bytearray: pa.binary(),
    dec.Decimal: pa.decimal128(38,18),
    dt.datetime: pa.timestamp("us"),
    dt.date: pa.date32(),
}

ArrowTabular = pa.Table | pa.RecordBatch


def safe_arrow_tabular(obj) -> ArrowTabular:
    if isinstance(obj, (pa.RecordBatch, pa.Table)):
        return obj
    if isinstance(obj, pl.DataFrame):
        return obj.to_arrow()
    if isinstance(obj, pd.DataFrame):
        return pa.table(obj)
    if isinstance(obj, pd.S):
        return pa.table(obj)

    raise TypeError(f"Cannot convert {type(obj)} to arrow Table or RecordBatch")