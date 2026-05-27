"""SQL function registry with Arrow-native UDF execution.

:class:`FunctionRegistry` maps SQL function names to Arrow compute
kernels.  Built-in functions (UPPER, LOWER, ABS, …) are pre-wired
to ``pyarrow.compute`` kernels.  User-defined functions register a
Python callable that operates on ``pa.Array`` arguments and returns
a ``pa.Array``.

The same kernels auto-register in Spark via ``spark.udf.register``
when :meth:`FunctionRegistry.register_in_spark` is called, wrapping
the Arrow callable in a ``pandas_udf`` so data stays columnar.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    pass

__all__ = ["FunctionRegistry", "FunctionMeta", "BUILTIN_REGISTRY"]

ArrowKernel = Callable[..., pa.Array]


@dataclasses.dataclass(slots=True)
class FunctionMeta:
    name: str
    category: str
    min_args: int = 0
    max_args: int | None = None
    deterministic: bool = True
    special_syntax: str | None = None
    kernel: ArrowKernel | None = None


class FunctionRegistry:
    __slots__ = ("_functions",)

    def __init__(self) -> None:
        self._functions: dict[str, FunctionMeta] = {}

    def register(self, name: str, *, category: str = "udf", min_args: int = 0,
                 max_args: int | None = None, deterministic: bool = True,
                 special_syntax: str | None = None,
                 kernel: ArrowKernel | None = None) -> FunctionMeta:
        meta = FunctionMeta(name=(u := name.upper()), category=category,
                            min_args=min_args, max_args=max_args,
                            deterministic=deterministic, special_syntax=special_syntax,
                            kernel=kernel)
        self._functions[u] = meta
        return meta

    def register_udf(self, name: str, kernel: ArrowKernel, *,
                     min_args: int = 1, max_args: int | None = None) -> FunctionMeta:
        """Register a user-defined Arrow-array function."""
        return self.register(name, category="udf", min_args=min_args,
                             max_args=max_args, kernel=kernel)

    def get(self, name: str) -> FunctionMeta | None: return self._functions.get(name.upper())
    def is_known(self, name: str) -> bool: return name.upper() in self._functions
    def names(self) -> frozenset[str]: return frozenset(self._functions)
    def __contains__(self, name: str) -> bool: return self.is_known(name)
    def __len__(self) -> int: return len(self._functions)

    def apply_arrow(self, name: str, *arrays: pa.Array) -> pa.Array | None:
        """Execute function on Arrow arrays. Returns None if no kernel."""
        meta = self._functions.get(name.upper())
        if meta is None or meta.kernel is None:
            return None
        return meta.kernel(*arrays)

    def apply_table(self, name: str, table: pa.Table,
                    col_names: list[str]) -> pa.Array | None:
        """Convenience: extract named columns and apply kernel."""
        meta = self._functions.get(name.upper())
        if meta is None or meta.kernel is None:
            return None
        arrays = [table.column(c) for c in col_names]
        return meta.kernel(*arrays)

    def register_in_spark(self, spark_session: Any) -> int:
        """Register all kerneled functions as Spark SQL UDFs.

        Uses ``pandas_udf`` so data stays columnar (Arrow-backed).
        Returns the number of functions registered.
        """
        count = 0
        for meta in self._functions.values():
            if meta.kernel is None:
                continue
            try:
                _register_spark_udf(spark_session, meta)
                count += 1
            except Exception:
                pass
        return count

    def copy(self) -> "FunctionRegistry":
        r = FunctionRegistry(); r._functions = dict(self._functions); return r


def _register_spark_udf(spark: Any, meta: FunctionMeta) -> None:
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StringType

    kernel = meta.kernel

    @pandas_udf(StringType())
    def _udf(*cols):
        arrays = [pa.array(c) for c in cols]
        result = kernel(*arrays)
        return result.to_pandas()

    spark.udf.register(meta.name, _udf)


# ---------------------------------------------------------------------------
# Arrow compute kernel mappings for common SQL functions
# ---------------------------------------------------------------------------

def _k1(fn: str) -> ArrowKernel:
    """Single-arg pyarrow.compute kernel."""
    f = getattr(pc, fn)
    return lambda a: f(a)

def _k2(fn: str) -> ArrowKernel:
    """Two-arg pyarrow.compute kernel."""
    f = getattr(pc, fn)
    return lambda a, b: f(a, b)

def _read_files_kernel(paths: pa.Array) -> pa.Array:
    """Read file contents from paths. Tries tabular formats first, raw text fallback."""
    import pathlib
    results = []
    for path_val in paths.to_pylist():
        if path_val is None:
            results.append(None); continue
        try:
            p = pathlib.Path(path_val)
            if not p.exists():
                results.append(None); continue
            from yggdrasil.io.tabular.base import Tabular
            tab = Tabular.from_(str(p), default=None)
            if tab is not None:
                try:
                    results.append(str(tab.read_arrow_table().to_pylist()))
                    continue
                except Exception:
                    pass
            results.append(p.read_bytes().decode("utf-8", errors="replace"))
        except Exception:
            results.append(None)
    return pa.array(results, type=pa.utf8())


def _read_paths_kernel(paths: pa.Array) -> pa.Array:
    """List directory contents for each path."""
    results = []
    for path_val in paths.to_pylist():
        if path_val is None:
            results.append(None); continue
        try:
            import pathlib
            p = pathlib.Path(path_val)
            if p.is_dir():
                results.append(str([str(c) for c in p.iterdir()]))
            else:
                results.append(str(path_val))
        except Exception:
            results.append(None)
    return pa.array(results, type=pa.utf8())


def _compress_kernel(data: pa.Array, codec_name=None) -> pa.Array:
    """Compress binary/string column with a codec."""
    codec_str = codec_name.as_py() if hasattr(codec_name, 'as_py') else str(codec_name) if codec_name else "gzip"
    try:
        from yggdrasil.enums.codec import Codec
        codec = Codec.from_(codec_str)
    except Exception:
        import gzip
        return pa.array([gzip.compress(v.encode() if isinstance(v, str) else v) if v else None
                         for v in data.to_pylist()], type=pa.binary())
    return pa.array([codec.compress_bytes(v.encode() if isinstance(v, str) else v) if v else None
                     for v in data.to_pylist()], type=pa.binary())


def _decompress_kernel(data: pa.Array, codec_name=None) -> pa.Array:
    """Decompress binary column with a codec."""
    codec_str = codec_name.as_py() if hasattr(codec_name, 'as_py') else str(codec_name) if codec_name else "gzip"
    try:
        from yggdrasil.enums.codec import Codec
        codec = Codec.from_(codec_str)
    except Exception:
        import gzip
        return pa.array([gzip.decompress(v) if v else None for v in data.to_pylist()], type=pa.binary())
    return pa.array([codec.decompress_bytes(v) if v else None
                     for v in data.to_pylist()], type=pa.binary())


def _parse_json_kernel(data: pa.Array) -> pa.Array:
    """Parse JSON strings to structured data."""
    import json
    return pa.array([json.loads(v) if v else None for v in data.to_pylist()])


def _to_json_kernel(data: pa.Array) -> pa.Array:
    """Serialize values to JSON strings."""
    import json
    return pa.array([json.dumps(v) if v is not None else None for v in data.to_pylist()], type=pa.utf8())


_ARROW_KERNELS: dict[str, ArrowKernel] = {
    # String
    "UPPER": _k1("utf8_upper"),
    "LOWER": _k1("utf8_lower"),
    "LENGTH": _k1("utf8_length"),
    "CHAR_LENGTH": _k1("utf8_length"),
    "TRIM": _k1("utf8_trim_whitespace"),
    "LTRIM": _k1("utf8_ltrim_whitespace"),
    "RTRIM": _k1("utf8_rtrim_whitespace"),
    "REVERSE": _k1("utf8_reverse"),
    "ASCII": _k1("utf8_length"),
    "INITCAP": _k1("utf8_capitalize"),
    "REPLACE": lambda a, old, new: pc.utf8_replace_substring(a, pattern=old.as_py() if hasattr(old, 'as_py') else str(old), replacement=new.as_py() if hasattr(new, 'as_py') else str(new)),

    # Math
    "ABS": _k1("abs"),
    "CEIL": _k1("ceil"),
    "CEILING": _k1("ceil"),
    "FLOOR": _k1("floor"),
    "ROUND": lambda a, n=None: pc.round(a, ndigits=n.as_py() if n is not None and hasattr(n, 'as_py') else 0 if n is None else int(n)),
    "SQRT": _k1("sqrt"),
    "LN": _k1("ln"),
    "LOG10": _k1("log10"),
    "LOG2": _k1("log2"),
    "EXP": _k1("exp"),
    "SIGN": _k1("sign"),
    "SIGNUM": _k1("sign"),
    "SIN": _k1("sin"),
    "COS": _k1("cos"),
    "TAN": _k1("tan"),
    "ASIN": _k1("asin"),
    "ACOS": _k1("acos"),
    "ATAN": _k1("atan"),
    "ATAN2": _k2("atan2"),
    "POWER": _k2("power"),
    "POW": _k2("power"),

    # Null handling
    "COALESCE": lambda *arrays: _coalesce_arrow(*arrays),
    "IFNULL": lambda a, b: pc.if_else(pc.is_null(a), b, a),
    "NVL": lambda a, b: pc.if_else(pc.is_null(a), b, a),
    "NULLIF": lambda a, b: pc.if_else(pc.equal(a, b), pa.scalar(None, type=a.type), a),

    # Conditional
    "IF": lambda cond, t, f: pc.if_else(cond, t, f),

    # Type
    "ISNULL": lambda a: pc.is_null(a),
    "ISNOTNULL": lambda a: pc.invert(pc.is_null(a)),

    # Date/time extraction
    "YEAR": _k1("year"),
    "MONTH": _k1("month"),
    "DAY": _k1("day"),
    "HOUR": _k1("hour"),
    "MINUTE": _k1("minute"),
    "SECOND": _k1("second"),
    "DAYOFWEEK": _k1("day_of_week"),
    "DAYOFYEAR": lambda a: pc.add(pc.day_of_year(a), 0),
    "QUARTER": _k1("quarter"),
    "WEEKOFYEAR": _k1("iso_calendar"),

    # Aggregates (these work on arrays and return scalars)
    "SUM": _k1("sum"),
    "MIN": _k1("min"),
    "MAX": _k1("max"),
    "COUNT": lambda a: pa.scalar(len(a) - a.null_count, type=pa.int64()),
    "AVG": _k1("mean"),
    "MEAN": _k1("mean"),

    # IO / path-aware UDFs
    "READ_FILES": _read_files_kernel,
    "READ_PATHS": _read_paths_kernel,
    "COMPRESS": _compress_kernel,
    "DECOMPRESS": _decompress_kernel,
    "PARSE_JSON": _parse_json_kernel,
    "TO_JSON": _to_json_kernel,
    "BASE64_ENCODE": lambda a: pc.binary_join_element_wise(a, pa.scalar("")),
    "BASE64_DECODE": lambda a: a,
}


def _coalesce_arrow(*arrays: pa.Array) -> pa.Array:
    result = arrays[0]
    for arr in arrays[1:]:
        if isinstance(arr, pa.Scalar):
            arr = pa.array([arr.as_py()] * len(result), type=result.type)
        result = pc.if_else(pc.is_null(result), arr, result)
    return result


# ---------------------------------------------------------------------------
# Build the registry
# ---------------------------------------------------------------------------

def _build() -> FunctionRegistry:
    r = FunctionRegistry()

    def _b(cat: str, fns: dict[str, tuple[int, int | None]], **kw: Any) -> None:
        for n, (mn, mx) in fns.items():
            kernel = _ARROW_KERNELS.get(n)
            r.register(n, category=cat, min_args=mn, max_args=mx,
                       kernel=kernel, **kw)

    _b("aggregate", {"COUNT": (0, 1), "SUM": (1, 1), "AVG": (1, 1), "MIN": (1, 1),
        "MAX": (1, 1), "MEAN": (1, 1), "STDDEV": (1, 1), "STDDEV_POP": (1, 1),
        "STDDEV_SAMP": (1, 1), "VARIANCE": (1, 1), "VAR_POP": (1, 1), "VAR_SAMP": (1, 1),
        "APPROX_COUNT_DISTINCT": (1, 1), "PERCENTILE": (2, 3), "PERCENTILE_APPROX": (2, 3),
        "FIRST": (1, 2), "LAST": (1, 2), "ANY_VALUE": (1, 1), "COUNT_IF": (1, 1),
        "BOOL_AND": (1, 1), "BOOL_OR": (1, 1), "SOME": (1, 1), "EVERY": (1, 1),
        "COLLECT_LIST": (1, 1), "COLLECT_SET": (1, 1), "CORR": (2, 2),
        "COVAR_POP": (2, 2), "COVAR_SAMP": (2, 2), "KURTOSIS": (1, 1),
        "SKEWNESS": (1, 1), "BIT_AND": (1, 1), "BIT_OR": (1, 1), "BIT_XOR": (1, 1)})

    _b("window", {"ROW_NUMBER": (0, 0), "RANK": (0, 0), "DENSE_RANK": (0, 0),
        "NTILE": (1, 1), "CUME_DIST": (0, 0), "PERCENT_RANK": (0, 0),
        "LAG": (1, 3), "LEAD": (1, 3), "FIRST_VALUE": (1, 2),
        "LAST_VALUE": (1, 2), "NTH_VALUE": (2, 2)})

    _b("datetime", {"CURRENT_DATE": (0, 0), "CURRENT_TIMESTAMP": (0, 0), "NOW": (0, 0),
        "DATE_TRUNC": (2, 2), "DATE_ADD": (2, 2), "DATE_SUB": (2, 2),
        "DATEDIFF": (2, 3), "DATE_FORMAT": (2, 2), "DATEADD": (2, 3), "DATESUB": (2, 3),
        "TO_DATE": (1, 2), "TO_TIMESTAMP": (1, 2), "TO_UNIX_TIMESTAMP": (1, 2),
        "UNIX_TIMESTAMP": (0, 2), "FROM_UNIXTIME": (1, 2),
        "FROM_UTC_TIMESTAMP": (2, 2), "TO_UTC_TIMESTAMP": (2, 2),
        "MONTHS_BETWEEN": (2, 3), "ADD_MONTHS": (2, 2), "LAST_DAY": (1, 1),
        "NEXT_DAY": (2, 2), "TRUNC": (1, 2), "YEAR": (1, 1), "MONTH": (1, 1),
        "DAY": (1, 1), "DAYOFWEEK": (1, 1), "DAYOFYEAR": (1, 1), "HOUR": (1, 1),
        "MINUTE": (1, 1), "SECOND": (1, 1), "WEEKOFYEAR": (1, 1), "QUARTER": (1, 1),
        "MAKE_DATE": (3, 3), "MAKE_TIMESTAMP": (6, 7), "MAKE_INTERVAL": (0, 7),
        "DATE_PART": (2, 2), "DATEPART": (2, 2), "EXTRACT": (2, 2),
        "TIMESTAMP_SECONDS": (1, 1), "TIMESTAMP_MILLIS": (1, 1),
        "TIMESTAMP_MICROS": (1, 1), "DATE_FROM_UNIX_DATE": (1, 1), "UNIX_DATE": (1, 1)})

    _b("string", {"CONCAT": (1, None), "CONCAT_WS": (2, None), "SUBSTRING": (2, 3),
        "SUBSTR": (2, 3), "TRIM": (1, 1), "LTRIM": (1, 2), "RTRIM": (1, 2),
        "UPPER": (1, 1), "LOWER": (1, 1), "LENGTH": (1, 1), "CHAR_LENGTH": (1, 1),
        "REPLACE": (2, 3), "REGEXP_REPLACE": (2, 4), "REGEXP_EXTRACT": (2, 3),
        "REGEXP_EXTRACT_ALL": (2, 3), "REGEXP_LIKE": (2, 3), "SPLIT": (2, 3),
        "LPAD": (2, 3), "RPAD": (2, 3), "INITCAP": (1, 1), "REVERSE": (1, 1),
        "REPEAT": (2, 2), "TRANSLATE": (3, 3), "BASE64": (1, 1), "UNBASE64": (1, 1),
        "DECODE": (2, None), "ENCODE": (2, 2), "FORMAT_STRING": (1, None),
        "FORMAT_NUMBER": (2, 2), "INSTR": (2, 2), "LOCATE": (2, 3),
        "LEFT": (2, 2), "RIGHT": (2, 2), "OVERLAY": (3, 4), "POSITION": (2, 2),
        "SOUNDEX": (1, 1), "LEVENSHTEIN": (2, 3), "ASCII": (1, 1), "CHR": (1, 1),
        "SPACE": (1, 1), "PRINTF": (1, None)})

    _b("null", {"COALESCE": (1, None), "NVL": (2, 2), "NVL2": (3, 3),
        "IFNULL": (2, 2), "NULLIF": (2, 2), "ISNULL": (1, 1), "ISNOTNULL": (1, 1)})

    _b("math", {"ABS": (1, 1), "CEIL": (1, 1), "CEILING": (1, 1), "FLOOR": (1, 1),
        "ROUND": (1, 2), "BROUND": (1, 2), "MOD": (2, 2), "POWER": (2, 2),
        "POW": (2, 2), "SQRT": (1, 1), "CBRT": (1, 1), "LOG": (1, 2), "LOG2": (1, 1),
        "LOG10": (1, 1), "LN": (1, 1), "EXP": (1, 1), "SIGN": (1, 1), "SIGNUM": (1, 1),
        "GREATEST": (1, None), "LEAST": (1, None), "PI": (0, 0), "E": (0, 0),
        "CONV": (3, 3), "HEX": (1, 1), "UNHEX": (1, 1), "BIN": (1, 1),
        "DEGREES": (1, 1), "RADIANS": (1, 1), "SIN": (1, 1), "COS": (1, 1),
        "TAN": (1, 1), "ASIN": (1, 1), "ACOS": (1, 1), "ATAN": (1, 1), "ATAN2": (2, 2),
        "SINH": (1, 1), "COSH": (1, 1), "TANH": (1, 1), "FACTORIAL": (1, 1),
        "SHIFTLEFT": (2, 2), "SHIFTRIGHT": (2, 2), "WIDTH_BUCKET": (4, 4),
        "PMOD": (2, 2), "RINT": (1, 1)})

    _b("collection", {"SIZE": (1, 1), "FLATTEN": (1, 1), "ARRAY_CONTAINS": (2, 2),
        "ARRAY_DISTINCT": (1, 1), "ARRAY_EXCEPT": (2, 2), "ARRAY_INTERSECT": (2, 2),
        "ARRAY_JOIN": (2, 3), "ARRAY_MAX": (1, 1), "ARRAY_MIN": (1, 1),
        "ARRAY_POSITION": (2, 2), "ARRAY_REMOVE": (2, 2), "ARRAY_REPEAT": (2, 2),
        "ARRAY_SORT": (1, 2), "ARRAY_UNION": (2, 2), "ARRAYS_OVERLAP": (2, 2),
        "ARRAYS_ZIP": (1, None), "ELEMENT_AT": (2, 2), "SLICE": (3, 3),
        "SORT_ARRAY": (1, 2), "SEQUENCE": (2, 3), "MAP_KEYS": (1, 1),
        "MAP_VALUES": (1, 1), "MAP_ENTRIES": (1, 1), "MAP_FROM_ENTRIES": (1, 1),
        "MAP_FROM_ARRAYS": (2, 2), "MAP_CONCAT": (1, None), "MAP_FILTER": (2, 2),
        "MAP_ZIP_WITH": (3, 3), "TRANSFORM_KEYS": (2, 2), "TRANSFORM_VALUES": (2, 2),
        "NAMED_STRUCT": (2, None), "STRUCT": (1, None), "ARRAY": (0, None),
        "MAP": (0, None), "CARDINALITY": (1, 1), "ARRAY_COMPACT": (1, 1),
        "ARRAY_APPEND": (2, 2), "ARRAY_PREPEND": (2, 2), "ARRAY_INSERT": (3, 3)})

    _b("generator", {"EXPLODE": (1, 1), "POSEXPLODE": (1, 1), "INLINE": (1, 1),
        "STACK": (2, None), "EXPLODE_OUTER": (1, 1), "POSEXPLODE_OUTER": (1, 1),
        "INLINE_OUTER": (1, 1)})
    _b("type", {"CAST": (1, 1), "TRY_CAST": (1, 2), "TYPEOF": (1, 1)})
    _b("json", {"FROM_JSON": (2, 3), "TO_JSON": (1, 2), "SCHEMA_OF_JSON": (1, 2),
        "GET_JSON_OBJECT": (2, 2), "JSON_TUPLE": (2, None), "JSON_ARRAY_LENGTH": (1, 1)})
    _b("higher_order", {"TRANSFORM": (2, 2), "FILTER": (2, 2), "AGGREGATE": (3, 4),
        "EXISTS": (2, 2), "FORALL": (2, 2), "ZIP_WITH": (3, 3), "REDUCE": (3, 4)})
    _b("hash", {"HASH": (1, None), "XXHASH64": (1, None), "MD5": (1, 1),
        "SHA1": (1, 1), "SHA": (1, 1), "SHA2": (2, 2), "CRC32": (1, 1)})
    _b("conditional", {"IF": (3, 3), "IIF": (3, 3)})
    _b("misc", {"UUID": (0, 0), "INPUT_FILE_NAME": (0, 0),
        "MONOTONICALLY_INCREASING_ID": (0, 0), "SPARK_PARTITION_ID": (0, 0),
        "CURRENT_USER": (0, 0), "CURRENT_CATALOG": (0, 0), "CURRENT_DATABASE": (0, 0),
        "CURRENT_SCHEMA": (0, 0), "VERSION": (0, 0), "ASSERT_TRUE": (1, 2),
        "RAISE_ERROR": (1, 1)}, deterministic=False)
    r.register("INTERVAL", category="datetime", min_args=1, max_args=2,
               special_syntax="INTERVAL 'value' unit")

    # IO / codec / serialization UDFs
    _b("io", {"READ_FILES": (1, 1), "READ_PATHS": (1, 1),
        "COMPRESS": (1, 2), "DECOMPRESS": (1, 2),
        "PARSE_JSON": (1, 1)})
    return r


BUILTIN_REGISTRY: FunctionRegistry = _build()
