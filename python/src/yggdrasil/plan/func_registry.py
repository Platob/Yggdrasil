"""SQL function registry — canonical catalog of known functions.

Provides a centralized :class:`FunctionRegistry` that tracks function
metadata (name, category, min/max args, deterministic flag). Used by:

- The SQL parser to recognize function names that might otherwise
  be mis-classified as column references or keywords.
- The SQL emitter to apply dialect-specific rendering (e.g., EXTRACT
  uses ``EXTRACT(field FROM source)`` syntax).
- Future optimization passes to identify pure/impure functions.

The registry is seeded with all standard SQL and Databricks built-in
functions. Users can register additional UDFs via :meth:`register`.
"""

from __future__ import annotations

import dataclasses
from typing import Any


__all__ = ["FunctionRegistry", "FunctionMeta", "BUILTIN_REGISTRY"]


@dataclasses.dataclass(slots=True, frozen=True)
class FunctionMeta:
    """Metadata about a registered SQL function."""
    name: str
    category: str
    min_args: int = 0
    max_args: int | None = None
    deterministic: bool = True
    special_syntax: str | None = None


class FunctionRegistry:
    """Mutable registry of known SQL functions.

    ``get(name)`` returns :class:`FunctionMeta` or ``None``.
    ``is_known(name)`` is a fast membership check.
    ``register(name, ...)`` adds a UDF at runtime.
    ``names()`` returns the full set of registered names.
    """

    __slots__ = ("_functions",)

    def __init__(self) -> None:
        self._functions: dict[str, FunctionMeta] = {}

    def register(
        self,
        name: str,
        *,
        category: str = "udf",
        min_args: int = 0,
        max_args: int | None = None,
        deterministic: bool = True,
        special_syntax: str | None = None,
    ) -> FunctionMeta:
        upper = name.upper()
        meta = FunctionMeta(
            name=upper,
            category=category,
            min_args=min_args,
            max_args=max_args,
            deterministic=deterministic,
            special_syntax=special_syntax,
        )
        self._functions[upper] = meta
        return meta

    def get(self, name: str) -> FunctionMeta | None:
        return self._functions.get(name.upper())

    def is_known(self, name: str) -> bool:
        return name.upper() in self._functions

    def names(self) -> frozenset[str]:
        return frozenset(self._functions)

    def __contains__(self, name: str) -> bool:
        return self.is_known(name)

    def __len__(self) -> int:
        return len(self._functions)

    def copy(self) -> "FunctionRegistry":
        r = FunctionRegistry()
        r._functions = dict(self._functions)
        return r


def _build_builtin_registry() -> FunctionRegistry:
    """Seed the registry with all standard SQL + Databricks functions."""
    r = FunctionRegistry()

    def _batch(category: str, names: dict[str, tuple[int, int | None]],
               **kwargs: Any) -> None:
        for name, (mn, mx) in names.items():
            r.register(name, category=category, min_args=mn, max_args=mx,
                       **kwargs)

    # ------------------------------------------------------------------
    # Aggregate functions
    # ------------------------------------------------------------------
    _batch("aggregate", {
        "COUNT": (0, 1), "SUM": (1, 1), "AVG": (1, 1),
        "MIN": (1, 1), "MAX": (1, 1), "MEAN": (1, 1),
        "STDDEV": (1, 1), "STDDEV_POP": (1, 1), "STDDEV_SAMP": (1, 1),
        "VARIANCE": (1, 1), "VAR_POP": (1, 1), "VAR_SAMP": (1, 1),
        "APPROX_COUNT_DISTINCT": (1, 1), "PERCENTILE": (2, 3),
        "PERCENTILE_APPROX": (2, 3),
        "FIRST": (1, 2), "LAST": (1, 2), "ANY_VALUE": (1, 1),
        "COUNT_IF": (1, 1), "BOOL_AND": (1, 1), "BOOL_OR": (1, 1),
        "SOME": (1, 1), "EVERY": (1, 1),
        "COLLECT_LIST": (1, 1), "COLLECT_SET": (1, 1),
        "CORR": (2, 2), "COVAR_POP": (2, 2), "COVAR_SAMP": (2, 2),
        "REGR_AVGX": (2, 2), "REGR_AVGY": (2, 2),
        "KURTOSIS": (1, 1), "SKEWNESS": (1, 1),
        "BIT_AND": (1, 1), "BIT_OR": (1, 1), "BIT_XOR": (1, 1),
    })

    # ------------------------------------------------------------------
    # Window functions
    # ------------------------------------------------------------------
    _batch("window", {
        "ROW_NUMBER": (0, 0), "RANK": (0, 0), "DENSE_RANK": (0, 0),
        "NTILE": (1, 1), "CUME_DIST": (0, 0), "PERCENT_RANK": (0, 0),
        "LAG": (1, 3), "LEAD": (1, 3),
        "FIRST_VALUE": (1, 2), "LAST_VALUE": (1, 2), "NTH_VALUE": (2, 2),
    })

    # ------------------------------------------------------------------
    # Date / time functions
    # ------------------------------------------------------------------
    _batch("datetime", {
        "CURRENT_DATE": (0, 0), "CURRENT_TIMESTAMP": (0, 0), "NOW": (0, 0),
        "DATE_TRUNC": (2, 2), "DATE_ADD": (2, 2), "DATE_SUB": (2, 2),
        "DATEDIFF": (2, 3), "DATE_FORMAT": (2, 2),
        "DATEADD": (2, 3), "DATESUB": (2, 3),
        "TO_DATE": (1, 2), "TO_TIMESTAMP": (1, 2),
        "TO_UNIX_TIMESTAMP": (1, 2), "UNIX_TIMESTAMP": (0, 2),
        "FROM_UNIXTIME": (1, 2),
        "FROM_UTC_TIMESTAMP": (2, 2), "TO_UTC_TIMESTAMP": (2, 2),
        "MONTHS_BETWEEN": (2, 3), "ADD_MONTHS": (2, 2),
        "LAST_DAY": (1, 1), "NEXT_DAY": (2, 2),
        "TRUNC": (1, 2),
        "YEAR": (1, 1), "MONTH": (1, 1), "DAY": (1, 1),
        "DAYOFWEEK": (1, 1), "DAYOFYEAR": (1, 1),
        "HOUR": (1, 1), "MINUTE": (1, 1), "SECOND": (1, 1),
        "WEEKOFYEAR": (1, 1), "QUARTER": (1, 1),
        "MAKE_DATE": (3, 3), "MAKE_TIMESTAMP": (6, 7),
        "MAKE_INTERVAL": (0, 7),
        "DATE_PART": (2, 2), "DATEPART": (2, 2),
        "EXTRACT": (2, 2),
        "TIMESTAMP_SECONDS": (1, 1), "TIMESTAMP_MILLIS": (1, 1),
        "TIMESTAMP_MICROS": (1, 1),
        "DATE_FROM_UNIX_DATE": (1, 1), "UNIX_DATE": (1, 1),
    }, deterministic=True)

    # ------------------------------------------------------------------
    # String functions
    # ------------------------------------------------------------------
    _batch("string", {
        "CONCAT": (1, None), "CONCAT_WS": (2, None),
        "SUBSTRING": (2, 3), "SUBSTR": (2, 3),
        "TRIM": (1, 1), "LTRIM": (1, 2), "RTRIM": (1, 2),
        "UPPER": (1, 1), "LOWER": (1, 1), "LENGTH": (1, 1),
        "CHAR_LENGTH": (1, 1), "CHARACTER_LENGTH": (1, 1),
        "REPLACE": (2, 3),
        "REGEXP_REPLACE": (2, 4), "REGEXP_EXTRACT": (2, 3),
        "REGEXP_EXTRACT_ALL": (2, 3), "REGEXP_LIKE": (2, 3),
        "SPLIT": (2, 3), "LPAD": (2, 3), "RPAD": (2, 3),
        "INITCAP": (1, 1), "REVERSE": (1, 1), "REPEAT": (2, 2),
        "TRANSLATE": (3, 3),
        "BASE64": (1, 1), "UNBASE64": (1, 1),
        "DECODE": (2, None), "ENCODE": (2, 2),
        "FORMAT_STRING": (1, None), "FORMAT_NUMBER": (2, 2),
        "INSTR": (2, 2), "LOCATE": (2, 3),
        "LEFT": (2, 2), "RIGHT": (2, 2),
        "OVERLAY": (3, 4), "POSITION": (2, 2),
        "BTRIM": (1, 2), "SOUNDEX": (1, 1),
        "LEVENSHTEIN": (2, 3), "ASCII": (1, 1), "CHR": (1, 1),
        "CHAR": (1, 1), "BIT_LENGTH": (1, 1), "OCTET_LENGTH": (1, 1),
        "SPACE": (1, 1), "PRINTF": (1, None),
    })

    # ------------------------------------------------------------------
    # Null handling
    # ------------------------------------------------------------------
    _batch("null", {
        "COALESCE": (1, None), "NVL": (2, 2), "NVL2": (3, 3),
        "IFNULL": (2, 2), "NULLIF": (2, 2),
        "ISNULL": (1, 1), "ISNOTNULL": (1, 1),
    })

    # ------------------------------------------------------------------
    # Math functions
    # ------------------------------------------------------------------
    _batch("math", {
        "ABS": (1, 1), "CEIL": (1, 1), "CEILING": (1, 1),
        "FLOOR": (1, 1), "ROUND": (1, 2), "BROUND": (1, 2),
        "MOD": (2, 2), "POWER": (2, 2), "POW": (2, 2),
        "SQRT": (1, 1), "CBRT": (1, 1),
        "LOG": (1, 2), "LOG2": (1, 1), "LOG10": (1, 1), "LN": (1, 1),
        "EXP": (1, 1), "EXPM1": (1, 1),
        "SIGN": (1, 1), "SIGNUM": (1, 1),
        "GREATEST": (1, None), "LEAST": (1, None),
        "PI": (0, 0), "E": (0, 0),
        "CONV": (3, 3), "HEX": (1, 1), "UNHEX": (1, 1), "BIN": (1, 1),
        "DEGREES": (1, 1), "RADIANS": (1, 1),
        "SIN": (1, 1), "COS": (1, 1), "TAN": (1, 1),
        "ASIN": (1, 1), "ACOS": (1, 1), "ATAN": (1, 1), "ATAN2": (2, 2),
        "SINH": (1, 1), "COSH": (1, 1), "TANH": (1, 1),
        "FACTORIAL": (1, 1), "SHIFTLEFT": (2, 2), "SHIFTRIGHT": (2, 2),
        "WIDTH_BUCKET": (4, 4), "POSITIVE": (1, 1), "NEGATIVE": (1, 1),
        "PMOD": (2, 2), "RINT": (1, 1),
    })

    # ------------------------------------------------------------------
    # Array / map / struct
    # ------------------------------------------------------------------
    _batch("collection", {
        "SIZE": (1, 1), "FLATTEN": (1, 1),
        "ARRAY_CONTAINS": (2, 2), "ARRAY_DISTINCT": (1, 1),
        "ARRAY_EXCEPT": (2, 2), "ARRAY_INTERSECT": (2, 2),
        "ARRAY_JOIN": (2, 3), "ARRAY_MAX": (1, 1), "ARRAY_MIN": (1, 1),
        "ARRAY_POSITION": (2, 2), "ARRAY_REMOVE": (2, 2),
        "ARRAY_REPEAT": (2, 2), "ARRAY_SORT": (1, 2),
        "ARRAY_UNION": (2, 2), "ARRAYS_OVERLAP": (2, 2),
        "ARRAYS_ZIP": (1, None), "ELEMENT_AT": (2, 2),
        "SLICE": (3, 3), "SORT_ARRAY": (1, 2), "SEQUENCE": (2, 3),
        "MAP_KEYS": (1, 1), "MAP_VALUES": (1, 1),
        "MAP_ENTRIES": (1, 1), "MAP_FROM_ENTRIES": (1, 1),
        "MAP_FROM_ARRAYS": (2, 2), "MAP_CONCAT": (1, None),
        "MAP_FILTER": (2, 2), "MAP_ZIP_WITH": (3, 3),
        "TRANSFORM_KEYS": (2, 2), "TRANSFORM_VALUES": (2, 2),
        "NAMED_STRUCT": (2, None), "STRUCT": (1, None),
        "ARRAY": (0, None), "MAP": (0, None),
        "CARDINALITY": (1, 1), "ARRAY_COMPACT": (1, 1),
        "ARRAY_APPEND": (2, 2), "ARRAY_PREPEND": (2, 2),
        "ARRAY_INSERT": (3, 3),
    })

    # ------------------------------------------------------------------
    # Explode / generator
    # ------------------------------------------------------------------
    _batch("generator", {
        "EXPLODE": (1, 1), "POSEXPLODE": (1, 1),
        "INLINE": (1, 1), "STACK": (2, None),
        "EXPLODE_OUTER": (1, 1), "POSEXPLODE_OUTER": (1, 1),
        "INLINE_OUTER": (1, 1),
    })

    # ------------------------------------------------------------------
    # Type / cast
    # ------------------------------------------------------------------
    _batch("type", {
        "CAST": (1, 1), "TRY_CAST": (1, 2), "TYPEOF": (1, 1),
        "BOOLEAN": (1, 1), "TINYINT": (1, 1), "SMALLINT": (1, 1),
        "INT": (1, 1), "BIGINT": (1, 1), "FLOAT": (1, 1), "DOUBLE": (1, 1),
        "STRING": (1, 1), "BINARY": (1, 1), "DECIMAL": (1, 3),
    })

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    _batch("json", {
        "FROM_JSON": (2, 3), "TO_JSON": (1, 2),
        "SCHEMA_OF_JSON": (1, 2), "GET_JSON_OBJECT": (2, 2),
        "JSON_TUPLE": (2, None), "JSON_ARRAY_LENGTH": (1, 1),
    })

    # ------------------------------------------------------------------
    # Higher-order (lambda)
    # ------------------------------------------------------------------
    _batch("higher_order", {
        "TRANSFORM": (2, 2), "FILTER": (2, 2),
        "AGGREGATE": (3, 4), "EXISTS": (2, 2), "FORALL": (2, 2),
        "ZIP_WITH": (3, 3), "REDUCE": (3, 4),
    })

    # ------------------------------------------------------------------
    # Hash / crypto
    # ------------------------------------------------------------------
    _batch("hash", {
        "HASH": (1, None), "XXHASH64": (1, None),
        "MD5": (1, 1), "SHA1": (1, 1), "SHA": (1, 1), "SHA2": (2, 2),
        "CRC32": (1, 1), "FNV_HASH": (1, None),
    })

    # ------------------------------------------------------------------
    # Conditional
    # ------------------------------------------------------------------
    _batch("conditional", {
        "IF": (3, 3), "IIF": (3, 3),
    })

    # ------------------------------------------------------------------
    # Misc / system
    # ------------------------------------------------------------------
    _batch("misc", {
        "UUID": (0, 0), "INPUT_FILE_NAME": (0, 0),
        "MONOTONICALLY_INCREASING_ID": (0, 0),
        "SPARK_PARTITION_ID": (0, 0),
        "CURRENT_USER": (0, 0), "CURRENT_CATALOG": (0, 0),
        "CURRENT_DATABASE": (0, 0), "CURRENT_SCHEMA": (0, 0),
        "VERSION": (0, 0),
        "ASSERT_TRUE": (1, 2), "RAISE_ERROR": (1, 1),
        "TYPEOF": (1, 1),
    }, deterministic=False)

    # Special syntax functions
    r.register("EXTRACT", category="datetime", min_args=2, max_args=2,
               special_syntax="EXTRACT(field FROM source)")
    r.register("INTERVAL", category="datetime", min_args=1, max_args=2,
               special_syntax="INTERVAL 'value' unit")

    return r


BUILTIN_REGISTRY: FunctionRegistry = _build_builtin_registry()
