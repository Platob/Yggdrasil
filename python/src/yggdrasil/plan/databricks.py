"""Databricks-specific SQL parser extensions.

Extends :class:`SQLQueryParser` with Databricks SQL features:

- ``LATERAL VIEW [OUTER] EXPLODE / POSEXPLODE / INLINE / STACK``
- Date/time functions: ``DATE_TRUNC``, ``DATE_ADD``, ``DATE_SUB``,
  ``DATEDIFF``, ``DATE_FORMAT``, ``TO_DATE``, ``TO_TIMESTAMP``,
  ``DATEADD``, ``CURRENT_DATE``, ``CURRENT_TIMESTAMP``, ``NOW``
- String functions: ``CONCAT``, ``SUBSTRING``, ``TRIM``, ``UPPER``,
  ``LOWER``, ``REPLACE``, ``REGEXP_REPLACE``, ``REGEXP_EXTRACT``,
  ``SPLIT``, ``LPAD``, ``RPAD``
- Null functions: ``COALESCE``, ``NVL``, ``NVL2``, ``IFNULL``, ``NULLIF``
- Collection: ``COLLECT_LIST``, ``COLLECT_SET``, ``SIZE``, ``FLATTEN``,
  ``ARRAY_CONTAINS``, ``ELEMENT_AT``, ``TRANSFORM``, ``FILTER``
- Aggregate: ``COUNT``, ``SUM``, ``AVG``, ``MIN``, ``MAX``,
  ``APPROX_COUNT_DISTINCT``, ``PERCENTILE_APPROX``
- Window: ``ROW_NUMBER``, ``RANK``, ``DENSE_RANK``, ``NTILE``,
  ``LAG``, ``LEAD``, ``FIRST_VALUE``, ``LAST_VALUE``
- Math: ``ABS``, ``CEIL``, ``FLOOR``, ``ROUND``, ``MOD``, ``POWER``,
  ``SQRT``, ``LOG``, ``LN``, ``EXP``
- Type: ``TRY_CAST``
- Misc: ``EXPLODE``, ``POSEXPLODE``, ``INLINE``, ``STACK``,
  ``NAMED_STRUCT``, ``MAP_KEYS``, ``MAP_VALUES``, ``FROM_JSON``,
  ``TO_JSON``, ``SCHEMA_OF_JSON``
"""

from __future__ import annotations

from yggdrasil.enums.dialect import Dialect
from yggdrasil.execution.expr.nodes import Expression, Literal

from .func_registry import BUILTIN_REGISTRY
from .sql_parser import SQLQueryParser, _DIALECT_PARSERS


# All Databricks built-in functions recognized as keywords in expression
# context — the parser sees ``ident + lparen`` and routes to function
# call parsing, but for reserved-word functions that the tokenizer
# might mis-classify, this set lets the parser handle them.
_DATABRICKS_FUNCTIONS = frozenset({
    # Date/time
    "DATE_TRUNC", "DATE_ADD", "DATE_SUB", "DATEDIFF", "DATE_FORMAT",
    "DATEADD", "DATESUB", "TO_DATE", "TO_TIMESTAMP", "TO_UNIX_TIMESTAMP",
    "UNIX_TIMESTAMP", "FROM_UNIXTIME", "FROM_UTC_TIMESTAMP",
    "TO_UTC_TIMESTAMP", "MONTHS_BETWEEN", "ADD_MONTHS", "LAST_DAY",
    "NEXT_DAY", "TRUNC", "YEAR", "MONTH", "DAY", "DAYOFWEEK",
    "DAYOFYEAR", "HOUR", "MINUTE", "SECOND", "WEEKOFYEAR", "QUARTER",
    "MAKE_DATE", "MAKE_TIMESTAMP",
    # String
    "CONCAT", "CONCAT_WS", "SUBSTRING", "SUBSTR", "TRIM", "LTRIM",
    "RTRIM", "UPPER", "LOWER", "LENGTH", "REPLACE", "REGEXP_REPLACE",
    "REGEXP_EXTRACT", "REGEXP_EXTRACT_ALL", "SPLIT", "LPAD", "RPAD",
    "INITCAP", "REVERSE", "REPEAT", "TRANSLATE", "BASE64",
    "UNBASE64", "DECODE", "ENCODE", "FORMAT_STRING", "FORMAT_NUMBER",
    "INSTR", "LOCATE", "LEFT", "RIGHT",
    # Null
    "COALESCE", "NVL", "NVL2", "IFNULL", "NULLIF",
    "ISNULL", "ISNOTNULL",
    # Collection / array / map
    "COLLECT_LIST", "COLLECT_SET", "SIZE", "FLATTEN",
    "ARRAY_CONTAINS", "ARRAY_DISTINCT", "ARRAY_EXCEPT", "ARRAY_INTERSECT",
    "ARRAY_JOIN", "ARRAY_MAX", "ARRAY_MIN", "ARRAY_POSITION",
    "ARRAY_REMOVE", "ARRAY_REPEAT", "ARRAY_SORT", "ARRAY_UNION",
    "ARRAYS_OVERLAP", "ARRAYS_ZIP", "ELEMENT_AT", "SLICE",
    "SORT_ARRAY", "SEQUENCE",
    "MAP_KEYS", "MAP_VALUES", "MAP_ENTRIES", "MAP_FROM_ENTRIES",
    "MAP_FROM_ARRAYS", "MAP_CONCAT", "MAP_FILTER", "MAP_ZIP_WITH",
    "TRANSFORM_KEYS", "TRANSFORM_VALUES",
    "NAMED_STRUCT", "STRUCT",
    # Explode family
    "EXPLODE", "POSEXPLODE", "INLINE", "STACK",
    "EXPLODE_OUTER", "POSEXPLODE_OUTER", "INLINE_OUTER",
    # Aggregate
    "COUNT", "SUM", "AVG", "MIN", "MAX", "MEAN",
    "STDDEV", "STDDEV_POP", "STDDEV_SAMP",
    "VARIANCE", "VAR_POP", "VAR_SAMP",
    "APPROX_COUNT_DISTINCT", "PERCENTILE", "PERCENTILE_APPROX",
    "FIRST", "LAST", "ANY_VALUE",
    "COUNT_IF", "BOOL_AND", "BOOL_OR", "SOME", "EVERY",
    # Window
    "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "CUME_DIST",
    "PERCENT_RANK", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE",
    "NTH_VALUE",
    # Math
    "ABS", "CEIL", "CEILING", "FLOOR", "ROUND", "BROUND",
    "MOD", "POWER", "POW", "SQRT", "CBRT", "LOG", "LOG2", "LOG10",
    "LN", "EXP", "SIGN", "SIGNUM", "RAND", "RANDN",
    "GREATEST", "LEAST",
    "PI", "E",
    "CONV", "HEX", "UNHEX", "BIN",
    "DEGREES", "RADIANS", "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN",
    "ATAN2", "SINH", "COSH", "TANH",
    "FACTORIAL", "SHIFTLEFT", "SHIFTRIGHT",
    # Type / cast
    "TRY_CAST", "TYPEOF",
    # JSON
    "FROM_JSON", "TO_JSON", "SCHEMA_OF_JSON", "GET_JSON_OBJECT",
    "JSON_TUPLE",
    # Higher-order (lambdas)
    "TRANSFORM", "FILTER", "AGGREGATE", "EXISTS", "FORALL",
    "ZIP_WITH", "REDUCE",
    # Misc
    "IF", "IIF", "DECODE", "HASH", "MD5", "SHA1", "SHA2",
    "CRC32", "INPUT_FILE_NAME", "MONOTONICALLY_INCREASING_ID",
    "SPARK_PARTITION_ID", "UUID",
})


class DatabricksSQLParser(SQLQueryParser):
    """Databricks SQL dialect parser.

    Extends the base parser with Databricks-specific function
    recognition and LATERAL VIEW syntax. All standard Databricks
    functions are recognized as function calls (not columns) even
    without backtick quoting.
    """

    def _parse_primary(self) -> Expression:
        t = self.cur
        # TRY_CAST has CAST-like syntax
        if t.kind == "ident" and t.upper == "TRY_CAST":
            return self._parse_try_cast()
        # IF(cond, true_val, false_val) — Databricks shorthand
        if t.kind == "ident" and t.upper == "IF" and self._peek(1).kind == "lparen":
            return self._parse_function_call()
        # Databricks functions: check the static set AND the registry
        if (t.kind == "ident" and self._peek(1).kind == "lparen"
                and (t.upper in _DATABRICKS_FUNCTIONS
                     or BUILTIN_REGISTRY.is_known(t.upper))):
            return self._parse_function_call()
        return super()._parse_primary()

    def _parse_try_cast(self) -> Expression:
        self._eat()  # TRY_CAST
        self._expect_kind("lparen")
        inner = self._parse_or()
        self._expect_kw("AS")
        type_name = self._parse_type_head()
        self._expect_kind("rparen")
        from yggdrasil.execution.expr.nodes import FunctionCall
        return FunctionCall(name="TRY_CAST", args=(inner, Literal(value=type_name)))


# Register Databricks parser for the DATABRICKS dialect
_DIALECT_PARSERS[Dialect.DATABRICKS] = DatabricksSQLParser
