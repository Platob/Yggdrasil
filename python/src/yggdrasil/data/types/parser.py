from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

from .id import DataTypeId

__all__ = [
    "DataTypeMetadata",
    "ParsedDataType",
    "Token",
    "parse_data_type",
]


# ---------------------------------------------------------------------
# Byte-size defaults for named integer / float aliases.
#
# Returns None for unknown names so callers can distinguish "explicitly
# 8 bytes" from "no declared size". The earlier version fell back to 8
# unconditionally, which caused plain `int` / `float` tokens to emit a
# byte_size the user never specified — indistinguishable from `int64`.
# ---------------------------------------------------------------------

_INTEGER_BYTE_SIZES: dict[str, int] = {
    "byte": 1, "tinyint": 1, "i8": 1, "u8": 1, "int8": 1, "uint8": 1, "utinyint": 1,
    "short": 2, "smallint": 2, "i16": 2, "u16": 2, "int16": 2, "uint16": 2, "usmallint": 2,
    "int": 4, "integer": 4, "i32": 4, "u32": 4, "int32": 4, "uint32": 4,
    "long": 8, "bigint": 8, "i64": 8, "u64": 8, "int64": 8, "uint64": 8, "ubigint": 8,
    "i128": 16, "u128": 16, "int128": 16, "uint128": 16,
    "hugeint": 16, "uhugeint": 16,
}


_FLOAT_BYTE_SIZES: dict[str, int] = {
    "f8": 1, "float8": 1, "fp8": 1, "e4m3": 1, "e5m2": 1,
    "f16": 2, "float16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "f32": 4, "float32": 4, "float": 4, "real": 4,
    "double": 8, "double_precision": 8, "f64": 8, "float64": 8,
}


def _default_integer_byte_size(name: str) -> int | None:
    """Look up the byte size for a named integer alias, or None if unknown."""
    return _INTEGER_BYTE_SIZES.get(name)


def _default_float_byte_size(name: str) -> int | None:
    """Look up the byte size for a named float alias, or None if unknown."""
    return _FLOAT_BYTE_SIZES.get(name)


@dataclass(frozen=True, slots=True)
class DataTypeMetadata:
    name: str | None = None

    nullable: bool | None = None
    ordered: bool | None = None
    sorted: bool | None = None

    precision: int | None = None
    scale: int | None = None
    length: int | None = None
    byte_size: int | None = None

    timezone: str | None = None
    unit: str | None = None
    encoding: str | None = None
    format: str | None = None

    enum_values: tuple[str, ...] = ()
    literals: tuple[object, ...] = ()

    args: tuple[object, ...] = ()
    flags: tuple[str, ...] = ()
    extras: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDataType:
    type_id: DataTypeId
    # Share the empty-metadata singleton instead of allocating a fresh
    # ``DataTypeMetadata`` (and its empty ``extras`` dict) on every
    # default construction. ``DataTypeMetadata`` is itself immutable
    # and downstream callers treat ``extras`` as read-only — see the
    # note on :func:`_parse_cached`.
    metadata: "DataTypeMetadata" = field(
        default_factory=lambda: _EMPTY_METADATA
    )

    name: str | None = None
    children: tuple["ParsedDataType", ...] = ()

    @property
    def nullable(self) -> bool | None:
        return self.metadata.nullable

    @property
    def byte_size(self) -> int | None:
        return self.metadata.byte_size

    @property
    def item(self) -> "ParsedDataType | None":
        return (
            self.children[0]
            if self.type_id is DataTypeId.ARRAY and self.children
            else None
        )

    @property
    def key(self) -> "ParsedDataType | None":
        return (
            self.children[0]
            if self.type_id is DataTypeId.MAP and len(self.children) >= 1
            else None
        )

    @property
    def value(self) -> "ParsedDataType | None":
        return (
            self.children[1]
            if self.type_id is DataTypeId.MAP and len(self.children) >= 2
            else None
        )

    @property
    def fields(self) -> tuple["ParsedDataType", ...]:
        return self.children if self.type_id is DataTypeId.STRUCT else ()

    @property
    def variants(self) -> tuple["ParsedDataType", ...]:
        return self.children if self.type_id is DataTypeId.UNION else ()

    @property
    def index_type(self) -> "ParsedDataType | None":
        return (
            self.children[0]
            if self.type_id is DataTypeId.DICTIONARY and len(self.children) >= 1
            else None
        )

    @property
    def value_type(self) -> "ParsedDataType | None":
        return (
            self.children[1]
            if self.type_id is DataTypeId.DICTIONARY and len(self.children) >= 2
            else None
        )

    @classmethod
    def parse(
        cls,
        value: str,
        *,
        default: Any = ...,
    ) -> "ParsedDataType":
        """Parse a DataType string.

        On parse failure the dispatch is driven by ``default``:

        * ``default = ...`` (the sentinel default) — re-raise the
          underlying :class:`ValueError`.  Use this when the caller
          treats parse failure as a programming error.
        * ``default`` is a :class:`ParsedDataType` — return it as-is.
        * ``default`` is a :class:`DataTypeId` — wrap into a bare
          :class:`ParsedDataType` with no metadata.
        * Anything else (including ``None``) — collapse to
          :data:`DataTypeId.OBJECT`, the "I don't know what this is,
          treat it as opaque" type. Strings used to land here too;
          OBJECT keeps the value untouched whereas STRING would have
          coerced through ``str()`` on cast.

        Successful parses are memoized via :func:`_parse_cached` so a
        schema that mentions ``map<string, struct<...>>`` a million
        times only walks the AST once. Failures are not cached — each
        call re-raises so a fix in the input doesn't get masked.
        """
        try:
            return _parse_cached(value)
        except ValueError as e:
            if default is ...:
                raise ValueError(
                    f"Failed to parse DataType string {value!r} as a DataType: {e}"
                ) from e
            if isinstance(default, ParsedDataType):
                return default
            if isinstance(default, DataTypeId):
                return cls(type_id=default, metadata=DataTypeMetadata())
            return cls(type_id=DataTypeId.OBJECT, metadata=DataTypeMetadata())

    @classmethod
    def parse_type_id(
        cls,
        value: str,
        *,
        default: Any = ...,
    ) -> DataTypeId:
        return cls.parse(
            value,
            default=default,
        ).type_id


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    value: str
    pos: int


_TOKEN_PUNCT = set("()[]<>{},:=|?!")
_QUOTE_CHARS = ("'", '"', "`", "´")

_MULTI_TOKEN_TYPE_NAMES = {
    "double precision",
    "character varying",
    "timestamp with time zone",
    "timestamp without time zone",
}

# First-word prefixes of every entry in ``_MULTI_TOKEN_TYPE_NAMES``.
# Used as a cheap gate in :meth:`_Parser._parse_type_head_name` so the
# 1-token fast path doesn't pay for the lookahead loop on every
# ``int`` / ``string`` / ``timestamp_ntz`` it sees.
_MULTI_TOKEN_FIRST_WORDS = frozenset(
    name.split(" ", 1)[0] for name in _MULTI_TOKEN_TYPE_NAMES
)

_TYPE_METADATA_KEYS = {
    "item",
    "item_type",
    "key",
    "key_type",
    "value",
    "value_type",
    "index",
    "index_type",
}


# ---------------------------------------------------------------------
# Fast-path metadata for single-token temporal / sized aliases.
#
# These spellings (Spark/Databricks/Polars/Arrow DDL: `date32`,
# `timestamp_ms`, `interval_year_month`, ...) imply both a type_id and
# partial metadata without taking a bracketed payload. They are
# terminal — a user who needs to combine unit with `tz` or `nullable`
# should use the canonical `timestamp[unit=..., tz=...]` form.
# ---------------------------------------------------------------------

_FAST_PATH_METADATA: dict[str, tuple[DataTypeId, DataTypeMetadata]] = {
    "timestamp_with_time_zone": (
        DataTypeId.TIMESTAMP,
        DataTypeMetadata(timezone="with_time_zone"),
    ),
    "timestamp_without_time_zone": (
        DataTypeId.TIMESTAMP,
        DataTypeMetadata(timezone="without_time_zone"),
    ),
    "timestamp_ntz": (DataTypeId.TIMESTAMP, DataTypeMetadata(timezone="ntz")),
    "timestamp_ltz": (DataTypeId.TIMESTAMP, DataTypeMetadata(timezone="ltz")),
    "timestamp_s": (DataTypeId.TIMESTAMP, DataTypeMetadata(unit="s")),
    "timestamp_ms": (DataTypeId.TIMESTAMP, DataTypeMetadata(unit="ms")),
    "timestamp_us": (DataTypeId.TIMESTAMP, DataTypeMetadata(unit="us")),
    "timestamp_ns": (DataTypeId.TIMESTAMP, DataTypeMetadata(unit="ns")),
    "datetime_s": (DataTypeId.TIMESTAMP, DataTypeMetadata(unit="s")),
    "datetime_ms": (DataTypeId.TIMESTAMP, DataTypeMetadata(unit="ms")),
    "datetime_us": (DataTypeId.TIMESTAMP, DataTypeMetadata(unit="us")),
    "datetime_ns": (DataTypeId.TIMESTAMP, DataTypeMetadata(unit="ns")),
    "duration_s": (DataTypeId.DURATION, DataTypeMetadata(unit="s")),
    "duration_ms": (DataTypeId.DURATION, DataTypeMetadata(unit="ms")),
    "duration_us": (DataTypeId.DURATION, DataTypeMetadata(unit="us")),
    "duration_ns": (DataTypeId.DURATION, DataTypeMetadata(unit="ns")),
    "interval_s": (DataTypeId.DURATION, DataTypeMetadata(unit="s")),
    "interval_ms": (DataTypeId.DURATION, DataTypeMetadata(unit="ms")),
    "interval_us": (DataTypeId.DURATION, DataTypeMetadata(unit="us")),
    "interval_ns": (DataTypeId.DURATION, DataTypeMetadata(unit="ns")),
    "interval_year_month": (
        DataTypeId.DURATION,
        DataTypeMetadata(unit="year_month"),
    ),
    "interval_day_time": (
        DataTypeId.DURATION,
        DataTypeMetadata(unit="day_time"),
    ),
    "interval_month_day_nano": (
        DataTypeId.DURATION,
        DataTypeMetadata(unit="month_day_nano"),
    ),
    "date32": (DataTypeId.DATE, DataTypeMetadata(byte_size=4, unit="day")),
    "date64": (DataTypeId.DATE, DataTypeMetadata(byte_size=8, unit="ms")),
    "time32": (DataTypeId.TIME, DataTypeMetadata(byte_size=4, unit="ms")),
    "time64": (DataTypeId.TIME, DataTypeMetadata(byte_size=8, unit="ns")),
    "time32_s": (DataTypeId.TIME, DataTypeMetadata(byte_size=4, unit="s")),
    "time32_ms": (DataTypeId.TIME, DataTypeMetadata(byte_size=4, unit="ms")),
    "time64_us": (DataTypeId.TIME, DataTypeMetadata(byte_size=8, unit="us")),
    "time64_ns": (DataTypeId.TIME, DataTypeMetadata(byte_size=8, unit="ns")),
}


# ---------------------------------------------------------------------
# Canonical type-name alias table.
#
# Hoisted to module scope so it's allocated exactly once — the parser
# is in a hot path during schema parsing / casting. Covers Arrow,
# Spark, Polars, Databricks, PostgreSQL, DuckDB, BigQuery, Snowflake,
# Rust, NumPy, and the internal spellings.
# ---------------------------------------------------------------------

_NAME_ALIASES: dict[str, tuple[str, DataTypeId | None]] = {
    # object / variant / null
    "object":       ("object",  DataTypeId.OBJECT),
    "any":          ("object",  DataTypeId.OBJECT),
    "variant":      ("object",  DataTypeId.OBJECT),
    "none":         ("none",    DataTypeId.NULL),
    "null":         ("none",    DataTypeId.NULL),
    "nil":          ("none",    DataTypeId.NULL),

    # bool
    "bool":         ("bool",    DataTypeId.BOOL),
    "boolean":      ("bool",    DataTypeId.BOOL),
    "bit":          ("bool",    DataTypeId.BOOL),

    # integers — Python / Arrow / Spark / Databricks / SQL
    # Sized aliases route to specialized type ids so the parsed result
    # carries signedness + width without leaning on metadata. Names with
    # no canonical width ("int128", "hugeint") stay on the generic
    # INTEGER id and rely on byte_size metadata.
    "int":          ("int",     DataTypeId.INT32),
    "integer":      ("integer", DataTypeId.INT32),
    "bigint":       ("bigint",  DataTypeId.INT64),
    "smallint":     ("smallint", DataTypeId.INT16),
    "tinyint":      ("tinyint", DataTypeId.INT8),
    "byte":         ("byte",    DataTypeId.INT8),
    "short":        ("short",   DataTypeId.INT16),
    "long":         ("long",    DataTypeId.INT64),
    # Rust / Arrow shorthand
    "i8":           ("i8",      DataTypeId.INT8),
    "i16":          ("i16",     DataTypeId.INT16),
    "i32":          ("i32",     DataTypeId.INT32),
    "i64":          ("i64",     DataTypeId.INT64),
    "i128":         ("i128",    DataTypeId.INTEGER),
    "u8":           ("u8",      DataTypeId.UINT8),
    "u16":          ("u16",     DataTypeId.UINT16),
    "u32":          ("u32",     DataTypeId.UINT32),
    "u64":          ("u64",     DataTypeId.UINT64),
    "u128":         ("u128",    DataTypeId.INTEGER),
    # Polars / NumPy spelling
    "int8":         ("int8",    DataTypeId.INT8),
    "int16":        ("int16",   DataTypeId.INT16),
    "int32":        ("int32",   DataTypeId.INT32),
    "int64":        ("int64",   DataTypeId.INT64),
    "int128":       ("int128",  DataTypeId.INTEGER),
    "uint8":        ("uint8",   DataTypeId.UINT8),
    "uint16":       ("uint16",  DataTypeId.UINT16),
    "uint32":       ("uint32",  DataTypeId.UINT32),
    "uint64":       ("uint64",  DataTypeId.UINT64),
    "uint128":      ("uint128", DataTypeId.INTEGER),
    # DuckDB wide / unsigned
    "utinyint":     ("utinyint", DataTypeId.UINT8),
    "usmallint":    ("usmallint", DataTypeId.UINT16),
    "uinteger":     ("uinteger", DataTypeId.UINT32),
    "ubigint":      ("ubigint", DataTypeId.UINT64),
    "hugeint":      ("hugeint", DataTypeId.INTEGER),
    "uhugeint":     ("uhugeint", DataTypeId.INTEGER),

    # floats
    "float":        ("float",   DataTypeId.FLOAT32),
    "double":       ("double",  DataTypeId.FLOAT64),
    "double_precision": ("double_precision", DataTypeId.FLOAT64),
    "real":         ("real",    DataTypeId.FLOAT32),
    "f8":           ("f8",      DataTypeId.FLOAT8),
    "fp8":          ("fp8",     DataTypeId.FLOAT8),
    "float8":       ("float8",  DataTypeId.FLOAT8),
    "e4m3":         ("e4m3",    DataTypeId.FLOAT8),
    "e5m2":         ("e5m2",    DataTypeId.FLOAT8),
    "f16":          ("f16",     DataTypeId.FLOAT16),
    "f32":          ("f32",     DataTypeId.FLOAT32),
    "f64":          ("f64",     DataTypeId.FLOAT64),
    "float16":      ("float16", DataTypeId.FLOAT16),
    "float32":      ("float32", DataTypeId.FLOAT32),
    "float64":      ("float64", DataTypeId.FLOAT64),
    "half":         ("half",    DataTypeId.FLOAT16),
    "bfloat16":     ("bfloat16", DataTypeId.FLOAT16),
    "bf16":         ("bfloat16", DataTypeId.FLOAT16),

    # decimals
    "decimal":      ("decimal", DataTypeId.DECIMAL),
    "decimal128":   ("decimal", DataTypeId.DECIMAL),
    "decimal256":   ("decimal", DataTypeId.DECIMAL),
    "numeric":      ("decimal", DataTypeId.DECIMAL),
    "number":       ("decimal", DataTypeId.DECIMAL),
    "bignumeric":   ("decimal", DataTypeId.DECIMAL),
    "money":        ("decimal", DataTypeId.DECIMAL),

    # date / time
    "date":         ("date",    DataTypeId.DATE),
    "date32":       ("date32",  DataTypeId.DATE),
    "date64":       ("date64",  DataTypeId.DATE),
    "time":         ("time",    DataTypeId.TIME),
    "time32":       ("time32",  DataTypeId.TIME),
    "time64":       ("time64",  DataTypeId.TIME),
    "time32_s":     ("time32_s", DataTypeId.TIME),
    "time32_ms":    ("time32_ms", DataTypeId.TIME),
    "time64_us":    ("time64_us", DataTypeId.TIME),
    "time64_ns":    ("time64_ns", DataTypeId.TIME),

    # timestamps
    "timestamp":    ("timestamp", DataTypeId.TIMESTAMP),
    "datetime":     ("timestamp", DataTypeId.TIMESTAMP),
    "timestamp_with_time_zone":    ("timestamp_with_time_zone", DataTypeId.TIMESTAMP),
    "timestamp_without_time_zone": ("timestamp_without_time_zone", DataTypeId.TIMESTAMP),
    "timestamp_ntz": ("timestamp_ntz", DataTypeId.TIMESTAMP),
    "timestamp_ltz": ("timestamp_ltz", DataTypeId.TIMESTAMP),
    "timestamp_s":  ("timestamp_s", DataTypeId.TIMESTAMP),
    "timestamp_ms": ("timestamp_ms", DataTypeId.TIMESTAMP),
    "timestamp_us": ("timestamp_us", DataTypeId.TIMESTAMP),
    "timestamp_ns": ("timestamp_ns", DataTypeId.TIMESTAMP),
    "datetime_s":   ("datetime_s", DataTypeId.TIMESTAMP),
    "datetime_ms":  ("datetime_ms", DataTypeId.TIMESTAMP),
    "datetime_us":  ("datetime_us", DataTypeId.TIMESTAMP),
    "datetime_ns":  ("datetime_ns", DataTypeId.TIMESTAMP),

    # durations / intervals
    "duration":     ("duration", DataTypeId.DURATION),
    "duration_s":   ("duration_s", DataTypeId.DURATION),
    "duration_ms":  ("duration_ms", DataTypeId.DURATION),
    "duration_us":  ("duration_us", DataTypeId.DURATION),
    "duration_ns":  ("duration_ns", DataTypeId.DURATION),
    "interval":     ("duration", DataTypeId.DURATION),
    "interval_s":   ("interval_s", DataTypeId.DURATION),
    "interval_ms":  ("interval_ms", DataTypeId.DURATION),
    "interval_us":  ("interval_us", DataTypeId.DURATION),
    "interval_ns":  ("interval_ns", DataTypeId.DURATION),
    "interval_year_month": ("interval_year_month", DataTypeId.DURATION),
    "interval_day_time":   ("interval_day_time",   DataTypeId.DURATION),
    "interval_month_day_nano": ("interval_month_day_nano", DataTypeId.DURATION),
    "timedelta":    ("duration", DataTypeId.DURATION),

    # binary
    "binary":       ("binary",  DataTypeId.BINARY),
    "bytes":        ("binary",  DataTypeId.BINARY),
    "bytea":        ("binary",  DataTypeId.BINARY),
    "blob":         ("binary",  DataTypeId.BINARY),
    "large_binary": ("binary",  DataTypeId.BINARY),
    "fixed_size_binary": ("binary", DataTypeId.BINARY),

    # strings
    "string":       ("string",  DataTypeId.STRING),
    "str":          ("string",  DataTypeId.STRING),
    "text":         ("string",  DataTypeId.STRING),
    "utf8":         ("string",  DataTypeId.STRING),
    "large_utf8":   ("string",  DataTypeId.STRING),
    "large_string": ("string",  DataTypeId.STRING),
    "varchar":      ("varchar", DataTypeId.STRING),
    "char":         ("char",    DataTypeId.STRING),
    "character_varying": ("character_varying", DataTypeId.STRING),
    "character":    ("character", DataTypeId.STRING),

    # collections
    "array":        ("array",   DataTypeId.ARRAY),
    "list":         ("array",   DataTypeId.ARRAY),
    "large_list":   ("array",   DataTypeId.ARRAY),
    "fixed_size_list": ("array", DataTypeId.ARRAY),
    "set":          ("set",     DataTypeId.ARRAY),
    "frozenset":    ("frozenset", DataTypeId.ARRAY),

    # map
    "map":          ("map",     DataTypeId.MAP),
    "dict":         ("map",     DataTypeId.MAP),
    "mapping":      ("map",     DataTypeId.MAP),

    # struct / record
    "struct":       ("struct",  DataTypeId.STRUCT),
    "row":          ("struct",  DataTypeId.STRUCT),
    "record":       ("struct",  DataTypeId.STRUCT),
    "tuple":        ("tuple",   DataTypeId.STRUCT),

    # union / json / enum / dictionary
    "union":        ("union",   DataTypeId.UNION),
    # Bare ``json`` resolves to BJSON (the binary/packed variant) — the
    # text-shaped form is reachable as ``sjson`` / ``json_string``.
    "json":         ("bjson",   DataTypeId.BJSON),
    "bjson":        ("bjson",   DataTypeId.BJSON),
    "jsonb":        ("bjson",   DataTypeId.BJSON),
    "json_binary":  ("bjson",   DataTypeId.BJSON),
    "sjson":        ("sjson",   DataTypeId.SJSON),
    "json_string":  ("sjson",   DataTypeId.SJSON),
    "json_text":    ("sjson",   DataTypeId.SJSON),
    "enum":         ("enum",    DataTypeId.ENUM),
    "literal":      ("literal", DataTypeId.ENUM),
    "dictionary":   ("dictionary", DataTypeId.DICTIONARY),
    "categorical":  ("dictionary", DataTypeId.DICTIONARY),

    # wrappers handled specially in parse_primary (no direct DataTypeId)
    "optional":     ("optional", None),
    "annotated":    ("annotated", None),

    # geospatial — not a native type_id, fall through to string
    "geography":    ("string",  DataTypeId.STRING),
    "geometry":     ("string",  DataTypeId.STRING),
}


# ---------------------------------------------------------------------
# Set of type-ids whose single-token form accepts a bracketed payload.
# Used by the "metadata-only" fallback in parse_primary so plain types
# like `int32[nullable=false]` or `bool[encoding=thrift]` work instead
# of leaving the bracket as trailing and failing outright.
#
# Compound/parameterized types (DECIMAL / VARCHAR / ARRAY / MAP / ...)
# have dedicated branches earlier in parse_primary and don't need this.
# ---------------------------------------------------------------------

# Keyword heads that ``parse_primary`` dispatches before falling through
# to the regular type-name path. Keeping them in one frozenset lets the
# parser do a single ``str.lower()`` + ``in`` check instead of four
# separate ``_peek_ident_ci`` calls per primary.
_KEYWORD_HEADS = frozenset({"optional", "annotated", "union", "literal"})


# Shared singleton for "no metadata declared". Every ``DataTypeMetadata()``
# default allocates an ``extras`` dict via ``default_factory``; sharing
# one instance turns the no-op metadata path into a pointer copy.
# Safe because ``ParsedDataType`` is immutable and downstream code is
# documented to treat ``DataTypeMetadata.extras`` as read-only (see the
# note on :func:`_parse_cached`).
_EMPTY_METADATA = DataTypeMetadata()


_BRACKET_METADATA_TYPE_IDS = frozenset({
    DataTypeId.BOOL,
    DataTypeId.INTEGER,
    DataTypeId.INT8, DataTypeId.INT16, DataTypeId.INT32, DataTypeId.INT64,
    DataTypeId.UINT8, DataTypeId.UINT16, DataTypeId.UINT32, DataTypeId.UINT64,
    DataTypeId.FLOAT,
    DataTypeId.FLOAT8, DataTypeId.FLOAT16,
    DataTypeId.FLOAT32, DataTypeId.FLOAT64,
    DataTypeId.DATE,
    DataTypeId.NULL,
    DataTypeId.OBJECT,
    DataTypeId.SJSON,
    DataTypeId.BJSON,
})


class _Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.length = len(text)
        self.index = 0

    def lex(self) -> list[Token]:
        tokens: list[Token] = []

        while self.index < self.length:
            ch = self.text[self.index]

            if ch.isspace():
                self.index += 1
                continue

            if ch in _TOKEN_PUNCT:
                tokens.append(Token("punct", ch, self.index))
                self.index += 1
                continue

            if ch in _QUOTE_CHARS:
                tokens.append(self._read_string())
                continue

            if ch.isdigit() or (ch in "+-" and self._peek_is_digit()):
                tokens.append(self._read_number())
                continue

            tokens.append(self._read_identifier())

        return tokens

    def _peek_is_digit(self) -> bool:
        return self.index + 1 < self.length and self.text[self.index + 1].isdigit()

    def _read_string(self) -> Token:
        quote = self.text[self.index]
        start = self.index
        self.index += 1
        chars: list[str] = []

        while self.index < self.length:
            ch = self.text[self.index]
            if ch == "\\" and self.index + 1 < self.length:
                chars.append(self.text[self.index + 1])
                self.index += 2
                continue
            if ch == quote:
                self.index += 1
                return Token("string", "".join(chars), start)
            chars.append(ch)
            self.index += 1

        raise ValueError(f"Unterminated quoted string at position {start}")

    def _read_number(self) -> Token:
        start = self.index
        if self.text[self.index] in "+-":
            self.index += 1

        has_dot = False
        while self.index < self.length:
            ch = self.text[self.index]
            if ch.isdigit():
                self.index += 1
                continue
            if ch == "." and not has_dot:
                has_dot = True
                self.index += 1
                continue
            break

        return Token("number", self.text[start : self.index], start)

    def _read_identifier(self) -> Token:
        start = self.index
        while self.index < self.length:
            ch = self.text[self.index]
            if ch.isspace() or ch in _TOKEN_PUNCT or ch in _QUOTE_CHARS:
                break
            self.index += 1
        return Token("ident", self.text[start : self.index], start)


class _Parser:
    __slots__ = ("text", "tokens", "n_tokens", "index", "default")

    def __init__(self, text: str, *, default: Any = ...) -> None:
        self.text = text
        self.tokens = _Lexer(text).lex()
        # Cached once so the per-token ``_current`` / ``_peek_token`` /
        # ``_at_end`` checks don't pay a ``len()`` call each — those
        # three helpers fire on every token boundary, so the cost adds
        # up across deeply nested DDL.
        self.n_tokens = len(self.tokens)
        self.index = 0
        # Stored for subclass / debug use; the parser body always
        # raises on failure and lets :meth:`ParsedDataType.parse`
        # decide whether to re-raise or fall back to ``default``.
        self.default = default

    def parse(self) -> ParsedDataType:
        if not self.tokens:
            return self._fail("DataType string cannot be empty")

        result = self.parse_type()

        if self._match_ident_phrase("not", "null"):
            result = _set_nullable(result, False)
        elif self._match_ident_phrase("non", "null"):
            result = _set_nullable(result, False)

        if self._peek_punct("?"):
            self._advance()
            result = _set_nullable(result, True)
        elif self._peek_punct("!"):
            self._advance()
            result = _set_nullable(result, False)

        if not self._at_end():
            return self._fail(
                f"Unexpected trailing tokens starting at {self._current().value!r}"
            )

        return result

    def parse_type(self) -> ParsedDataType:
        left = self._apply_postfix_array(self.parse_primary())

        variants = [left]
        saw_pipe = False

        while self._peek_punct("|"):
            saw_pipe = True
            self._advance()
            variants.append(self._apply_postfix_array(self.parse_primary()))

        if not saw_pipe:
            return left

        non_null: list[ParsedDataType] = []
        nullable = False

        for variant in variants:
            if variant.type_id is DataTypeId.NULL:
                nullable = True
            else:
                non_null.append(variant)

        if len(non_null) == 1:
            return _set_nullable(
                non_null[0], True if nullable else non_null[0].metadata.nullable
            )

        return ParsedDataType(
            type_id=DataTypeId.UNION,
            metadata=DataTypeMetadata(nullable=True if nullable else None),
            children=tuple(non_null),
        )

    def parse_primary(self) -> ParsedDataType:
        # Dispatch the four special keyword heads (``optional`` /
        # ``annotated`` / ``union`` / ``literal``) through a single
        # ``.lower()`` + set check instead of four separate
        # ``_peek_ident_ci`` calls. ``parse_primary`` runs once per
        # type node, so this gate fires on every nested struct field.
        tok = self.tokens[self.index] if self.index < self.n_tokens else None
        if tok is not None and tok.kind == "ident":
            head_low = tok.value.lower()
            if head_low in _KEYWORD_HEADS:
                if head_low == "optional":
                    self.index += 1
                    inner = self._parse_generic_single()
                    return _set_nullable(inner, True)

                if head_low == "annotated":
                    self.index += 1
                    inner, extras = self._parse_annotated()
                    return ParsedDataType(
                        type_id=inner.type_id,
                        metadata=replace(
                            inner.metadata,
                            extras={**inner.metadata.extras, **extras},
                        ),
                        name=inner.name,
                        children=inner.children,
                    )

                if head_low == "union":
                    self.index += 1
                    parts = self._parse_generic_list()
                    non_null: list[ParsedDataType] = []
                    nullable = False

                    for part in parts:
                        if part.type_id is DataTypeId.NULL:
                            nullable = True
                        else:
                            non_null.append(part)

                    if len(non_null) == 1:
                        return _set_nullable(
                            non_null[0],
                            True if nullable else non_null[0].metadata.nullable,
                        )

                    return ParsedDataType(
                        type_id=DataTypeId.UNION,
                        metadata=DataTypeMetadata(
                            nullable=True if nullable else None
                        ),
                        children=tuple(non_null),
                    )

                # head_low == "literal"
                self.index += 1
                literals = self._parse_literal_list()
                return ParsedDataType(
                    type_id=DataTypeId.ENUM,
                    metadata=DataTypeMetadata(
                        literals=tuple(literals),
                        enum_values=tuple(
                            v for v in literals if isinstance(v, str)
                        ),
                    ),
                )

        if self._peek_punct("("):
            self._advance()
            inner = self.parse_type()
            self._expect_punct(")")
            return inner

        token = self._current()
        if token is None:
            return self._fail("Unexpected end of type expression")

        if token.kind == "number":
            self._advance()
            try:
                return ParsedDataType(type_id=DataTypeId(int(token.value)))
            except ValueError:
                return self._fail(f"Unknown numeric DataTypeId: {token.value!r}")

        if token.kind not in {"ident", "string"}:
            return self._fail(f"Unexpected token {token.value!r}")

        raw_name = self._parse_type_head_name()
        canonical, dtype = _canonical_name(raw_name)

        fast_path = _FAST_PATH_METADATA.get(canonical)
        if fast_path is not None:
            type_id, metadata = fast_path
            return ParsedDataType(type_id=type_id, metadata=metadata)

        if canonical == "none":
            return ParsedDataType(type_id=DataTypeId.NULL)

        if canonical == "enum":
            if self._peek_any_generic_open():
                args = self._parse_scalar_args()
                return ParsedDataType(
                    type_id=DataTypeId.ENUM,
                    metadata=DataTypeMetadata(
                        literals=tuple(args),
                        enum_values=tuple(v for v in args if isinstance(v, str)),
                    ),
                )
            return ParsedDataType(type_id=DataTypeId.ENUM)

        if canonical == "array":
            if self._peek_any_generic_open():
                item = self._parse_generic_single()
                return ParsedDataType(
                    type_id=DataTypeId.ARRAY,
                    children=(item,),
                )
            return ParsedDataType(type_id=DataTypeId.ARRAY)

        if canonical in {"set", "frozenset"}:
            if self._peek_any_generic_open():
                item = self._parse_generic_single()
                return ParsedDataType(
                    type_id=DataTypeId.ARRAY,
                    metadata=DataTypeMetadata(
                        ordered=False,
                        extras={"container": canonical},
                    ),
                    children=(item,),
                )
            return ParsedDataType(
                type_id=DataTypeId.ARRAY,
                metadata=DataTypeMetadata(
                    ordered=False,
                    extras={"container": canonical},
                ),
            )

        if canonical == "map":
            if self._peek_any_generic_open():
                parts = self._parse_generic_list()
                if len(parts) != 2:
                    return self._fail("map/dict requires exactly two type parameters")
                return ParsedDataType(
                    type_id=DataTypeId.MAP,
                    children=(parts[0], parts[1]),
                )
            return ParsedDataType(type_id=DataTypeId.MAP)

        if canonical == "tuple":
            if self._peek_any_generic_open():
                parts = self._parse_generic_list()
                fields = tuple(
                    ParsedDataType(
                        type_id=part.type_id,
                        metadata=part.metadata,
                        name=f"f{idx}",
                        children=part.children,
                    )
                    for idx, part in enumerate(parts)
                )
                return ParsedDataType(
                    type_id=DataTypeId.STRUCT,
                    metadata=DataTypeMetadata(
                        ordered=True,
                        extras={"container": "tuple"},
                    ),
                    children=fields,
                )
            return ParsedDataType(
                type_id=DataTypeId.STRUCT,
                metadata=DataTypeMetadata(
                    ordered=True,
                    extras={"container": "tuple"},
                ),
            )

        if canonical == "struct":
            if self._peek_any_generic_open():
                open_tok = self._advance()
                close_char = _matching_close(open_tok.value)
                fields = self._parse_struct_fields(close_char)
                return ParsedDataType(
                    type_id=DataTypeId.STRUCT,
                    children=tuple(fields),
                )
            return ParsedDataType(type_id=DataTypeId.STRUCT)

        if canonical == "decimal":
            # DECIMAL accepts three bracketed shapes:
            #   decimal(10, 2)                — positional precision/scale
            #   decimal[precision=10, scale=2] — keyword form
            #   decimal(10, 2, "native")      — free-form args bucket
            if self._peek_any_generic_open():
                return self._parse_parameterized_decimal()
            return ParsedDataType(type_id=DataTypeId.DECIMAL)

        if canonical in {"varchar", "char", "character_varying", "character"}:
            # Length can come as `varchar(10)` (positional) or
            # `varchar[length=10]` (keyword).
            if self._peek_any_generic_open():
                return self._parse_parameterized_string()
            return ParsedDataType(type_id=DataTypeId.STRING)

        if dtype in {
            DataTypeId.STRING,
            DataTypeId.BINARY,
            DataTypeId.SJSON,
            DataTypeId.BJSON,
            DataTypeId.TIME,
            DataTypeId.TIMESTAMP,
            DataTypeId.DURATION,
        }:
            if (
                self._peek_any_generic_open()
                and not self._peek_postfix_empty_array()
            ):
                open_tok = self._advance()
                close_char = _matching_close(open_tok.value)
                metadata = self._parse_metadata(close_char)
                return ParsedDataType(type_id=dtype, metadata=metadata)
            return ParsedDataType(type_id=dtype)

        if dtype is DataTypeId.DICTIONARY:
            if (
                self._peek_any_generic_open()
                and not self._peek_postfix_empty_array()
            ):
                open_tok = self._advance()
                close_char = _matching_close(open_tok.value)
                metadata, children = self._parse_dictionary_payload(close_char)
                return ParsedDataType(
                    type_id=DataTypeId.DICTIONARY,
                    metadata=metadata,
                    children=children,
                )
            return ParsedDataType(type_id=DataTypeId.DICTIONARY)

        if dtype is None:
            # Unknown identifier (e.g. a dataclass forward-ref string like
            # "dt.datetime"). Treat as opaque OBJECT and keep the original
            # name on the parsed metadata so callers can still see it.
            if (
                self._peek_any_generic_open()
                and not self._peek_postfix_empty_array()
            ):
                open_tok = self._advance()
                close_char = _matching_close(open_tok.value)
                # Drain the bracketed payload so the outer tokenstream stays
                # balanced; we don't carry the args forward.
                self._parse_scalar_items(close_char)
            return ParsedDataType(
                type_id=DataTypeId.OBJECT,
                metadata=DataTypeMetadata(name=raw_name),
                name=raw_name,
            )

        # Final fallback for simple types (INTEGER/FLOAT/BOOL/DATE/...).
        # These have no compound syntax of their own, but we still accept
        # a bracketed metadata payload so callers can always attach
        # nullability / encoding / format without having to reach for the
        # generic `[...]` form of a wider type.
        base_metadata = self._default_metadata_for(dtype, canonical)

        if (
            dtype in _BRACKET_METADATA_TYPE_IDS
            and self._peek_any_generic_open()
            and not self._peek_postfix_empty_array()
        ):
            open_tok = self._advance()
            close_char = _matching_close(open_tok.value)
            payload = self._parse_metadata(close_char)
            # Preserve the byte_size default (not represented in the
            # bracketed payload) unless the user explicitly overrode it.
            if base_metadata.byte_size is not None and payload.byte_size is None:
                payload = replace(payload, byte_size=base_metadata.byte_size)
            return ParsedDataType(type_id=dtype, metadata=payload)

        return ParsedDataType(type_id=dtype, metadata=base_metadata)

    # ------------------------------------------------------------------
    # Parameterized forms split out so parse_primary stays readable.
    # ------------------------------------------------------------------

    def _parse_parameterized_decimal(self) -> ParsedDataType:
        """Handle decimal(p, s), decimal[precision=.., scale=..], and mixed."""
        open_tok = self._advance()
        close_char = _matching_close(open_tok.value)
        items: list[tuple[str | None, object]] = []

        if self._peek_punct(close_char):
            self._advance()
            return ParsedDataType(type_id=DataTypeId.DECIMAL)

        while True:
            items.append(self._parse_metadata_item())
            if self._peek_punct(","):
                self._advance()
                if self._peek_punct(close_char):
                    break
                continue
            break
        self._expect_punct(close_char)

        # All-positional: classic decimal(p, s) [, extra...].
        if all(key is None for key, _ in items):
            values = [value for _, value in items]
            if len(values) == 2 and all(isinstance(v, int) for v in values):
                return ParsedDataType(
                    type_id=DataTypeId.DECIMAL,
                    metadata=DataTypeMetadata(
                        precision=int(values[0]),
                        scale=int(values[1]),
                    ),
                )
            return ParsedDataType(
                type_id=DataTypeId.DECIMAL,
                metadata=DataTypeMetadata(args=tuple(values)),
            )

        # Mixed / keyword: route through the generic metadata builder.
        metadata = _metadata_from_items(items)
        return ParsedDataType(type_id=DataTypeId.DECIMAL, metadata=metadata)

    def _parse_parameterized_string(self) -> ParsedDataType:
        """Handle varchar(n), varchar[length=n], and friends."""
        open_tok = self._advance()
        close_char = _matching_close(open_tok.value)
        items: list[tuple[str | None, object]] = []

        if self._peek_punct(close_char):
            self._advance()
            return ParsedDataType(type_id=DataTypeId.STRING)

        while True:
            items.append(self._parse_metadata_item())
            if self._peek_punct(","):
                self._advance()
                if self._peek_punct(close_char):
                    break
                continue
            break
        self._expect_punct(close_char)

        # All-positional: varchar(10) [, extra...].
        if all(key is None for key, _ in items):
            values = [value for _, value in items]
            length = int(values[0]) if values and isinstance(values[0], int) else None
            return ParsedDataType(
                type_id=DataTypeId.STRING,
                metadata=DataTypeMetadata(length=length, args=tuple(values)),
            )

        metadata = _metadata_from_items(items)
        return ParsedDataType(type_id=DataTypeId.STRING, metadata=metadata)

    def _default_metadata_for(
        self,
        dtype: DataTypeId,
        canonical: str,
    ) -> DataTypeMetadata:
        """Default metadata for a plain-token type (no brackets seen yet)."""
        # Inline the bounds check from ``DataTypeId.is_integer`` /
        # ``is_floating_point`` — both are simple range tests, and the
        # property-attribute lookup is the dominant cost here.
        value = dtype.value
        if 20 <= value < 40:
            byte_size = _INTEGER_BYTE_SIZES.get(canonical)
            if byte_size is None:
                return _EMPTY_METADATA
            return DataTypeMetadata(byte_size=byte_size)
        if 40 <= value < 50:
            byte_size = _FLOAT_BYTE_SIZES.get(canonical)
            if byte_size is None:
                return _EMPTY_METADATA
            return DataTypeMetadata(byte_size=byte_size)
        return _EMPTY_METADATA

    # ------------------------------------------------------------------

    def _parse_type_head_name(self) -> str:
        tok = self._current()
        if tok is None or tok.kind not in {"ident", "string"}:
            return self._fail("Expected type name")

        # Fast path: the overwhelming majority of type heads are a
        # single token (``int`` / ``bigint`` / ``timestamp_ntz`` /
        # ``array`` / …). Only the four entries in
        # ``_MULTI_TOKEN_TYPE_NAMES`` need lookahead, and they all
        # start with one of three identifiers — gate on that prefix
        # set instead of building a list + ``" ".join`` + ``.lower()``
        # per token on every type.
        head_low = tok.value.lower()
        if head_low not in _MULTI_TOKEN_FIRST_WORDS:
            self.index += 1
            return tok.value

        max_parts = 4
        best_len = 1
        lookahead_parts = [head_low]

        for offset in range(1, max_parts):
            nxt = self._peek_token(offset)
            if nxt is None or nxt.kind != "ident":
                break
            lookahead_parts.append(nxt.value.lower())
            candidate = " ".join(lookahead_parts)
            if candidate in _MULTI_TOKEN_TYPE_NAMES:
                best_len = len(lookahead_parts)

        if best_len == 1:
            self.index += 1
            return tok.value

        selected: list[str] = []
        for _ in range(best_len):
            current = self._current()
            if current is None:
                break
            selected.append(current.value)
            self.index += 1

        return " ".join(selected)

    def _parse_annotated(self) -> tuple[ParsedDataType, dict[str, object]]:
        parts = self._parse_generic_mixed()
        if not parts:
            return self._fail("Annotated[...] requires at least one argument")

        first = parts[0]
        if not isinstance(first, ParsedDataType):
            return self._fail("Annotated first argument must be a type")

        extras = {
            f"annotation_{idx}": value for idx, value in enumerate(parts[1:], start=1)
        }
        return first, extras

    def _parse_literal_list(self) -> list[object]:
        open_tok = self._expect_any_generic_open()
        close_char = _matching_close(open_tok.value)
        values = self._parse_scalar_items(close_char)
        return [_coerce_literal_value(v) for v in values]

    def _parse_generic_single(self) -> ParsedDataType:
        parts = self._parse_generic_list()
        if len(parts) != 1:
            return self._fail("Expected exactly one type parameter")
        return parts[0]

    def _parse_generic_list(self) -> list[ParsedDataType]:
        open_tok = self._expect_any_generic_open()
        close_char = _matching_close(open_tok.value)
        items: list[ParsedDataType] = []

        if self._peek_punct(close_char):
            self._advance()
            return items

        while True:
            items.append(self.parse_type())
            if self._peek_punct(","):
                self._advance()
                # Trailing comma — `array<int,>` / `map<k, v,>` — is
                # accepted to match formatted SQL DDL output.
                if self._peek_punct(close_char):
                    break
                continue
            break

        self._expect_punct(close_char)
        return items

    def _parse_generic_mixed(self) -> list[object]:
        open_tok = self._expect_any_generic_open()
        close_char = _matching_close(open_tok.value)
        return self._parse_mixed_items(close_char)

    def _parse_mixed_items(self, close_char: str) -> list[object]:
        items: list[object] = []

        if self._peek_punct(close_char):
            self._advance()
            return items

        while True:
            items.append(self._parse_mixed_value())
            if self._peek_punct(","):
                self._advance()
                if self._peek_punct(close_char):
                    break
                continue
            break

        self._expect_punct(close_char)
        return items

    def _parse_scalar_items(self, close_char: str) -> list[object]:
        items: list[object] = []

        if self._peek_punct(close_char):
            self._advance()
            return items

        while True:
            items.append(self._parse_scalar_or_symbol())
            if self._peek_punct(","):
                self._advance()
                if self._peek_punct(close_char):
                    break
                continue
            break

        self._expect_punct(close_char)
        return items

    def _parse_mixed_value(self) -> object:
        tok = self._current()
        if tok is None:
            return self._fail("Unexpected end of expression")

        if tok.kind == "string":
            self._advance()
            return tok.value

        if tok.kind == "number":
            self._advance()
            return _parse_number(tok.value)

        if tok.kind == "ident":
            if self._looks_like_type():
                return self.parse_type()
            return self._parse_scalar_or_symbol()

        if tok.kind == "punct" and tok.value == "(":
            return self.parse_type()

        return self._fail(f"Unexpected mixed item token {tok.value!r}")

    def _parse_scalar_or_symbol(self) -> object:
        tok = self._current()
        if tok is None:
            return self._fail("Unexpected end of expression")

        if tok.kind == "string":
            self._advance()
            return tok.value

        if tok.kind == "number":
            self._advance()
            return _parse_number(tok.value)

        if tok.kind == "ident":
            self._advance()
            low = tok.value.lower()
            if low == "true":
                return True
            if low == "false":
                return False
            if low in {"none", "null", "nil"}:
                return None
            return tok.value

        return self._fail(f"Unexpected scalar token {tok.value!r}")

    def _parse_scalar_args(self) -> list[object]:
        open_tok = self._expect_any_generic_open()
        close_char = _matching_close(open_tok.value)
        return self._parse_scalar_items(close_char)

    def _parse_metadata(self, close_char: str) -> DataTypeMetadata:
        items: list[tuple[str | None, object]] = []

        if self._peek_punct(close_char):
            self._advance()
            return DataTypeMetadata()

        while True:
            key, value = self._parse_metadata_item()
            items.append((key, value))
            if self._peek_punct(","):
                self._advance()
                if self._peek_punct(close_char):
                    break
                continue
            break

        self._expect_punct(close_char)

        if all(key is None for key, _ in items):
            return DataTypeMetadata(args=tuple(value for _, value in items))

        return _metadata_from_items(items)

    def _parse_dictionary_payload(
        self,
        close_char: str,
    ) -> tuple[DataTypeMetadata, tuple[ParsedDataType, ...]]:
        items: list[tuple[str | None, object]] = []

        if self._peek_punct(close_char):
            self._advance()
            return DataTypeMetadata(), ()

        while True:
            key, value = self._parse_metadata_item()
            items.append((key, value))
            if self._peek_punct(","):
                self._advance()
                if self._peek_punct(close_char):
                    break
                continue
            break

        self._expect_punct(close_char)

        if all(key is None for key, _ in items):
            args = [value for _, value in items]
            if len(args) == 2 and all(isinstance(v, ParsedDataType) for v in args):
                return DataTypeMetadata(), (args[0], args[1])
            return DataTypeMetadata(args=tuple(args)), ()

        extras: dict[str, object] = {}
        flags: list[str] = []
        ordered: bool | None = None
        nullable: bool | None = None

        index_type: ParsedDataType | None = None
        value_type: ParsedDataType | None = None

        for key, value in items:
            if key is None:
                if isinstance(value, str):
                    flags.append(value)
                else:
                    extras[f"arg_{len(extras)}"] = value
                continue

            low = key.lower()

            if low in {"index", "index_type"} and isinstance(value, ParsedDataType):
                index_type = value
            elif low in {"value", "value_type"} and isinstance(value, ParsedDataType):
                value_type = value
            elif low == "ordered" and isinstance(value, bool):
                ordered = value
            elif low == "nullable" and isinstance(value, bool):
                nullable = value
            else:
                extras[key] = value

        children: tuple[ParsedDataType, ...] = ()
        if index_type is not None and value_type is not None:
            children = (index_type, value_type)

        return (
            DataTypeMetadata(
                ordered=ordered,
                nullable=nullable,
                flags=tuple(flags),
                extras=extras,
            ),
            children,
        )

    def _parse_metadata_item(self) -> tuple[str | None, object]:
        tok = self._current()
        if tok is None:
            return self._fail("Unexpected end in metadata")

        if tok.kind == "ident":
            if self._looks_like_key_value():
                key = tok.value
                self._advance()
                self._advance()

                if key.lower() in _TYPE_METADATA_KEYS:
                    value = self.parse_type()
                else:
                    value = self._parse_scalar_or_symbol()

                return key, value

            if self._looks_like_type():
                return None, self.parse_type()

            return None, self._parse_scalar_or_symbol()

        if tok.kind == "string":
            self._advance()
            if self._peek_punct("=") or self._peek_punct(":"):
                key = tok.value
                self._advance()
                if key.lower() in _TYPE_METADATA_KEYS:
                    value = self.parse_type()
                else:
                    value = self._parse_scalar_or_symbol()
                return key, value
            return None, tok.value

        if tok.kind == "number":
            self._advance()
            return None, _parse_number(tok.value)

        return None, self._parse_mixed_value()

    def _parse_struct_fields(self, close_char: str) -> list[ParsedDataType]:
        fields: list[ParsedDataType] = []

        if self._peek_punct(close_char):
            self._advance()
            return fields

        while True:
            fields.append(self._parse_struct_field())
            if self._peek_punct(","):
                self._advance()
                if self._peek_punct(close_char):
                    break
                continue
            break

        self._expect_punct(close_char)
        return fields

    def _parse_struct_field(self) -> ParsedDataType:
        tok = self._current()
        if tok is None:
            return self._fail("Unexpected end in struct field")

        if tok.kind not in {"ident", "string"}:
            return self._fail(
                "Struct field name must be an identifier or quoted string"
            )

        self._advance()
        name = tok.value

        # XML-namespaced field names contain ':' (e.g. `_xml:lang`,
        # `_rtr:msgType`, `rtr:versionedId`). The lexer treats `:`
        # as punctuation and splits those names into separate tokens,
        # so glue them back here: while the lookahead is `:` `ident`
        # `:` — i.e. another colon follows that would act as the
        # real name/type separator — fold `:ident` into the name.
        # Quoted names are already a single token and don't take
        # this path.
        if tok.kind == "ident":
            while (
                self._peek_punct(":")
                and (next_tok := self._peek_token(1)) is not None
                and next_tok.kind == "ident"
                and (after := self._peek_token(2)) is not None
                and after.kind == "punct"
                and after.value == ":"
            ):
                self._advance()
                part = self._advance()
                name = f"{name}:{part.value}"

        nullable: bool | None = None

        if self._peek_punct("?"):
            self._advance()
            nullable = True
        elif self._peek_punct("!"):
            self._advance()
            nullable = False

        # Accept three field-name/type separators:
        #   * `:`  — Spark / Databricks / Polars (`a: int`)
        #   * `=`  — Python-typing style (`a = int`)
        #   * (none) — Hive / BigQuery DDL (`a int`, `a STRING`)
        # The space-separated form falls through whenever the next
        # token can start a type (an identifier, quoted name, number,
        # or a `(`-grouped expression). That matches what users
        # actually paste from `SHOW CREATE TABLE` output.
        sep = self._current()
        if sep is not None and sep.kind == "punct" and sep.value in {":", "="}:
            self._advance()
        elif sep is None or not (
            sep.kind in {"ident", "string", "number"}
            or (sep.kind == "punct" and sep.value == "(")
        ):
            return self._fail(
                f"Struct field {name!r} expected ':' / '=' or a type, "
                f"got {sep.value!r}" if sep is not None
                else f"Struct field {name!r} expected ':' / '=' or a type"
            )
        field_type = self.parse_type()

        if self._match_ident_phrase("not", "null") or self._match_ident_phrase(
            "non", "null"
        ):
            if nullable is True:
                return self._fail(
                    f"Struct field {name!r} marks both nullable ('?') and "
                    "non-nullable ('NOT NULL'); pick one"
                )
            nullable = False
        elif self._peek_punct("!"):
            self._advance()
            nullable = False
        elif self._peek_punct("?"):
            self._advance()
            nullable = True

        effective_nullable = (
            nullable if nullable is not None else field_type.metadata.nullable
        )

        return ParsedDataType(
            type_id=field_type.type_id,
            metadata=replace(field_type.metadata, nullable=effective_nullable),
            name=name,
            children=field_type.children,
        )

    def _looks_like_type(self) -> bool:
        tok = self._current()
        if tok is None:
            return False
        if tok.kind == "number":
            return True
        if tok.kind == "string":
            return False
        if tok.kind != "ident":
            return False
        return tok.value.lower() not in {"true", "false"}

    def _looks_like_key_value(self) -> bool:
        tok = self._current()
        if tok is None or tok.kind not in {"ident", "string"}:
            return False
        nxt = self._peek_token(1)
        return nxt is not None and nxt.kind == "punct" and nxt.value in {"=", ":"}

    def _peek_any_generic_open(self) -> bool:
        tok = self._current()
        return tok is not None and tok.kind == "punct" and tok.value in {"(", "[", "<"}

    def _peek_postfix_empty_array(self) -> bool:
        """True iff the next two tokens are ``[`` then ``]``.

        That bracket pair is the PostgreSQL / Hive postfix-array marker
        (``int[]`` → ``array<int>``) and must NOT be consumed by the
        bracketed-metadata branches in :meth:`parse_primary`.
        """
        if not self._peek_punct("["):
            return False
        nxt = self._peek_token(1)
        return nxt is not None and nxt.kind == "punct" and nxt.value == "]"

    def _apply_postfix_array(self, parsed: ParsedDataType) -> ParsedDataType:
        """Wrap ``parsed`` in ``array<...>`` for each trailing ``[]`` pair.

        Handles the PostgreSQL postfix syntax — ``int[]``, ``text[][]``,
        ``struct<a:int>[]`` — by greedily consuming empty bracket pairs.
        Non-empty brackets (``int[nullable=true]``) are NOT touched here;
        they belong to :meth:`parse_primary`'s metadata path.
        """
        while self._peek_postfix_empty_array():
            self._advance()  # consume '['
            self._advance()  # consume ']'
            parsed = ParsedDataType(
                type_id=DataTypeId.ARRAY,
                children=(parsed,),
            )
        return parsed

    def _expect_any_generic_open(self) -> Token:
        tok = self._current()
        if tok is None or tok.kind != "punct" or tok.value not in {"(", "[", "<"}:
            return self._fail("Expected one of '(', '[', '<'")
        self._advance()
        return tok

    def _expect_any_punct(self, *values: str) -> Token:
        tok = self._current()
        if tok is None or tok.kind != "punct" or tok.value not in values:
            return self._fail(f"Expected one of {values!r}")
        self._advance()
        return tok

    def _expect_punct(self, value: str) -> Token:
        tok = self._current()
        if tok is None or tok.kind != "punct" or tok.value != value:
            return self._fail(f"Expected {value!r}")
        self._advance()
        return tok

    def _peek_punct(self, value: str) -> bool:
        tok = self._current()
        return tok is not None and tok.kind == "punct" and tok.value == value

    def _peek_ident_ci(self, value: str) -> bool:
        tok = self._current()
        return (
            tok is not None
            and tok.kind == "ident"
            and tok.value.lower() == value.lower()
        )

    def _match_ident_phrase(self, *parts: str) -> bool:
        for offset, part in enumerate(parts):
            tok = self._peek_token(offset)
            if tok is None or tok.kind != "ident" or tok.value.lower() != part.lower():
                return False
        self.index += len(parts)
        return True

    def _current(self) -> Token | None:
        return self.tokens[self.index] if self.index < self.n_tokens else None

    def _peek_token(self, offset: int) -> Token | None:
        pos = self.index + offset
        return self.tokens[pos] if pos < self.n_tokens else None

    def _advance(self) -> Token:
        if self.index >= self.n_tokens:
            return self._fail("Unexpected end of input")
        tok = self.tokens[self.index]
        self.index += 1
        return tok

    def _at_end(self) -> bool:
        return self.index >= self.n_tokens

    def _fail(self, message: str) -> Any:
        """Signal a parse error by raising ``ValueError``.

        Always raises — the outer :meth:`ParsedDataType.parse` owns the
        recovery decision via its ``raise_error`` flag. Raising here
        (rather than returning a sentinel) keeps mid-parse state
        coherent: a sentinel return would cascade into
        ``AttributeError``s further up (e.g. ``_canonical_name(raw_name)``
        calling ``.strip()`` on a ``ParsedDataType``).
        """
        raise ValueError(message)


def _metadata_from_items(
    items: list[tuple[str | None, object]],
) -> DataTypeMetadata:
    """Build a DataTypeMetadata from mixed-key items.

    Centralizes the key → field mapping used by both the generic
    `type[key=value]` metadata form and the parameterized DECIMAL /
    VARCHAR branches. Unrecognized keys land in ``extras``, positional
    string values go to ``flags``, and positional non-string values go
    into ``extras`` under ``arg_N``.
    """
    extras: dict[str, object] = {}
    flags: list[str] = []

    nullable: bool | None = None
    ordered: bool | None = None
    sorted_: bool | None = None
    timezone: str | None = None
    unit: str | None = None
    encoding: str | None = None
    format_: str | None = None
    length: int | None = None
    precision: int | None = None
    scale: int | None = None
    byte_size: int | None = None

    for key, value in items:
        if key is None:
            if isinstance(value, str):
                flags.append(value)
            else:
                extras[f"arg_{len(extras)}"] = value
            continue

        low = key.lower()

        if low == "nullable" and isinstance(value, bool):
            nullable = value
        elif low == "ordered" and isinstance(value, bool):
            ordered = value
        elif low == "sorted" and isinstance(value, bool):
            sorted_ = value
        elif low in {"tz", "timezone"} and isinstance(value, str):
            timezone = value
        elif low == "unit" and isinstance(value, str):
            unit = value
        elif low in {"encoding", "codec"} and isinstance(value, str):
            encoding = value
        elif low == "format" and isinstance(value, str):
            format_ = value
        elif low == "length" and isinstance(value, int):
            length = value
        elif low == "precision" and isinstance(value, int):
            precision = value
        elif low == "scale" and isinstance(value, int):
            scale = value
        elif low in {"byte_size", "bytes", "size"} and isinstance(value, int):
            byte_size = value
        else:
            extras[key] = value

    return DataTypeMetadata(
        nullable=nullable,
        ordered=ordered,
        sorted=sorted_,
        timezone=timezone,
        unit=unit,
        encoding=encoding,
        format=format_,
        length=length,
        precision=precision,
        scale=scale,
        byte_size=byte_size,
        flags=tuple(flags),
        extras=extras,
    )


def _matching_close(open_char: str) -> str:
    return {
        "(": ")",
        "[": "]",
        "<": ">",
    }[open_char]


def _parse_number(value: str) -> int | float:
    return float(value) if "." in value else int(value)


def _coerce_literal_value(value: object) -> object:
    if isinstance(value, ParsedDataType) and value.type_id is DataTypeId.NULL:
        return None
    return value


def _set_nullable(parsed: ParsedDataType, nullable: bool | None) -> ParsedDataType:
    if nullable is None:
        return parsed
    return ParsedDataType(
        type_id=parsed.type_id,
        metadata=replace(parsed.metadata, nullable=nullable),
        name=parsed.name,
        children=parsed.children,
    )


@lru_cache(maxsize=2048)
def _canonical_name(name: str) -> tuple[str, DataTypeId | None]:
    """Normalize a raw type name to its canonical alias + DataTypeId.

    Returns ``(canonical, None)`` for unknown names so ``parse_primary``
    can route them to the OBJECT-forward-ref fallback with the raw name
    preserved on metadata.

    Memoized — the input domain is small (handful of dialect spellings
    across every schema we see) and the function fires on every type
    head during a cold parse; the cache turns repeated normalizations
    into a single dict lookup.
    """
    low = name.strip().lower().replace(" ", "_").replace("-", "_")
    return _NAME_ALIASES.get(low, (low, None))


# ---------------------------------------------------------------------
# Parse cache.
#
# Schemas in the wild repeat the same handful of type strings across
# many fields and many rows — `string`, `bigint`, `timestamp_ntz`,
# `array<struct<...>>`, ... Memoizing the lex+parse step turns the hot
# call into a dict lookup. The cap is sized to comfortably hold a few
# realistic schemas without unbounded growth from pathological inputs
# (e.g. machine-generated DDL with embedded ids).
#
# Only successful parses are cached — :class:`functools.lru_cache`
# never stores raised exceptions, which is exactly what we want: a
# typo in the input keeps re-raising rather than getting frozen into
# the cache.
# ---------------------------------------------------------------------

_PARSE_CACHE_SIZE = 1024


@lru_cache(maxsize=_PARSE_CACHE_SIZE)
def _parse_cached(value: str) -> ParsedDataType:
    """Parse ``value`` into a :class:`ParsedDataType`, with memoization.

    Returned :class:`ParsedDataType` instances are immutable (frozen
    dataclass with slots), so sharing them across callers is safe.
    The one mutable corner — :attr:`DataTypeMetadata.extras` — must
    not be mutated in place by downstream code; treat it as read-only.
    """
    return _Parser(value).parse()


def parse_data_type(
    value: str,
    *,
    default: Any = ...,
) -> ParsedDataType:
    """Module-level alias for :meth:`ParsedDataType.parse`.

    See that method for the ``default`` semantics — passing
    :data:`...` raises on failure, anything else is a fallback.
    """
    return ParsedDataType.parse(
        value,
        default=default,
    )