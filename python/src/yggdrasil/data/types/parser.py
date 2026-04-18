from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .id import DataTypeId

__all__ = [
    "DataTypeMetadata",
    "ParsedDataType",
    "Token",
    "parse_data_type",
]


def _default_integer_byte_size(name: str) -> int | None:
    mapping = {
        "byte": 1,
        "tinyint": 1,
        "i8": 1,
        "u8": 1,
        "int8": 1,
        "uint8": 1,
        "short": 2,
        "smallint": 2,
        "i16": 2,
        "u16": 2,
        "int16": 2,
        "uint16": 2,
        "int": 4,
        "integer": 4,
        "i32": 4,
        "u32": 4,
        "int32": 4,
        "uint32": 4,
        "long": 8,
        "bigint": 8,
        "i64": 8,
        "u64": 8,
        "int64": 8,
        "uint64": 8,
    }
    return mapping.get(name, 8)


def _default_float_byte_size(name: str) -> int | None:
    mapping = {
        "f16": 2,
        "float16": 2,
        "half": 2,
        "f32": 4,
        "float32": 4,
        "float": 4,
        "real": 4,
        "double": 8,
        "double_precision": 8,
        "f64": 8,
        "float64": 8,
    }
    return mapping.get(name, 8)


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
    metadata: DataTypeMetadata = field(default_factory=DataTypeMetadata)

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
        raise_error: bool = True,
        default: DataTypeId = DataTypeId.OBJECT,
    ) -> "ParsedDataType":
        parser = _Parser(value, raise_error=raise_error, default=default)

        try:
            result = parser.parse()
        except ValueError as e:
            raise ValueError(
                f"Failed to parse DataType string {value!r} as a DataType: {e}"
            ) from e

        if isinstance(result, ParsedDataType):
            return result
        return cls(type_id=default, metadata=DataTypeMetadata())

    @classmethod
    def parse_type_id(
        cls,
        value: str,
        *,
        raise_error: bool = True,
        default: DataTypeId = DataTypeId.OBJECT,
    ) -> DataTypeId:
        return cls.parse(
            value,
            raise_error=raise_error,
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
    def __init__(self, text: str, *, raise_error: bool, default: DataTypeId) -> None:
        self.text = text
        self.tokens = _Lexer(text).lex()
        self.index = 0
        self.raise_error = raise_error
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
        left = self.parse_primary()

        variants = [left]
        saw_pipe = False

        while self._peek_punct("|"):
            saw_pipe = True
            self._advance()
            variants.append(self.parse_primary())

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
        if self._peek_ident_ci("optional"):
            self._advance()
            inner = self._parse_generic_single()
            return _set_nullable(inner, True)

        if self._peek_ident_ci("annotated"):
            self._advance()
            inner, extras = self._parse_annotated()
            return ParsedDataType(
                type_id=inner.type_id,
                metadata=replace(
                    inner.metadata, extras={**inner.metadata.extras, **extras}
                ),
                name=inner.name,
                children=inner.children,
            )

        if self._peek_ident_ci("union"):
            self._advance()
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
                    non_null[0], True if nullable else non_null[0].metadata.nullable
                )

            return ParsedDataType(
                type_id=DataTypeId.UNION,
                metadata=DataTypeMetadata(nullable=True if nullable else None),
                children=tuple(non_null),
            )

        if self._peek_ident_ci("literal"):
            self._advance()
            literals = self._parse_literal_list()
            return ParsedDataType(
                type_id=DataTypeId.ENUM,
                metadata=DataTypeMetadata(
                    literals=tuple(literals),
                    enum_values=tuple(v for v in literals if isinstance(v, str)),
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

        if canonical == "timestamp_with_time_zone":
            return ParsedDataType(
                type_id=DataTypeId.TIMESTAMP,
                metadata=DataTypeMetadata(timezone="with_time_zone"),
            )

        if canonical == "timestamp_without_time_zone":
            return ParsedDataType(
                type_id=DataTypeId.TIMESTAMP,
                metadata=DataTypeMetadata(timezone="without_time_zone"),
            )

        if canonical == "timestamp_ntz":
            return ParsedDataType(
                type_id=DataTypeId.TIMESTAMP,
                metadata=DataTypeMetadata(timezone="ntz"),
            )

        if canonical == "timestamp_ltz":
            return ParsedDataType(
                type_id=DataTypeId.TIMESTAMP,
                metadata=DataTypeMetadata(timezone="ltz"),
            )

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
            if self._peek_any_generic_open():
                args = self._parse_scalar_args()
                if len(args) == 2 and all(isinstance(v, int) for v in args):
                    return ParsedDataType(
                        type_id=DataTypeId.DECIMAL,
                        metadata=DataTypeMetadata(
                            precision=int(args[0]),
                            scale=int(args[1]),
                        ),
                    )
                return ParsedDataType(
                    type_id=DataTypeId.DECIMAL,
                    metadata=DataTypeMetadata(args=tuple(args)),
                )
            return ParsedDataType(type_id=DataTypeId.DECIMAL)

        if canonical in {"varchar", "char", "character_varying", "character"}:
            if self._peek_any_generic_open():
                args = self._parse_scalar_args()
                length = (
                    int(args[0])
                    if len(args) == 1 and isinstance(args[0], int)
                    else None
                )
                return ParsedDataType(
                    type_id=DataTypeId.STRING,
                    metadata=DataTypeMetadata(length=length, args=tuple(args)),
                )
            return ParsedDataType(type_id=DataTypeId.STRING)

        if dtype in {
            DataTypeId.STRING,
            DataTypeId.BINARY,
            DataTypeId.JSON,
            DataTypeId.TIME,
            DataTypeId.TIMESTAMP,
            DataTypeId.DURATION,
        }:
            if self._peek_any_generic_open():
                open_tok = self._advance()
                close_char = _matching_close(open_tok.value)
                metadata = self._parse_metadata(close_char)
                return ParsedDataType(type_id=dtype, metadata=metadata)
            return ParsedDataType(type_id=dtype)

        if dtype is DataTypeId.DICTIONARY:
            if self._peek_any_generic_open():
                open_tok = self._advance()
                close_char = _matching_close(open_tok.value)
                metadata, children = self._parse_dictionary_payload(close_char)
                return ParsedDataType(
                    type_id=DataTypeId.DICTIONARY,
                    metadata=metadata,
                    children=children,
                )
            return ParsedDataType(type_id=DataTypeId.DICTIONARY)

        if dtype is DataTypeId.GEOGRAPHY:
            if self._peek_any_generic_open():
                args = self._parse_geography_args()
                return ParsedDataType(
                    type_id=DataTypeId.GEOGRAPHY,
                    metadata=DataTypeMetadata(args=tuple(args)),
                )
            return ParsedDataType(type_id=DataTypeId.GEOGRAPHY)

        if dtype is DataTypeId.EXTENSION or dtype is None:
            if self._peek_any_generic_open():
                open_tok = self._advance()
                close_char = _matching_close(open_tok.value)
                args = tuple(self._parse_scalar_items(close_char))
                return ParsedDataType(
                    type_id=DataTypeId.EXTENSION,
                    metadata=DataTypeMetadata(
                        name=raw_name,
                        args=args,
                    ),
                    name=raw_name,
                )
            return ParsedDataType(
                type_id=DataTypeId.EXTENSION,
                metadata=DataTypeMetadata(name=raw_name),
                name=raw_name,
            )

        if dtype is DataTypeId.INTEGER:
            return ParsedDataType(
                type_id=dtype,
                metadata=DataTypeMetadata(
                    byte_size=_default_integer_byte_size(canonical)
                ),
            )

        if dtype is DataTypeId.FLOAT:
            return ParsedDataType(
                type_id=dtype,
                metadata=DataTypeMetadata(
                    byte_size=_default_float_byte_size(canonical)
                ),
            )

        return ParsedDataType(type_id=dtype)

    def _parse_type_head_name(self) -> str:
        tok = self._current()
        if tok is None or tok.kind not in {"ident", "string"}:
            return self._fail("Expected type name")

        max_parts = 4
        best_len = 1

        parts = [tok.value]
        lookahead_parts = [tok.value]

        for offset in range(1, max_parts):
            nxt = self._peek_token(offset)
            if nxt is None or nxt.kind != "ident":
                break
            lookahead_parts.append(nxt.value)
            candidate = " ".join(lookahead_parts).lower()
            if candidate in _MULTI_TOKEN_TYPE_NAMES:
                best_len = len(lookahead_parts)

        selected = []
        for _ in range(best_len):
            current = self._current()
            if current is None:
                break
            selected.append(current.value)
            self._advance()

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

    def _parse_geography_args(self) -> list[object]:
        """Parse geography-specific args like ``(OGC:CRS84, SPHERICAL)``.

        Handles compound identifiers joined by ``:`` (e.g. ``OGC:CRS84``)
        that the generic scalar parser would choke on, since ``:`` is
        normally a punctuation token.
        """
        open_tok = self._expect_any_generic_open()
        close_char = _matching_close(open_tok.value)
        items: list[object] = []

        if self._peek_punct(close_char):
            self._advance()
            return items

        while True:
            items.append(self._parse_geography_arg())
            if self._peek_punct(","):
                self._advance()
                continue
            break

        self._expect_punct(close_char)
        return items

    def _parse_geography_arg(self) -> object:
        """Parse a single geography arg, joining ``ident:ident`` into one string."""
        tok = self._current()
        if tok is None:
            return self._fail("Unexpected end of geography args")

        if tok.kind == "number":
            self._advance()
            return _parse_number(tok.value)

        if tok.kind == "string":
            self._advance()
            return tok.value

        if tok.kind == "ident":
            # Consume the identifier and any colon-separated continuations
            # so OGC:CRS84 becomes the single string "OGC:CRS84".
            self._advance()
            parts = [tok.value]
            while self._peek_punct(":"):
                self._advance()  # consume ':'
                nxt = self._current()
                if nxt is not None and nxt.kind in ("ident", "number"):
                    self._advance()
                    parts.append(nxt.value)
                else:
                    # Trailing colon with nothing after — just keep what we have.
                    parts.append("")
                    break
            compound = ":".join(parts) if len(parts) > 1 else parts[0]

            low = compound.lower()
            if low == "true":
                return True
            if low == "false":
                return False
            if low in {"none", "null", "nil"}:
                return None
            return compound

        return self._fail(f"Unexpected token in geography args: {tok.value!r}")

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
                continue
            break

        self._expect_punct(close_char)

        if all(key is None for key, _ in items):
            return DataTypeMetadata(args=tuple(value for _, value in items))

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
            flags=tuple(flags),
            extras=extras,
        )

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
        nullable: bool | None = None

        if self._peek_punct("?"):
            self._advance()
            nullable = True
        elif self._peek_punct("!"):
            self._advance()
            nullable = False

        self._expect_any_punct(":", "=")
        field_type = self.parse_type()
        return ParsedDataType(
            type_id=field_type.type_id,
            metadata=replace(
                field_type.metadata,
                nullable=(
                    nullable if nullable is not None else field_type.metadata.nullable
                ),
            ),
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
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _peek_token(self, offset: int) -> Token | None:
        pos = self.index + offset
        return self.tokens[pos] if pos < len(self.tokens) else None

    def _advance(self) -> Token:
        tok = self._current()
        if tok is None:
            return self._fail("Unexpected end of input")
        self.index += 1
        return tok

    def _at_end(self) -> bool:
        return self.index >= len(self.tokens)

    def _fail(self, message: str) -> Any:
        if self.raise_error:
            raise ValueError(message)
        return ParsedDataType(type_id=self.default, metadata=DataTypeMetadata())


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


def _canonical_name(name: str) -> tuple[str, DataTypeId | None]:
    low = name.strip().lower().replace(" ", "_").replace("-", "_")

    aliases: dict[str, tuple[str, DataTypeId | None]] = {
        "object": ("object", DataTypeId.OBJECT),
        "any": ("object", DataTypeId.OBJECT),
        "variant": ("object", DataTypeId.OBJECT),
        "none": ("none", DataTypeId.NULL),
        "null": ("none", DataTypeId.NULL),
        "nil": ("none", DataTypeId.NULL),
        "bool": ("bool", DataTypeId.BOOL),
        "boolean": ("bool", DataTypeId.BOOL),
        "int": ("int", DataTypeId.INTEGER),
        "integer": ("integer", DataTypeId.INTEGER),
        "bigint": ("bigint", DataTypeId.INTEGER),
        "smallint": ("smallint", DataTypeId.INTEGER),
        "tinyint": ("tinyint", DataTypeId.INTEGER),
        "byte": ("byte", DataTypeId.INTEGER),
        "short": ("short", DataTypeId.INTEGER),
        "long": ("long", DataTypeId.INTEGER),
        "i8": ("i8", DataTypeId.INTEGER),
        "i16": ("i16", DataTypeId.INTEGER),
        "i32": ("i32", DataTypeId.INTEGER),
        "i64": ("i64", DataTypeId.INTEGER),
        "u8": ("u8", DataTypeId.INTEGER),
        "u16": ("u16", DataTypeId.INTEGER),
        "u32": ("u32", DataTypeId.INTEGER),
        "u64": ("u64", DataTypeId.INTEGER),
        "int8": ("int8", DataTypeId.INTEGER),
        "int16": ("int16", DataTypeId.INTEGER),
        "int32": ("int32", DataTypeId.INTEGER),
        "int64": ("int64", DataTypeId.INTEGER),
        "uint8": ("uint8", DataTypeId.INTEGER),
        "uint16": ("uint16", DataTypeId.INTEGER),
        "uint32": ("uint32", DataTypeId.INTEGER),
        "uint64": ("uint64", DataTypeId.INTEGER),
        "float": ("float", DataTypeId.FLOAT),
        "double": ("double", DataTypeId.FLOAT),
        "double_precision": ("double_precision", DataTypeId.FLOAT),
        "real": ("real", DataTypeId.FLOAT),
        "f16": ("f16", DataTypeId.FLOAT),
        "f32": ("f32", DataTypeId.FLOAT),
        "f64": ("f64", DataTypeId.FLOAT),
        "float16": ("float16", DataTypeId.FLOAT),
        "float32": ("float32", DataTypeId.FLOAT),
        "float64": ("float64", DataTypeId.FLOAT),
        "half": ("half", DataTypeId.FLOAT),
        "decimal": ("decimal", DataTypeId.DECIMAL),
        "numeric": ("decimal", DataTypeId.DECIMAL),
        "date": ("date", DataTypeId.DATE),
        "time": ("time", DataTypeId.TIME),
        "timestamp": ("timestamp", DataTypeId.TIMESTAMP),
        "datetime": ("timestamp", DataTypeId.TIMESTAMP),
        "timestamp_with_time_zone": ("timestamp_with_time_zone", DataTypeId.TIMESTAMP),
        "timestamp_without_time_zone": (
            "timestamp_without_time_zone",
            DataTypeId.TIMESTAMP,
        ),
        "timestamp_ntz": ("timestamp_ntz", DataTypeId.TIMESTAMP),
        "timestamp_ltz": ("timestamp_ltz", DataTypeId.TIMESTAMP),
        "duration": ("duration", DataTypeId.DURATION),
        "interval": ("duration", DataTypeId.DURATION),
        "timedelta": ("duration", DataTypeId.DURATION),
        "binary": ("binary", DataTypeId.BINARY),
        "bytes": ("binary", DataTypeId.BINARY),
        "bytea": ("binary", DataTypeId.BINARY),
        "blob": ("binary", DataTypeId.BINARY),
        "string": ("string", DataTypeId.STRING),
        "str": ("string", DataTypeId.STRING),
        "text": ("string", DataTypeId.STRING),
        "varchar": ("varchar", DataTypeId.STRING),
        "char": ("char", DataTypeId.STRING),
        "character_varying": ("character_varying", DataTypeId.STRING),
        "character": ("character", DataTypeId.STRING),
        "array": ("array", DataTypeId.ARRAY),
        "list": ("array", DataTypeId.ARRAY),
        "set": ("set", DataTypeId.ARRAY),
        "frozenset": ("frozenset", DataTypeId.ARRAY),
        "map": ("map", DataTypeId.MAP),
        "dict": ("map", DataTypeId.MAP),
        "mapping": ("map", DataTypeId.MAP),
        "struct": ("struct", DataTypeId.STRUCT),
        "row": ("struct", DataTypeId.STRUCT),
        "record": ("struct", DataTypeId.STRUCT),
        "tuple": ("tuple", DataTypeId.STRUCT),
        "union": ("union", DataTypeId.UNION),
        "json": ("json", DataTypeId.JSON),
        "enum": ("enum", DataTypeId.ENUM),
        "literal": ("literal", DataTypeId.ENUM),
        "dictionary": ("dictionary", DataTypeId.DICTIONARY),
        "categorical": ("dictionary", DataTypeId.DICTIONARY),
        "geography": ("geography", DataTypeId.GEOGRAPHY),
        "geo": ("geography", DataTypeId.GEOGRAPHY),
        "geozone": ("geography", DataTypeId.GEOGRAPHY),
        "geolocation": ("geography", DataTypeId.GEOGRAPHY),
        "udd": ("udd", DataTypeId.EXTENSION),
        "user_defined": ("udd", DataTypeId.EXTENSION),
        "user_defined_datatype": ("udd", DataTypeId.EXTENSION),
        "custom": ("udd", DataTypeId.EXTENSION),
        "optional": ("optional", None),
        "annotated": ("annotated", None),
    }

    return aliases.get(low, (low, None))


def parse_data_type(
    value: str,
    *,
    raise_error: bool = True,
    default: DataTypeId = DataTypeId.OBJECT,
) -> ParsedDataType:
    return ParsedDataType.parse(
        value,
        raise_error=raise_error,
        default=default,
    )
