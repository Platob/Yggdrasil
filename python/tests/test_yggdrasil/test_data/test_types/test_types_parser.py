"""Behaviors of :mod:`yggdrasil.data.types.parser`.

The parser is the front door for ``DataType.from_str`` and
``DataType.from_dict`` (via the JSON branch). It accepts every common
type-string dialect the codebase encounters in the wild — Arrow,
Databricks SQL, Python typing, generic SQL — and produces a
:class:`ParsedDataType` AST that the rest of the type system reads.

Tests are grouped by the dialect / surface under test:

* aliases — Arrow / SQL primitive name variants.
* temporal fast paths — ``timestamp_us``, ``time64_ns``, etc.
* Python syntax — ``list[T]``, ``Optional``, unions, ``Annotated``.
* literals / enums.
* SQL syntax — Databricks ``MAP<...>`` / ``STRUCT<...>``.
* metadata — bracket-form ``[unit=…, nullable=…]`` and the suffix
  shorthand (``?`` / ``!`` / ``not null``).
* edge cases — empty input, unbalanced brackets, unknown types.
* type-id matrix — one parametrized check per top-level type.
* tree access — ``.key`` / ``.value`` / ``.item`` / ``.children``.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.parser import (
    DataTypeMetadata,
    ParsedDataType,
    _parse_cached,
    parse_data_type,
)


# ---------------------------------------------------------------------------
# Aliases — primitive name variants
# ---------------------------------------------------------------------------


class TestParserAliases:

    @pytest.mark.parametrize(
        "name,expected_id",
        [
            # Sized aliases route to the specialized type ids — generic
            # ``INTEGER`` / ``FLOAT`` are reserved for unknown widths.
            ("int", DataTypeId.INT32),
            ("bigint", DataTypeId.INT64),
            ("double precision", DataTypeId.FLOAT64),
            ("bytea", DataTypeId.BINARY),
            ("json", DataTypeId.BJSON),
            ("sjson", DataTypeId.SJSON),
            ("jsonb", DataTypeId.BJSON),
            ("timestamp_ntz", DataTypeId.TIMESTAMP),
        ],
    )
    def test_simple_aliases(self, name: str, expected_id: DataTypeId) -> None:
        assert ParsedDataType.parse_type_id(name) is expected_id

    @pytest.mark.parametrize(
        "expr,expected_id,byte_size",
        [
            ("int8",   DataTypeId.INT8,   1),
            ("int16",  DataTypeId.INT16,  2),
            ("int32",  DataTypeId.INT32,  4),
            ("int64",  DataTypeId.INT64,  8),
            ("uint8",  DataTypeId.UINT8,  1),
            ("uint16", DataTypeId.UINT16, 2),
            ("uint32", DataTypeId.UINT32, 4),
            ("uint64", DataTypeId.UINT64, 8),
            ("INT64",  DataTypeId.INT64,  8),
        ],
    )
    def test_sized_int_aliases(
        self, expr: str, expected_id: DataTypeId, byte_size: int
    ) -> None:
        assert parse_data_type(expr) == ParsedDataType(
            expected_id, DataTypeMetadata(byte_size=byte_size)
        )

    @pytest.mark.parametrize(
        "expr,expected_id,byte_size",
        [
            ("float16", DataTypeId.FLOAT16, 2),
            ("float32", DataTypeId.FLOAT32, 4),
            ("float64", DataTypeId.FLOAT64, 8),
            ("FLOAT64", DataTypeId.FLOAT64, 8),
        ],
    )
    def test_sized_float_aliases(
        self, expr: str, expected_id: DataTypeId, byte_size: int
    ) -> None:
        assert parse_data_type(expr) == ParsedDataType(
            expected_id, DataTypeMetadata(byte_size=byte_size)
        )

    @pytest.mark.parametrize(
        "expr", ["bfloat16", "bf16", "BFLOAT16"]
    )
    def test_bfloat16_aliases(self, expr: str) -> None:
        # ``bfloat16`` shares storage width with IEEE half-precision and
        # routes to the ``FLOAT16`` specialized id.
        assert parse_data_type(expr) == ParsedDataType(
            DataTypeId.FLOAT16, DataTypeMetadata(byte_size=2)
        )

    @pytest.mark.parametrize(
        "expr", ["utf8", "large_utf8", "large_string"]
    )
    def test_string_aliases(self, expr: str) -> None:
        assert parse_data_type(expr).type_id is DataTypeId.STRING

    @pytest.mark.parametrize(
        "expr", ["large_binary", "fixed_size_binary"]
    )
    def test_binary_aliases(self, expr: str) -> None:
        assert parse_data_type(expr).type_id is DataTypeId.BINARY

    @pytest.mark.parametrize(
        "expr", ["large_list", "fixed_size_list"]
    )
    def test_list_aliases(self, expr: str) -> None:
        assert parse_data_type(expr).type_id is DataTypeId.ARRAY

    @pytest.mark.parametrize(
        "expr", ["list<int64>", "list[int64]", "array<int64>"]
    )
    def test_list_of_int64_round_trips(self, expr: str) -> None:
        assert parse_data_type(expr) == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(),
            children=(
                ParsedDataType(DataTypeId.INT64, DataTypeMetadata(byte_size=8)),
            ),
        )

    def test_decimal128_with_params(self) -> None:
        assert parse_data_type("decimal128(18, 4)") == ParsedDataType(
            DataTypeId.DECIMAL, DataTypeMetadata(precision=18, scale=4)
        )

    def test_decimal256_with_params(self) -> None:
        assert parse_data_type("decimal256(38, 10)") == ParsedDataType(
            DataTypeId.DECIMAL, DataTypeMetadata(precision=38, scale=10)
        )


# ---------------------------------------------------------------------------
# Temporal fast paths — concrete byte_size + unit per alias
# ---------------------------------------------------------------------------


class TestParserTemporalFastPaths:

    def test_date32(self) -> None:
        assert parse_data_type("date32") == ParsedDataType(
            DataTypeId.DATE, DataTypeMetadata(byte_size=4, unit="day")
        )

    def test_date64(self) -> None:
        assert parse_data_type("date64") == ParsedDataType(
            DataTypeId.DATE, DataTypeMetadata(byte_size=8, unit="ms")
        )

    def test_time32_default_ms(self) -> None:
        assert parse_data_type("time32") == ParsedDataType(
            DataTypeId.TIME, DataTypeMetadata(byte_size=4, unit="ms")
        )

    def test_time64_default_ns(self) -> None:
        assert parse_data_type("time64") == ParsedDataType(
            DataTypeId.TIME, DataTypeMetadata(byte_size=8, unit="ns")
        )

    @pytest.mark.parametrize(
        "expr,byte_size,unit",
        [
            ("time32_s", 4, "s"),
            ("time32_ms", 4, "ms"),
            ("time64_us", 8, "us"),
            ("time64_ns", 8, "ns"),
        ],
    )
    def test_time_with_explicit_unit_suffix(
        self, expr: str, byte_size: int, unit: str
    ) -> None:
        assert parse_data_type(expr) == ParsedDataType(
            DataTypeId.TIME,
            DataTypeMetadata(byte_size=byte_size, unit=unit),
        )

    @pytest.mark.parametrize("prefix", ["timestamp", "datetime"])
    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_timestamp_unit_suffix(self, prefix: str, unit: str) -> None:
        assert parse_data_type(f"{prefix}_{unit}") == ParsedDataType(
            DataTypeId.TIMESTAMP, DataTypeMetadata(unit=unit)
        )

    @pytest.mark.parametrize("prefix", ["duration", "interval"])
    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_duration_unit_suffix(self, prefix: str, unit: str) -> None:
        assert parse_data_type(f"{prefix}_{unit}") == ParsedDataType(
            DataTypeId.DURATION, DataTypeMetadata(unit=unit)
        )

    @pytest.mark.parametrize(
        "expr,unit",
        [
            ("interval_year_month", "year_month"),
            ("interval_day_time", "day_time"),
            ("interval_month_day_nano", "month_day_nano"),
        ],
    )
    def test_interval_compound_units(self, expr: str, unit: str) -> None:
        assert parse_data_type(expr) == ParsedDataType(
            DataTypeId.DURATION, DataTypeMetadata(unit=unit)
        )

    def test_fast_paths_nest_inside_array(self) -> None:
        assert parse_data_type("list<timestamp_ms>") == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(),
            children=(
                ParsedDataType(DataTypeId.TIMESTAMP, DataTypeMetadata(unit="ms")),
            ),
        )

    def test_fast_paths_nest_inside_struct(self) -> None:
        assert parse_data_type("struct<t:timestamp_us, d:date32>") == ParsedDataType(
            DataTypeId.STRUCT,
            DataTypeMetadata(),
            children=(
                ParsedDataType(
                    DataTypeId.TIMESTAMP,
                    DataTypeMetadata(unit="us"),
                    name="t",
                ),
                ParsedDataType(
                    DataTypeId.DATE,
                    DataTypeMetadata(byte_size=4, unit="day"),
                    name="d",
                ),
            ),
        )

    def test_timestamp_ntz_alias(self) -> None:
        assert parse_data_type("timestamp_ntz") == ParsedDataType(
            DataTypeId.TIMESTAMP, DataTypeMetadata(timezone="ntz")
        )

    def test_timestamp_with_time_zone_alias(self) -> None:
        assert parse_data_type("timestamp with time zone") == ParsedDataType(
            DataTypeId.TIMESTAMP, DataTypeMetadata(timezone="with_time_zone")
        )

    def test_parse_data_type_function_delegates(self) -> None:
        assert parse_data_type("int") == ParsedDataType(
            DataTypeId.INT32, DataTypeMetadata(byte_size=4)
        )

    def test_variant_normalizes_to_object(self) -> None:
        assert ParsedDataType.parse("variant") == ParsedDataType(
            DataTypeId.OBJECT, DataTypeMetadata()
        )


# ---------------------------------------------------------------------------
# Python typing syntax
# ---------------------------------------------------------------------------


class TestParserPythonSyntax:

    def test_list_int(self) -> None:
        assert ParsedDataType.parse("list[int]") == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(),
            children=(
                ParsedDataType(DataTypeId.INT32, DataTypeMetadata(byte_size=4)),
            ),
        )

    def test_dict_str_int(self) -> None:
        assert ParsedDataType.parse("dict[str, int]") == ParsedDataType(
            DataTypeId.MAP,
            DataTypeMetadata(),
            children=(
                ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                ParsedDataType(DataTypeId.INT32, DataTypeMetadata(byte_size=4)),
            ),
        )

    def test_tuple_fixed_becomes_struct_with_positional_names(self) -> None:
        assert ParsedDataType.parse("tuple[int, str]") == ParsedDataType(
            DataTypeId.STRUCT,
            DataTypeMetadata(ordered=True, extras={"container": "tuple"}),
            children=(
                ParsedDataType(
                    DataTypeId.INT32,
                    DataTypeMetadata(byte_size=4),
                    name="f0",
                ),
                ParsedDataType(
                    DataTypeId.STRING, DataTypeMetadata(), name="f1"
                ),
            ),
        )

    def test_set_normalises_to_array(self) -> None:
        assert ParsedDataType.parse("set[str]") == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(ordered=False, extras={"container": "set"}),
            children=(ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),),
        )

    def test_optional_marks_inner_nullable(self) -> None:
        assert ParsedDataType.parse("Optional[int]") == ParsedDataType(
            DataTypeId.INT32,
            DataTypeMetadata(nullable=True, byte_size=4),
        )

    def test_pipe_union_with_none_marks_nullable(self) -> None:
        assert ParsedDataType.parse("int | None") == ParsedDataType(
            DataTypeId.INT32,
            DataTypeMetadata(nullable=True, byte_size=4),
        )

    def test_pipe_union_with_three_children(self) -> None:
        assert ParsedDataType.parse("int | str | None") == ParsedDataType(
            DataTypeId.UNION,
            DataTypeMetadata(nullable=True),
            children=(
                ParsedDataType(DataTypeId.INT32, DataTypeMetadata(byte_size=4)),
                ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
            ),
        )

    def test_generic_union(self) -> None:
        assert ParsedDataType.parse("Union[int, str]") == ParsedDataType(
            DataTypeId.UNION,
            DataTypeMetadata(),
            children=(
                ParsedDataType(DataTypeId.INT32, DataTypeMetadata(byte_size=4)),
                ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
            ),
        )

    def test_annotated_preserves_inner_type_and_extras(self) -> None:
        assert ParsedDataType.parse('Annotated[str, "pk", "indexed"]') == ParsedDataType(
            DataTypeId.STRING,
            DataTypeMetadata(
                extras={"annotation_1": "pk", "annotation_2": "indexed"}
            ),
        )


# ---------------------------------------------------------------------------
# Literals & enums
# ---------------------------------------------------------------------------


class TestParserLiteralsAndEnums:

    def test_literal_with_strings(self) -> None:
        assert ParsedDataType.parse('Literal["a", "b"]') == ParsedDataType(
            DataTypeId.ENUM,
            DataTypeMetadata(literals=("a", "b"), enum_values=("a", "b")),
        )

    def test_literal_with_mixed_types(self) -> None:
        assert ParsedDataType.parse("Literal[1, 'x', true, None]") == ParsedDataType(
            DataTypeId.ENUM,
            DataTypeMetadata(
                literals=(1, "x", True, None),
                enum_values=("x",),
            ),
        )

    def test_enum_function_single(self) -> None:
        assert ParsedDataType.parse("enum('a')") == ParsedDataType(
            DataTypeId.ENUM,
            DataTypeMetadata(literals=("a",), enum_values=("a",)),
        )

    def test_enum_function_multiple(self) -> None:
        assert ParsedDataType.parse("enum('a', 'b', 'c')") == ParsedDataType(
            DataTypeId.ENUM,
            DataTypeMetadata(
                literals=("a", "b", "c"),
                enum_values=("a", "b", "c"),
            ),
        )


# ---------------------------------------------------------------------------
# SQL syntax — Databricks ``ARRAY<>`` / ``MAP<>`` / ``STRUCT<>``
# ---------------------------------------------------------------------------


class TestParserSQLSyntax:

    def test_decimal_python_style(self) -> None:
        assert ParsedDataType.parse("decimal(18, 4)") == ParsedDataType(
            DataTypeId.DECIMAL, DataTypeMetadata(precision=18, scale=4)
        )

    def test_decimal_sql_numeric(self) -> None:
        assert ParsedDataType.parse("numeric(38, 12)") == ParsedDataType(
            DataTypeId.DECIMAL, DataTypeMetadata(precision=38, scale=12)
        )

    def test_varchar_with_length(self) -> None:
        assert ParsedDataType.parse("varchar(255)") == ParsedDataType(
            DataTypeId.STRING, DataTypeMetadata(length=255, args=(255,))
        )

    def test_character_varying_with_length(self) -> None:
        assert ParsedDataType.parse("character varying(1024)") == ParsedDataType(
            DataTypeId.STRING, DataTypeMetadata(length=1024, args=(1024,))
        )

    def test_timestamp_with_time_zone(self) -> None:
        assert ParsedDataType.parse("timestamp with time zone") == ParsedDataType(
            DataTypeId.TIMESTAMP, DataTypeMetadata(timezone="with_time_zone")
        )

    def test_timestamp_without_time_zone(self) -> None:
        assert ParsedDataType.parse("timestamp without time zone") == ParsedDataType(
            DataTypeId.TIMESTAMP, DataTypeMetadata(timezone="without_time_zone")
        )

    def test_timestamp_ntz(self) -> None:
        assert ParsedDataType.parse("timestamp_ntz") == ParsedDataType(
            DataTypeId.TIMESTAMP, DataTypeMetadata(timezone="ntz")
        )

    def test_array_of_string(self) -> None:
        assert ParsedDataType.parse("array<string>") == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(),
            children=(ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),),
        )

    def test_map_string_int(self) -> None:
        assert ParsedDataType.parse("map<string, int>") == ParsedDataType(
            DataTypeId.MAP,
            DataTypeMetadata(),
            children=(
                ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                ParsedDataType(DataTypeId.INT32, DataTypeMetadata(byte_size=4)),
            ),
        )

    def test_struct_simple(self) -> None:
        assert ParsedDataType.parse("struct<a:int,b:string>") == ParsedDataType(
            DataTypeId.STRUCT,
            DataTypeMetadata(),
            children=(
                ParsedDataType(
                    DataTypeId.INT32,
                    DataTypeMetadata(byte_size=4),
                    name="a",
                ),
                ParsedDataType(
                    DataTypeId.STRING, DataTypeMetadata(), name="b"
                ),
            ),
        )

    def test_struct_with_quoted_names_and_nullability_suffixes(self) -> None:
        assert ParsedDataType.parse(
            'struct<"price"?:decimal(18,6),`ts`!:timestamp_ntz>'
        ) == ParsedDataType(
            DataTypeId.STRUCT,
            DataTypeMetadata(),
            children=(
                ParsedDataType(
                    DataTypeId.DECIMAL,
                    DataTypeMetadata(precision=18, scale=6, nullable=True),
                    name="price",
                ),
                ParsedDataType(
                    DataTypeId.TIMESTAMP,
                    DataTypeMetadata(timezone="ntz", nullable=False),
                    name="ts",
                ),
            ),
        )

    def test_struct_field_not_null_phrase(self) -> None:
        parsed = ParsedDataType.parse(
            "STRUCT<scheme: STRING NOT NULL, userinfo: STRING, "
            "host: STRING NOT NULL, port: INT, path: STRING NOT NULL, "
            "query: STRING, fragment: STRING>"
        )
        assert parsed.type_id is DataTypeId.STRUCT
        names_nullable = [
            (c.name, c.metadata.nullable) for c in parsed.children
        ]
        assert names_nullable == [
            ("scheme", False),
            ("userinfo", None),
            ("host", False),
            ("port", None),
            ("path", False),
            ("query", None),
            ("fragment", None),
        ]

    def test_struct_field_non_null_phrase(self) -> None:
        parsed = ParsedDataType.parse("struct<a: int NON NULL, b: string>")
        assert [(c.name, c.metadata.nullable) for c in parsed.children] == [
            ("a", False),
            ("b", None),
        ]

    def test_recursive_nested_expression(self) -> None:
        assert ParsedDataType.parse(
            "array<map<string, struct<a:int,b:array<string>>>> | null"
        ) == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(nullable=True),
            children=(
                ParsedDataType(
                    DataTypeId.MAP,
                    DataTypeMetadata(),
                    children=(
                        ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                        ParsedDataType(
                            DataTypeId.STRUCT,
                            DataTypeMetadata(),
                            children=(
                                ParsedDataType(
                                    DataTypeId.INT32,
                                    DataTypeMetadata(byte_size=4),
                                    name="a",
                                ),
                                ParsedDataType(
                                    DataTypeId.ARRAY,
                                    DataTypeMetadata(),
                                    name="b",
                                    children=(
                                        ParsedDataType(
                                            DataTypeId.STRING, DataTypeMetadata()
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    def test_dictionary_named_metadata(self) -> None:
        assert ParsedDataType.parse(
            "dictionary[index=int, value=string, ordered=true]"
        ) == ParsedDataType(
            DataTypeId.DICTIONARY,
            DataTypeMetadata(ordered=True),
            children=(
                ParsedDataType(DataTypeId.INT32, DataTypeMetadata(byte_size=4)),
                ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
            ),
        )

    def test_dictionary_positional_args(self) -> None:
        assert ParsedDataType.parse("dictionary[int, string]") == ParsedDataType(
            DataTypeId.DICTIONARY,
            DataTypeMetadata(),
            children=(
                ParsedDataType(DataTypeId.INT32, DataTypeMetadata(byte_size=4)),
                ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
            ),
        )


# ---------------------------------------------------------------------------
# Metadata — bracket form and suffix shorthand
# ---------------------------------------------------------------------------


class TestParserMetadata:

    def test_string_metadata_with_quoted_values(self) -> None:
        assert ParsedDataType.parse(
            "string[encoding='utf8', format=`email`, nullable=true]"
        ) == ParsedDataType(
            DataTypeId.STRING,
            DataTypeMetadata(encoding="utf8", format="email", nullable=True),
        )

    def test_binary_metadata(self) -> None:
        assert ParsedDataType.parse("binary[encoding=base64]") == ParsedDataType(
            DataTypeId.BINARY, DataTypeMetadata(encoding="base64")
        )

    def test_time_metadata(self) -> None:
        assert ParsedDataType.parse("time[unit=ms, nullable=false]") == ParsedDataType(
            DataTypeId.TIME, DataTypeMetadata(unit="ms", nullable=False)
        )

    def test_duration_metadata(self) -> None:
        assert ParsedDataType.parse("interval[unit=us]") == ParsedDataType(
            DataTypeId.DURATION, DataTypeMetadata(unit="us")
        )

    def test_timestamp_metadata_full(self) -> None:
        assert ParsedDataType.parse(
            'timestamp[tz="UTC", unit="ns", ordered=true, nullable=true]'
        ) == ParsedDataType(
            DataTypeId.TIMESTAMP,
            DataTypeMetadata(
                timezone="UTC", unit="ns", ordered=True, nullable=True
            ),
        )

    def test_metadata_identifier_values_treated_as_strings(self) -> None:
        assert ParsedDataType.parse(
            "timestamp[tz=UTC, unit=ms, encoding=base64]"
        ) == ParsedDataType(
            DataTypeId.TIMESTAMP,
            DataTypeMetadata(timezone="UTC", unit="ms", encoding="base64"),
        )

    @pytest.mark.parametrize(
        "expr,expected_nullable",
        [
            ("string not null", False),
            ("string non null", False),
            ("string?", True),
            ("string!", False),
        ],
    )
    def test_nullability_suffixes(
        self, expr: str, expected_nullable: bool
    ) -> None:
        assert ParsedDataType.parse(expr) == ParsedDataType(
            DataTypeId.STRING, DataTypeMetadata(nullable=expected_nullable)
        )

    def test_outer_pipe_None_overrides_inner_metadata(self) -> None:
        assert ParsedDataType.parse(
            "timestamp[nullable=false] | None"
        ) == ParsedDataType(
            DataTypeId.TIMESTAMP, DataTypeMetadata(nullable=True)
        )


# ---------------------------------------------------------------------------
# Edge cases & error paths
# ---------------------------------------------------------------------------


class TestParserEdgeCases:

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ParsedDataType.parse("")

    def test_empty_with_default_returns_default(self) -> None:
        assert ParsedDataType.parse(
            "", default=DataTypeId.NULL
        ) == ParsedDataType(DataTypeId.NULL, DataTypeMetadata())

    def test_invalid_with_default_returns_default(self) -> None:
        assert ParsedDataType.parse(
            "map<string>", default=DataTypeId.NULL
        ) == ParsedDataType(DataTypeId.NULL, DataTypeMetadata())

    def test_invalid_with_object_default_returns_object(self) -> None:
        assert ParsedDataType.parse(
            "map<string>", default=DataTypeId.OBJECT
        ) == ParsedDataType(DataTypeId.OBJECT, DataTypeMetadata())

    def test_invalid_with_string_default_falls_back_to_object(self) -> None:
        # Anything that isn't a DataTypeId / ParsedDataType / Ellipsis
        # collapses to OBJECT — strings used to land here, but OBJECT
        # is the safer "I don't know what this is" type.
        assert ParsedDataType.parse(
            "map<string>", default=None
        ) == ParsedDataType(DataTypeId.OBJECT, DataTypeMetadata())

    def test_map_requires_two_children(self) -> None:
        with pytest.raises(ValueError, match="exactly two"):
            ParsedDataType.parse("map<string>")

    def test_unbalanced_brackets_raise(self) -> None:
        with pytest.raises(ValueError):
            ParsedDataType.parse("array<map<string, int>")

    def test_unterminated_string_literal_raises(self) -> None:
        with pytest.raises(ValueError):
            ParsedDataType.parse('Literal["x]')

    def test_trailing_tokens_raise(self) -> None:
        with pytest.raises(ValueError, match="Unexpected trailing tokens"):
            ParsedDataType.parse("int garbage")

    def test_unknown_type_falls_back_to_object(self) -> None:
        assert ParsedDataType.parse("my_custom_type") == ParsedDataType(
            DataTypeId.OBJECT,
            DataTypeMetadata(name="my_custom_type"),
            name="my_custom_type",
        )

    def test_unknown_type_inside_map_falls_back_to_object(self) -> None:
        assert ParsedDataType.parse(
            "map<string, my_custom_type(point, 4326)>"
        ) == ParsedDataType(
            DataTypeId.MAP,
            DataTypeMetadata(),
            children=(
                ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                ParsedDataType(
                    DataTypeId.OBJECT,
                    DataTypeMetadata(name="my_custom_type"),
                    name="my_custom_type",
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Type-id matrix — every top-level type promotes correctly
# ---------------------------------------------------------------------------


class TestParserTypeIdMatrix:

    @pytest.mark.parametrize(
        "expr,expected_id",
        [
            ("bool", DataTypeId.BOOL),
            # Sized aliases now resolve to specialized ids.
            ("integer", DataTypeId.INT32),
            ("float", DataTypeId.FLOAT32),
            ("date", DataTypeId.DATE),
            ("time", DataTypeId.TIME),
            ("timestamp", DataTypeId.TIMESTAMP),
            ("interval", DataTypeId.DURATION),
            ("bytes", DataTypeId.BINARY),
            ("text", DataTypeId.STRING),
            ("array<int>", DataTypeId.ARRAY),
            ("map<string, int>", DataTypeId.MAP),
            ("struct<a:int>", DataTypeId.STRUCT),
            ("json", DataTypeId.BJSON),
            ("sjson", DataTypeId.SJSON),
            ("enum('a')", DataTypeId.ENUM),
            ("object", DataTypeId.OBJECT),
        ],
    )
    def test_top_level_type_ids(self, expr: str, expected_id: DataTypeId) -> None:
        assert ParsedDataType.parse(expr).type_id is expected_id


# ---------------------------------------------------------------------------
# Tree access — .key / .value / .item / .children
# ---------------------------------------------------------------------------


class TestParserTreeAccess:

    def test_map_string_string_exposes_key_and_value(self) -> None:
        parsed = parse_data_type("MAP<STRING,STRING>")

        assert parsed.type_id is DataTypeId.MAP
        assert parsed.key is not None
        assert parsed.value is not None
        assert parsed.key.type_id is DataTypeId.STRING
        assert parsed.value.type_id is DataTypeId.STRING

    def test_array_string_exposes_item(self) -> None:
        parsed = parse_data_type("ARRAY<STRING>")

        assert parsed.type_id is DataTypeId.ARRAY
        assert parsed.item is not None
        assert parsed.item.type_id is DataTypeId.STRING

    def test_struct_with_nested_children_exposes_full_tree(self) -> None:
        parsed = parse_data_type(
            "STRUCT<"
            "book: STRUCT<book_id: STRING, version: INT>, "
            "tags: ARRAY<STRING>, "
            "attrs: MAP<STRING, STRING>"
            ">"
        )

        assert parsed.type_id is DataTypeId.STRUCT
        assert parsed.children is not None
        assert len(parsed.children) == 3

        book = parsed.children[0]
        assert book.name == "book"
        assert book.type_id is DataTypeId.STRUCT
        assert book.children is not None
        assert len(book.children) == 2
        assert book.children[0].name == "book_id"
        assert book.children[0].type_id is DataTypeId.STRING
        assert book.children[1].name == "version"
        assert book.children[1].type_id is DataTypeId.INT32
        assert book.children[1].byte_size == 4

        tags = parsed.children[1]
        assert tags.name == "tags"
        assert tags.type_id is DataTypeId.ARRAY

        attrs = parsed.children[2]
        assert attrs.name == "attrs"
        assert attrs.type_id is DataTypeId.MAP
        assert attrs.key is not None
        assert attrs.key.type_id is DataTypeId.STRING
        assert attrs.value is not None
        assert attrs.value.type_id is DataTypeId.STRING


# ---------------------------------------------------------------------------
# Postfix `[]` array notation (PostgreSQL / Hive style)
# ---------------------------------------------------------------------------


class TestParserPostfixArray:

    def test_int_postfix_becomes_array(self) -> None:
        assert parse_data_type("int[]") == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(),
            children=(
                ParsedDataType(DataTypeId.INT32, DataTypeMetadata(byte_size=4)),
            ),
        )

    def test_text_postfix_becomes_array(self) -> None:
        assert parse_data_type("text[]") == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(),
            children=(ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),),
        )

    def test_string_postfix_does_not_collide_with_metadata(self) -> None:
        # `string[]` is array-of-string, NOT string with empty metadata.
        assert parse_data_type("string[]") == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(),
            children=(ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),),
        )

    def test_postfix_chained_for_multi_dim(self) -> None:
        assert parse_data_type("int[][]") == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(),
            children=(
                ParsedDataType(
                    DataTypeId.ARRAY,
                    DataTypeMetadata(),
                    children=(
                        ParsedDataType(
                            DataTypeId.INT32, DataTypeMetadata(byte_size=4)
                        ),
                    ),
                ),
            ),
        )

    def test_postfix_after_parameterized_type(self) -> None:
        assert parse_data_type("varchar(10)[]") == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(),
            children=(
                ParsedDataType(
                    DataTypeId.STRING,
                    DataTypeMetadata(length=10, args=(10,)),
                ),
            ),
        )

    def test_postfix_after_struct(self) -> None:
        parsed = parse_data_type("struct<a:int, b:string>[]")
        assert parsed.type_id is DataTypeId.ARRAY
        assert parsed.item is not None
        assert parsed.item.type_id is DataTypeId.STRUCT
        assert [c.name for c in parsed.item.children] == ["a", "b"]

    def test_postfix_inside_union_with_none(self) -> None:
        assert parse_data_type("int[] | None") == ParsedDataType(
            DataTypeId.ARRAY,
            DataTypeMetadata(nullable=True),
            children=(
                ParsedDataType(DataTypeId.INT32, DataTypeMetadata(byte_size=4)),
            ),
        )

    def test_postfix_inside_map_value(self) -> None:
        parsed = parse_data_type("map<string, int[]>")
        assert parsed.type_id is DataTypeId.MAP
        assert parsed.value is not None
        assert parsed.value.type_id is DataTypeId.ARRAY
        assert parsed.value.item is not None
        assert parsed.value.item.type_id is DataTypeId.INT32

    def test_metadata_brackets_still_work(self) -> None:
        # `[encoding=...]` is still metadata, NOT a postfix array.
        assert parse_data_type("string[encoding='utf8']") == ParsedDataType(
            DataTypeId.STRING, DataTypeMetadata(encoding="utf8")
        )


# ---------------------------------------------------------------------------
# Hive / BigQuery space-separated struct fields
# ---------------------------------------------------------------------------


class TestParserStructFieldSpaceSeparator:

    def test_struct_with_space_separated_fields(self) -> None:
        # Hive / BigQuery DDL style — no `:` between field name and type.
        assert parse_data_type("struct<a int, b string>") == ParsedDataType(
            DataTypeId.STRUCT,
            DataTypeMetadata(),
            children=(
                ParsedDataType(
                    DataTypeId.INT32,
                    DataTypeMetadata(byte_size=4),
                    name="a",
                ),
                ParsedDataType(
                    DataTypeId.STRING, DataTypeMetadata(), name="b"
                ),
            ),
        )

    def test_bigquery_uppercase_struct_no_colon(self) -> None:
        parsed = parse_data_type(
            "STRUCT<scheme STRING NOT NULL, port INT, host STRING>"
        )
        assert parsed.type_id is DataTypeId.STRUCT
        assert [(c.name, c.type_id, c.metadata.nullable) for c in parsed.children] == [
            ("scheme", DataTypeId.STRING, False),
            ("port", DataTypeId.INT32, None),
            ("host", DataTypeId.STRING, None),
        ]

    def test_space_separator_with_nested_type(self) -> None:
        parsed = parse_data_type(
            "struct<id bigint, tags array<string>, meta map<string, string>>"
        )
        assert parsed.type_id is DataTypeId.STRUCT
        names_and_ids = [(c.name, c.type_id) for c in parsed.children]
        assert names_and_ids == [
            ("id", DataTypeId.INT64),
            ("tags", DataTypeId.ARRAY),
            ("meta", DataTypeId.MAP),
        ]

    def test_colon_and_space_separators_mix(self) -> None:
        # Real-world DDL is sometimes irregular — both forms in one struct.
        parsed = parse_data_type("struct<a:int, b string>")
        assert [(c.name, c.type_id) for c in parsed.children] == [
            ("a", DataTypeId.INT32),
            ("b", DataTypeId.STRING),
        ]

    def test_missing_separator_and_no_type_still_errors(self) -> None:
        with pytest.raises(ValueError):
            parse_data_type("struct<a , b: int>")


# ---------------------------------------------------------------------------
# Trailing commas in nested type lists
# ---------------------------------------------------------------------------


class TestParserTrailingCommas:

    def test_trailing_comma_in_array(self) -> None:
        assert parse_data_type("array<int,>") == parse_data_type("array<int>")

    def test_trailing_comma_in_map(self) -> None:
        assert parse_data_type("map<string, int,>") == parse_data_type(
            "map<string, int>"
        )

    def test_trailing_comma_in_struct(self) -> None:
        assert parse_data_type("struct<a:int, b:string,>") == parse_data_type(
            "struct<a:int, b:string>"
        )

    def test_trailing_comma_in_decimal(self) -> None:
        assert parse_data_type("decimal(18, 4,)") == parse_data_type(
            "decimal(18, 4)"
        )

    def test_trailing_comma_in_enum(self) -> None:
        assert parse_data_type("enum('a', 'b',)") == parse_data_type(
            "enum('a', 'b')"
        )

    def test_trailing_comma_in_metadata(self) -> None:
        assert parse_data_type("timestamp[unit=ms, tz=UTC,]") == parse_data_type(
            "timestamp[unit=ms, tz=UTC]"
        )


# ---------------------------------------------------------------------------
# LRU cache behaviour for the parser front door
# ---------------------------------------------------------------------------


class TestParserCache:

    def test_cache_returns_identical_object(self) -> None:
        # Frozen-dataclass identity means callers that hash a parsed
        # type get cache reuse for free.
        _parse_cached.cache_clear()
        first = parse_data_type("array<map<string, struct<a:int, b:bigint>>>")
        second = parse_data_type("array<map<string, struct<a:int, b:bigint>>>")
        assert first is second

    def test_cache_records_hits(self) -> None:
        _parse_cached.cache_clear()
        parse_data_type("decimal(18, 4)")
        parse_data_type("decimal(18, 4)")
        parse_data_type("decimal(18, 4)")
        info = _parse_cached.cache_info()
        assert info.hits >= 2
        assert info.misses >= 1

    def test_cache_does_not_swallow_errors(self) -> None:
        _parse_cached.cache_clear()
        with pytest.raises(ValueError):
            parse_data_type("map<string>")
        # A second call still re-raises rather than returning a stale entry.
        with pytest.raises(ValueError):
            parse_data_type("map<string>")
        # And a fix to the input still parses normally.
        assert parse_data_type("map<string, int>").type_id is DataTypeId.MAP

    def test_cache_eviction_is_bounded(self) -> None:
        # Cap is finite — we don't grow without bound on adversarial input.
        info = _parse_cached.cache_info()
        assert info.maxsize is not None and info.maxsize > 0
