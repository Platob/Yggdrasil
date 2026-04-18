from __future__ import annotations

import unittest

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.parser import (
    DataTypeMetadata,
    ParsedDataType,
    parse_data_type,
)


class TestParserAliases(unittest.TestCase):

    def test_simple_aliases(self):
        self.assertIs(ParsedDataType.parse_type_id("int"), DataTypeId.INTEGER)
        self.assertIs(ParsedDataType.parse_type_id("bigint"), DataTypeId.INTEGER)
        self.assertIs(
            ParsedDataType.parse_type_id("double precision"), DataTypeId.FLOAT
        )
        self.assertIs(ParsedDataType.parse_type_id("bytea"), DataTypeId.BINARY)
        self.assertIs(ParsedDataType.parse_type_id("json"), DataTypeId.JSON)
        self.assertIs(
            ParsedDataType.parse_type_id("timestamp_ntz"), DataTypeId.TIMESTAMP
        )

    def test_sized_integer_aliases(self):
        cases = [
            ("int8", 1),
            ("int16", 2),
            ("int32", 4),
            ("int64", 8),
            ("uint8", 1),
            ("uint16", 2),
            ("uint32", 4),
            ("uint64", 8),
            ("INT64", 8),
        ]
        for expr, byte_size in cases:
            with self.subTest(expr=expr):
                self.assertEqual(
                    parse_data_type(expr),
                    ParsedDataType(
                        DataTypeId.INTEGER,
                        DataTypeMetadata(byte_size=byte_size),
                    ),
                )

    def test_sized_float_aliases(self):
        cases = [
            ("float16", 2),
            ("float32", 4),
            ("float64", 8),
            ("FLOAT64", 8),
        ]
        for expr, byte_size in cases:
            with self.subTest(expr=expr):
                self.assertEqual(
                    parse_data_type(expr),
                    ParsedDataType(
                        DataTypeId.FLOAT,
                        DataTypeMetadata(byte_size=byte_size),
                    ),
                )

    def test_list_of_int64(self):
        expected_child = ParsedDataType(
            DataTypeId.INTEGER, DataTypeMetadata(byte_size=8)
        )
        for expr in ("list<int64>", "list[int64]", "array<int64>"):
            with self.subTest(expr=expr):
                self.assertEqual(
                    parse_data_type(expr),
                    ParsedDataType(
                        DataTypeId.ARRAY,
                        DataTypeMetadata(),
                        children=(expected_child,),
                    ),
                )

    def test_numeric_wire_id(self):
        self.assertIs(ParsedDataType.parse_type_id("32"), DataTypeId.ARRAY)
        self.assertIs(ParsedDataType.parse_type_id("67"), DataTypeId.ENUM)

    def test_parse_data_type_function_delegates(self):
        self.assertEqual(
            parse_data_type("int"),
            ParsedDataType(DataTypeId.INTEGER, DataTypeMetadata(byte_size=4)),
        )

    def test_json_variant_normalization(self):
        self.assertEqual(
            ParsedDataType.parse("variant"),
            ParsedDataType(DataTypeId.OBJECT, DataTypeMetadata()),
        )


class TestParserPythonSyntax(unittest.TestCase):

    def test_python_list(self):
        self.assertEqual(
            ParsedDataType.parse("list[int]"),
            ParsedDataType(
                DataTypeId.ARRAY,
                DataTypeMetadata(),
                children=(
                    ParsedDataType(DataTypeId.INTEGER, DataTypeMetadata(byte_size=4)),
                ),
            ),
        )

    def test_python_dict(self):
        self.assertEqual(
            ParsedDataType.parse("dict[str, int]"),
            ParsedDataType(
                DataTypeId.MAP,
                DataTypeMetadata(),
                children=(
                    ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                    ParsedDataType(DataTypeId.INTEGER, DataTypeMetadata(byte_size=4)),
                ),
            ),
        )

    def test_python_tuple(self):
        self.assertEqual(
            ParsedDataType.parse("tuple[int, str]"),
            ParsedDataType(
                DataTypeId.STRUCT,
                DataTypeMetadata(
                    ordered=True,
                    extras={"container": "tuple"},
                ),
                children=(
                    ParsedDataType(
                        DataTypeId.INTEGER,
                        DataTypeMetadata(byte_size=4),
                        name="f0",
                    ),
                    ParsedDataType(
                        DataTypeId.STRING,
                        DataTypeMetadata(),
                        name="f1",
                    ),
                ),
            ),
        )

    def test_python_set_normalized_to_array(self):
        self.assertEqual(
            ParsedDataType.parse("set[str]"),
            ParsedDataType(
                DataTypeId.ARRAY,
                DataTypeMetadata(
                    ordered=False,
                    extras={"container": "set"},
                ),
                children=(ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),),
            ),
        )

    def test_optional(self):
        self.assertEqual(
            ParsedDataType.parse("Optional[int]"),
            ParsedDataType(
                DataTypeId.INTEGER,
                DataTypeMetadata(nullable=True, byte_size=4),
            ),
        )

    def test_union_pipe_nullable(self):
        self.assertEqual(
            ParsedDataType.parse("int | None"),
            ParsedDataType(
                DataTypeId.INTEGER,
                DataTypeMetadata(nullable=True, byte_size=4),
            ),
        )

    def test_union_pipe_multiple(self):
        self.assertEqual(
            ParsedDataType.parse("int | str | None"),
            ParsedDataType(
                DataTypeId.UNION,
                DataTypeMetadata(nullable=True),
                children=(
                    ParsedDataType(DataTypeId.INTEGER, DataTypeMetadata(byte_size=4)),
                    ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                ),
            ),
        )

    def test_union_generic(self):
        self.assertEqual(
            ParsedDataType.parse("Union[int, str]"),
            ParsedDataType(
                DataTypeId.UNION,
                DataTypeMetadata(),
                children=(
                    ParsedDataType(DataTypeId.INTEGER, DataTypeMetadata(byte_size=4)),
                    ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                ),
            ),
        )

    def test_annotated_preserves_type_and_extras(self):
        self.assertEqual(
            ParsedDataType.parse('Annotated[str, "pk", "indexed"]'),
            ParsedDataType(
                DataTypeId.STRING,
                DataTypeMetadata(
                    extras={
                        "annotation_1": "pk",
                        "annotation_2": "indexed",
                    },
                ),
            ),
        )


class TestParserLiteralsAndEnums(unittest.TestCase):

    def test_literal_strings(self):
        self.assertEqual(
            ParsedDataType.parse('Literal["a", "b"]'),
            ParsedDataType(
                DataTypeId.ENUM,
                DataTypeMetadata(
                    literals=("a", "b"),
                    enum_values=("a", "b"),
                ),
            ),
        )

    def test_literal_mixed(self):
        self.assertEqual(
            ParsedDataType.parse("Literal[1, 'x', true, None]"),
            ParsedDataType(
                DataTypeId.ENUM,
                DataTypeMetadata(
                    literals=(1, "x", True, None),
                    enum_values=("x",),
                ),
            ),
        )

    def test_enum_function_style(self):
        self.assertEqual(
            ParsedDataType.parse("enum('a')"),
            ParsedDataType(
                DataTypeId.ENUM,
                DataTypeMetadata(
                    literals=("a",),
                    enum_values=("a",),
                ),
            ),
        )

    def test_enum_function_style_multiple(self):
        self.assertEqual(
            ParsedDataType.parse("enum('a', 'b', 'c')"),
            ParsedDataType(
                DataTypeId.ENUM,
                DataTypeMetadata(
                    literals=("a", "b", "c"),
                    enum_values=("a", "b", "c"),
                ),
            ),
        )


class TestParserSQLSyntax(unittest.TestCase):

    def test_decimal_python_style(self):
        self.assertEqual(
            ParsedDataType.parse("decimal(18, 4)"),
            ParsedDataType(
                DataTypeId.DECIMAL,
                DataTypeMetadata(precision=18, scale=4),
            ),
        )

    def test_decimal_sql_numeric(self):
        self.assertEqual(
            ParsedDataType.parse("numeric(38, 12)"),
            ParsedDataType(
                DataTypeId.DECIMAL,
                DataTypeMetadata(precision=38, scale=12),
            ),
        )

    def test_varchar_length(self):
        self.assertEqual(
            ParsedDataType.parse("varchar(255)"),
            ParsedDataType(
                DataTypeId.STRING,
                DataTypeMetadata(length=255, args=(255,)),
            ),
        )

    def test_character_varying_length(self):
        self.assertEqual(
            ParsedDataType.parse("character varying(1024)"),
            ParsedDataType(
                DataTypeId.STRING,
                DataTypeMetadata(length=1024, args=(1024,)),
            ),
        )

    def test_timestamp_with_time_zone(self):
        self.assertEqual(
            ParsedDataType.parse("timestamp with time zone"),
            ParsedDataType(
                DataTypeId.TIMESTAMP,
                DataTypeMetadata(timezone="with_time_zone"),
            ),
        )

    def test_timestamp_without_time_zone(self):
        self.assertEqual(
            ParsedDataType.parse("timestamp without time zone"),
            ParsedDataType(
                DataTypeId.TIMESTAMP,
                DataTypeMetadata(timezone="without_time_zone"),
            ),
        )

    def test_timestamp_ntz(self):
        self.assertEqual(
            ParsedDataType.parse("timestamp_ntz"),
            ParsedDataType(
                DataTypeId.TIMESTAMP,
                DataTypeMetadata(timezone="ntz"),
            ),
        )

    def test_databricks_array(self):
        self.assertEqual(
            ParsedDataType.parse("array<string>"),
            ParsedDataType(
                DataTypeId.ARRAY,
                DataTypeMetadata(),
                children=(ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),),
            ),
        )

    def test_databricks_map(self):
        self.assertEqual(
            ParsedDataType.parse("map<string, int>"),
            ParsedDataType(
                DataTypeId.MAP,
                DataTypeMetadata(),
                children=(
                    ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                    ParsedDataType(DataTypeId.INTEGER, DataTypeMetadata(byte_size=4)),
                ),
            ),
        )

    def test_struct_sql_style(self):
        self.assertEqual(
            ParsedDataType.parse("struct<a:int,b:string>"),
            ParsedDataType(
                DataTypeId.STRUCT,
                DataTypeMetadata(),
                children=(
                    ParsedDataType(
                        DataTypeId.INTEGER,
                        DataTypeMetadata(byte_size=4),
                        name="a",
                    ),
                    ParsedDataType(
                        DataTypeId.STRING,
                        DataTypeMetadata(),
                        name="b",
                    ),
                ),
            ),
        )

    def test_struct_colon_with_quotes(self):
        self.assertEqual(
            ParsedDataType.parse('struct<"price"?:decimal(18,6),`ts`!:timestamp_ntz>'),
            ParsedDataType(
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
            ),
        )

    def test_recursive_nested_expression(self):
        self.assertEqual(
            ParsedDataType.parse(
                "array<map<string, struct<a:int,b:array<string>>>> | null"
            ),
            ParsedDataType(
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
                                        DataTypeId.INTEGER,
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
            ),
        )

    def test_dictionary_named_metadata(self):
        self.assertEqual(
            ParsedDataType.parse("dictionary[index=int, value=string, ordered=true]"),
            ParsedDataType(
                DataTypeId.DICTIONARY,
                DataTypeMetadata(ordered=True),
                children=(
                    ParsedDataType(DataTypeId.INTEGER, DataTypeMetadata(byte_size=4)),
                    ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                ),
            ),
        )

    def test_dictionary_positional_args(self):
        self.assertEqual(
            ParsedDataType.parse("dictionary[int, string]"),
            ParsedDataType(
                DataTypeId.DICTIONARY,
                DataTypeMetadata(),
                children=(
                    ParsedDataType(DataTypeId.INTEGER, DataTypeMetadata(byte_size=4)),
                    ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                ),
            ),
        )


class TestParserMetadata(unittest.TestCase):

    def test_string_metadata_brackets(self):
        self.assertEqual(
            ParsedDataType.parse(
                "string[encoding='utf8', format=`email`, nullable=true]"
            ),
            ParsedDataType(
                DataTypeId.STRING,
                DataTypeMetadata(encoding="utf8", format="email", nullable=True),
            ),
        )

    def test_binary_metadata(self):
        self.assertEqual(
            ParsedDataType.parse("binary[encoding=base64]"),
            ParsedDataType(
                DataTypeId.BINARY,
                DataTypeMetadata(encoding="base64"),
            ),
        )

    def test_time_metadata(self):
        self.assertEqual(
            ParsedDataType.parse("time[unit=ms, nullable=false]"),
            ParsedDataType(
                DataTypeId.TIME,
                DataTypeMetadata(unit="ms", nullable=False),
            ),
        )

    def test_duration_metadata(self):
        self.assertEqual(
            ParsedDataType.parse("interval[unit=us]"),
            ParsedDataType(
                DataTypeId.DURATION,
                DataTypeMetadata(unit="us"),
            ),
        )

    def test_timestamp_metadata(self):
        self.assertEqual(
            ParsedDataType.parse(
                'timestamp[tz="UTC", unit="ns", ordered=true, nullable=true]'
            ),
            ParsedDataType(
                DataTypeId.TIMESTAMP,
                DataTypeMetadata(
                    timezone="UTC",
                    unit="ns",
                    ordered=True,
                    nullable=True,
                ),
            ),
        )

    def test_metadata_identifier_values_are_scalar_strings(self):
        self.assertEqual(
            ParsedDataType.parse("timestamp[tz=UTC, unit=ms, encoding=base64]"),
            ParsedDataType(
                DataTypeId.TIMESTAMP,
                DataTypeMetadata(
                    timezone="UTC",
                    unit="ms",
                    encoding="base64",
                ),
            ),
        )

    def test_not_null_suffix(self):
        self.assertEqual(
            ParsedDataType.parse("string not null"),
            ParsedDataType(DataTypeId.STRING, DataTypeMetadata(nullable=False)),
        )

    def test_non_null_suffix(self):
        self.assertEqual(
            ParsedDataType.parse("string non null"),
            ParsedDataType(DataTypeId.STRING, DataTypeMetadata(nullable=False)),
        )

    def test_question_nullable_suffix(self):
        self.assertEqual(
            ParsedDataType.parse("string?"),
            ParsedDataType(DataTypeId.STRING, DataTypeMetadata(nullable=True)),
        )

    def test_bang_not_null_suffix(self):
        self.assertEqual(
            ParsedDataType.parse("string!"),
            ParsedDataType(DataTypeId.STRING, DataTypeMetadata(nullable=False)),
        )

    def test_outer_nullable_overrides_inner(self):
        self.assertEqual(
            ParsedDataType.parse("timestamp[nullable=false] | None"),
            ParsedDataType(DataTypeId.TIMESTAMP, DataTypeMetadata(nullable=True)),
        )


class TestParserEdgeCases(unittest.TestCase):

    def test_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            ParsedDataType.parse("")

    def test_empty_default_when_raise_error_false(self):
        self.assertEqual(
            ParsedDataType.parse("", raise_error=False, default=DataTypeId.NULL),
            ParsedDataType(DataTypeId.NULL, DataTypeMetadata()),
        )

    def test_raise_error_false_returns_default(self):
        self.assertEqual(
            ParsedDataType.parse(
                "map<string>", raise_error=False, default=DataTypeId.NULL
            ),
            ParsedDataType(DataTypeId.NULL, DataTypeMetadata()),
        )

    def test_invalid_map_raises(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            ParsedDataType.parse("map<string>")

    def test_unbalanced_brackets_raises(self):
        with self.assertRaises(ValueError):
            ParsedDataType.parse("array<map<string, int>")

    def test_unterminated_string_raises(self):
        with self.assertRaises(ValueError):
            ParsedDataType.parse('Literal["x]')

    def test_trailing_tokens_raises(self):
        with self.assertRaisesRegex(ValueError, "Unexpected trailing tokens"):
            ParsedDataType.parse("int garbage")

    def test_unknown_type_becomes_udd(self):
        self.assertEqual(
            ParsedDataType.parse("my_custom_type"),
            ParsedDataType(
                DataTypeId.EXTENSION,
                DataTypeMetadata(name="my_custom_type"),
                name="my_custom_type",
            ),
        )

    def test_unknown_type_recursive_becomes_udd(self):
        # geography is now a first-class type with DataTypeId.GEOGRAPHY.
        self.assertEqual(
            ParsedDataType.parse("geography(point, 4326)"),
            ParsedDataType(
                DataTypeId.GEOGRAPHY,
                DataTypeMetadata(args=("point", 4326)),
            ),
        )

    def test_unknown_nested_type_in_map(self):
        self.assertEqual(
            ParsedDataType.parse("map<string, geography(point, 4326)>"),
            ParsedDataType(
                DataTypeId.MAP,
                DataTypeMetadata(),
                children=(
                    ParsedDataType(DataTypeId.STRING, DataTypeMetadata()),
                    ParsedDataType(
                        DataTypeId.GEOGRAPHY,
                        DataTypeMetadata(args=("point", 4326)),
                    ),
                ),
            ),
        )


class TestParserTypeIdMatrix(unittest.TestCase):

    def test_type_id_param_matrix(self):
        cases = [
            ("bool", DataTypeId.BOOL),
            ("integer", DataTypeId.INTEGER),
            ("float", DataTypeId.FLOAT),
            ("date", DataTypeId.DATE),
            ("time", DataTypeId.TIME),
            ("timestamp", DataTypeId.TIMESTAMP),
            ("interval", DataTypeId.DURATION),
            ("bytes", DataTypeId.BINARY),
            ("text", DataTypeId.STRING),
            ("array<int>", DataTypeId.ARRAY),
            ("map<string, int>", DataTypeId.MAP),
            ("struct<a:int>", DataTypeId.STRUCT),
            ("json", DataTypeId.JSON),
            ("enum('a')", DataTypeId.ENUM),
            ("custom_type", DataTypeId.EXTENSION),
        ]
        for expr, expected in cases:
            with self.subTest(expr=expr):
                self.assertIs(ParsedDataType.parse(expr).type_id, expected)


class TestParserTreeAccess(unittest.TestCase):

    def test_parse_data_type_string_map_tree(self):
        parsed = parse_data_type("MAP<STRING,STRING>")

        self.assertIs(parsed.type_id, DataTypeId.MAP)
        self.assertIsNotNone(parsed.key)
        self.assertIsNotNone(parsed.value)
        self.assertIs(parsed.key.type_id, DataTypeId.STRING)
        self.assertIs(parsed.value.type_id, DataTypeId.STRING)

    def test_parse_data_type_string_array_tree(self):
        parsed = parse_data_type("ARRAY<STRING>")

        self.assertIs(parsed.type_id, DataTypeId.ARRAY)
        self.assertIsNotNone(parsed.item)
        self.assertIs(parsed.item.type_id, DataTypeId.STRING)

    def test_parse_data_type_string_struct_tree(self):
        parsed = parse_data_type(
            "STRUCT<"
            "book: STRUCT<book_id: STRING, version: INT>, "
            "tags: ARRAY<STRING>, "
            "attrs: MAP<STRING, STRING>"
            ">"
        )

        self.assertIs(parsed.type_id, DataTypeId.STRUCT)
        self.assertIsNotNone(parsed.children)
        self.assertEqual(len(parsed.children), 3)
        self.assertEqual(parsed.children[0].name, "book")
        self.assertIs(parsed.children[0].type_id, DataTypeId.STRUCT)
        self.assertIsNotNone(parsed.children[0].children)
        self.assertEqual(len(parsed.children[0].children), 2)
        self.assertEqual(parsed.children[0].children[0].name, "book_id")
        self.assertIs(parsed.children[0].children[0].type_id, DataTypeId.STRING)
        self.assertEqual(parsed.children[0].children[1].name, "version")
        self.assertIs(parsed.children[0].children[1].type_id, DataTypeId.INTEGER)
        self.assertEqual(parsed.children[0].children[1].byte_size, 4)
        self.assertEqual(parsed.children[1].name, "tags")
        self.assertIs(parsed.children[1].type_id, DataTypeId.ARRAY)
        self.assertEqual(parsed.children[2].name, "attrs")
        self.assertIs(parsed.children[2].type_id, DataTypeId.MAP)
        self.assertIsNotNone(parsed.children[2].key)
        self.assertIs(parsed.children[2].key.type_id, DataTypeId.STRING)
        self.assertIsNotNone(parsed.children[2].value)
        self.assertIs(parsed.children[2].value.type_id, DataTypeId.STRING)
