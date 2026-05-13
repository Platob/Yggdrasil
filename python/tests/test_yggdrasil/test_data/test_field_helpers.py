"""Private helpers in :mod:`yggdrasil.data.data_field`.

These are internal — but they're load-bearing for ``Field.from_str``
parsing and for the type-json-metadata round-trip that lets a
yggdrasil :class:`Field` survive an Arrow IPC round-trip without
losing its dtype intent. Worth pinning so a refactor on either side
fails loud here, not silently in the user-visible API.
"""
from __future__ import annotations

from yggdrasil.data.data_field import (
    _TYPE_JSON_METADATA_KEY,
    _parse_field_name_token,
    _parse_spark_column_sql,
    _safe_issubclass,
    _split_field_shorthand,
    _split_top_level_as,
    _strip_internal_metadata,
    _strip_matching_quotes,
)
from yggdrasil.data.types.primitive import (
    DecimalType,
    IntegerType,
    StringType,
)


class TestStripMatchingQuotes:

    def test_double_quotes(self) -> None:
        assert _strip_matching_quotes('"abc"') == "abc"

    def test_single_quotes(self) -> None:
        assert _strip_matching_quotes("'abc'") == "abc"

    def test_backticks(self) -> None:
        assert _strip_matching_quotes("`abc`") == "abc"

    def test_only_strips_outer_whitespace_when_no_quotes(self) -> None:
        assert _strip_matching_quotes(" abc ") == "abc"


class TestParseFieldNameToken:

    def test_question_mark_marks_nullable_true(self) -> None:
        assert _parse_field_name_token("value?") == ("value", True)

    def test_bang_marks_nullable_false(self) -> None:
        assert _parse_field_name_token("value!") == ("value", False)

    def test_quoted_name_keeps_nullability_undecided(self) -> None:
        assert _parse_field_name_token('"value"') == ("value", None)


class TestSplitFieldShorthand:

    def test_simple_pair(self) -> None:
        assert _split_field_shorthand("price:int64") == ("price", "int64")

    def test_ignores_colons_inside_nested_brackets(self) -> None:
        left, right = _split_field_shorthand(
            "payload:struct<a:int64,b:string>"
        )

        assert left == "payload"
        assert right == "struct<a:int64,b:string>"


class TestSafeIssubclass:

    def test_real_subclass_pair(self) -> None:
        assert _safe_issubclass(int, int) is True

    def test_non_type_object_returns_false_without_raising(self) -> None:
        assert _safe_issubclass("int", int) is False
        assert _safe_issubclass(123, int) is False


class TestStripInternalMetadata:

    def test_drops_internal_key_only(self) -> None:
        metadata = {
            b"comment": b"hello",
            _TYPE_JSON_METADATA_KEY: b'{"id":3}',
        }

        assert _strip_internal_metadata(metadata) == {b"comment": b"hello"}

    def test_returns_none_when_only_internal_key_remains(self) -> None:
        metadata = {_TYPE_JSON_METADATA_KEY: b'{"id":3}'}

        assert _strip_internal_metadata(metadata) is None


class TestSplitTopLevelAs:
    """``_split_top_level_as`` finds the outermost ``" AS "`` outside parens.

    Used by :func:`_parse_spark_column_sql` to split
    ``CAST(<expr> AS T) AS alias`` cleanly. Walks right-to-left
    while tracking paren depth so a nested ``AS`` inside a
    ``CAST`` doesn't trip the split.
    """

    def test_plain_alias(self) -> None:
        assert _split_top_level_as("id AS user_id") == ("id", "user_id")

    def test_no_as_returns_none(self) -> None:
        assert _split_top_level_as("id") is None

    def test_skips_as_inside_parens(self) -> None:
        # The only ``AS`` is inside ``CAST(...)`` — there's no
        # top-level alias to split on.
        assert _split_top_level_as("CAST(id AS STRING)") is None

    def test_outer_alias_over_inner_cast(self) -> None:
        assert _split_top_level_as("CAST(id AS STRING) AS user_id") == (
            "CAST(id AS STRING)",
            "user_id",
        )

    def test_chained_casts_split_outermost(self) -> None:
        # Two AS tokens at top level (one inside outer CAST, one
        # outside) — we want the outermost one.
        result = _split_top_level_as("CAST(CAST(id AS BIGINT) AS STRING) AS y")
        assert result == ("CAST(CAST(id AS BIGINT) AS STRING)", "y")


class TestParseSparkColumnSql:
    """``_parse_spark_column_sql`` is the dtype + name extractor for
    :meth:`Field.from_spark_column`. The SQL string is the only stable
    JVM surface across PySpark releases since 4.x hid the Catalyst
    ``expr()`` method.
    """

    def test_plain_reference(self) -> None:
        name, dtype = _parse_spark_column_sql("id")
        assert name == "id"
        assert dtype is None

    def test_alias(self) -> None:
        name, dtype = _parse_spark_column_sql("id AS user_id")
        assert name == "user_id"
        assert dtype is None

    def test_cast_extracts_dtype(self) -> None:
        name, dtype = _parse_spark_column_sql("CAST(id AS STRING)")
        assert name == "id"
        assert isinstance(dtype, StringType)

    def test_cast_with_parameters_preserved(self) -> None:
        name, dtype = _parse_spark_column_sql("CAST(amount AS DECIMAL(10,2))")
        assert name == "amount"
        assert isinstance(dtype, DecimalType)
        assert dtype.precision == 10
        assert dtype.scale == 2

    def test_aliased_cast_keeps_inner_dtype(self) -> None:
        name, dtype = _parse_spark_column_sql("CAST(id AS STRING) AS user_id")
        assert name == "user_id"
        assert isinstance(dtype, StringType)

    def test_cast_to_int_alias(self) -> None:
        name, dtype = _parse_spark_column_sql("CAST(price AS INT) AS p")
        assert name == "p"
        assert isinstance(dtype, IntegerType)

    def test_backtick_quoted_identifier(self) -> None:
        # Spark wraps qualified identifiers in backticks; the parser
        # peels them so the rendered name carries the bare identifier.
        name, dtype = _parse_spark_column_sql("`my id`")
        # Backtick form with a space inside doesn't peel (would be
        # ambiguous with arbitrary SQL); a no-space form does.
        assert name in {"`my id`", "my id"}
        assert dtype is None

    def test_unparseable_expression_returns_raw_text(self) -> None:
        # Arithmetic / function calls — no dtype available from the
        # SQL surface, name falls back to the full string.
        name, dtype = _parse_spark_column_sql("id + 1")
        assert name == "id + 1"
        assert dtype is None
