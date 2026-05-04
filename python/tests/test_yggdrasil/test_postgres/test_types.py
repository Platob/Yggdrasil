"""Pure-unit tests for the Postgres ↔ Arrow type mapping."""

from __future__ import annotations

import pyarrow as pa

from yggdrasil.postgres.types import (
    arrow_schema_to_postgres_columns,
    arrow_to_postgres_field,
    arrow_to_postgres_type,
    postgres_to_arrow_field,
    postgres_to_arrow_type,
)


class TestArrowToPostgres:
    def test_primitive_ints(self) -> None:
        assert arrow_to_postgres_type(pa.int8()) == "smallint"
        assert arrow_to_postgres_type(pa.int16()) == "smallint"
        assert arrow_to_postgres_type(pa.int32()) == "integer"
        assert arrow_to_postgres_type(pa.int64()) == "bigint"

    def test_unsigned_promotes(self) -> None:
        # uint32 → bigint (covers the full range), uint64 → numeric.
        assert arrow_to_postgres_type(pa.uint32()) == "bigint"
        assert arrow_to_postgres_type(pa.uint64()) == "numeric(20, 0)"

    def test_floats(self) -> None:
        assert arrow_to_postgres_type(pa.float32()) == "real"
        assert arrow_to_postgres_type(pa.float64()) == "double precision"

    def test_decimal(self) -> None:
        assert arrow_to_postgres_type(pa.decimal128(10, 2)) == "numeric(10, 2)"

    def test_string_and_binary(self) -> None:
        assert arrow_to_postgres_type(pa.string()) == "text"
        assert arrow_to_postgres_type(pa.binary()) == "bytea"

    def test_timestamp_with_tz(self) -> None:
        assert (
            arrow_to_postgres_type(pa.timestamp("us", tz="UTC"))
            == "timestamp with time zone"
        )
        assert arrow_to_postgres_type(pa.timestamp("us")) == "timestamp"

    def test_nested_falls_back_to_jsonb(self) -> None:
        assert arrow_to_postgres_type(pa.list_(pa.int64())) == "jsonb"
        assert (
            arrow_to_postgres_type(pa.struct([pa.field("a", pa.int32())]))
            == "jsonb"
        )

    def test_field_renders_nullability(self) -> None:
        f = pa.field("name", pa.string(), nullable=False)
        assert arrow_to_postgres_field(f) == '"name" text NOT NULL'

    def test_schema_to_columns(self) -> None:
        schema = pa.schema([
            pa.field("id", pa.int64(), nullable=False),
            pa.field("name", pa.string(), nullable=True),
        ])
        cols = arrow_schema_to_postgres_columns(schema)
        assert cols == ['"id" bigint NOT NULL', '"name" text']


class TestPostgresToArrow:
    def test_primitives(self) -> None:
        assert postgres_to_arrow_type("integer") == pa.int32()
        assert postgres_to_arrow_type("bigint") == pa.int64()
        assert postgres_to_arrow_type("smallint") == pa.int16()
        assert postgres_to_arrow_type("boolean") == pa.bool_()
        assert postgres_to_arrow_type("real") == pa.float32()
        assert postgres_to_arrow_type("double precision") == pa.float64()
        assert postgres_to_arrow_type("text") == pa.string()
        assert postgres_to_arrow_type("bytea") == pa.binary()

    def test_short_aliases(self) -> None:
        assert postgres_to_arrow_type("int4") == pa.int32()
        assert postgres_to_arrow_type("int8") == pa.int64()
        assert postgres_to_arrow_type("float8") == pa.float64()
        assert postgres_to_arrow_type("bool") == pa.bool_()

    def test_numeric_with_precision(self) -> None:
        assert postgres_to_arrow_type("numeric(10, 2)") == pa.decimal128(10, 2)
        assert postgres_to_arrow_type("numeric(38)") == pa.decimal128(38, 0)

    def test_timestamp_variants(self) -> None:
        assert postgres_to_arrow_type("timestamp") == pa.timestamp("us")
        assert (
            postgres_to_arrow_type("timestamp with time zone")
            == pa.timestamp("us", tz="UTC")
        )
        assert (
            postgres_to_arrow_type("timestamp without time zone")
            == pa.timestamp("us")
        )

    def test_varchar_and_char(self) -> None:
        assert postgres_to_arrow_type("character varying(50)") == pa.string()
        assert postgres_to_arrow_type("character(10)") == pa.string()

    def test_array_suffix_lifts_to_list(self) -> None:
        assert postgres_to_arrow_type("integer[]") == pa.list_(pa.int32())
        assert (
            postgres_to_arrow_type("text[]") == pa.list_(pa.string())
        )

    def test_unknown_falls_back_to_string(self) -> None:
        assert postgres_to_arrow_type("hstore") == pa.string()

    def test_field_builder(self) -> None:
        f = postgres_to_arrow_field("id", "bigint", nullable=False)
        assert f == pa.field("id", pa.int64(), nullable=False)
