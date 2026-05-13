"""``Field`` runtime API — defaults, fills, casts, and Arrow metadata.

These tests cover the parts of :class:`Field` that don't fit cleanly
into ``construction``, ``merge``, ``arrow``, or ``equals`` — mainly
the per-engine default-array / fill-nulls helpers, plus a few smoke
tests on Arrow-side metadata round-trips and the ``from_str``
nullability-suffix shorthand.
"""
from __future__ import annotations

import pandas as pd
import polars as pl
import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.primitive import IntegerType, StringType


class TestFieldDefault:

    def test_with_default_updates_metadata_backed_value(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
        )
        f = f.with_default(7)

        assert f.has_default is True
        assert f.default_value == 7
        assert f.default_arrow_scalar.as_py() == 7


class TestFieldArrowDefaults:

    def test_default_arrow_array_uses_field_default(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            default=11,
        )

        assert f.default_arrow_array(size=3).to_pylist() == [11, 11, 11]

    def test_fill_arrow_array_nulls_uses_field_default(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            default=9,
        )
        arr = pa.array([1, None, 3], type=pa.int64())

        assert f.fill_arrow_array_nulls(arr).to_pylist() == [1, 9, 3]


class TestFieldPandasHelpers:

    def test_default_pandas_series_uses_name_and_default(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            default=5,
        )

        out = f.default_pandas_series(size=3)

        assert out.name == "qty"
        assert out.tolist() == [5, 5, 5]

    def test_fill_pandas_series_nulls_uses_field_default(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            default=5,
        )
        series = pd.Series([1, None, 3], name="qty")

        assert f.fill_pandas_series_nulls(series).tolist() == [1.0, 5.0, 3.0]

    def test_cast_pandas_series_routes_through_dtype(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
        )
        series = pd.Series([1, 2, 3], name="qty")

        out = f.cast_pandas_series(series)

        assert out.name == "qty"
        assert out.tolist() == [1, 2, 3]


class TestFieldPolarsHelpers:

    def test_default_polars_series_uses_name_and_default(self) -> None:
        f = Field(
            name="book_id",
            dtype=StringType(),
            nullable=False,
            default="NA",
        )

        out = f.default_polars_series(size=2)

        assert out.name == "book_id"
        assert out.to_list() == ["NA", "NA"]

    def test_fill_polars_series_nulls_uses_field_default(self) -> None:
        f = Field(
            name="book_id",
            dtype=StringType(),
            nullable=False,
            default="NA",
        )
        series = pl.Series("book_id", ["A", None, "C"])

        assert f.fill_polars_array_nulls(series).to_list() == ["A", "NA", "C"]

    def test_cast_polars_series_routes_through_dtype(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
        )
        series = pl.Series("qty", [1, 2, 3])

        out = f.cast_polars_series(series)

        assert out.name == "qty"
        assert out.to_list() == [1, 2, 3]


class TestFieldArrowMetadata:

    def test_to_arrow_field_keeps_user_metadata_no_type_json_by_default(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            metadata={b"comment": b"quantity"},
        )

        out = f.to_arrow_field()

        assert out.name == "qty"
        assert out.nullable is False
        assert out.metadata is not None
        assert b"comment" in out.metadata
        # Arrow preserves dtype intent natively — no blob by default.
        assert b"type_json" not in out.metadata

    def test_to_arrow_field_dump_json_attaches_type_json(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            metadata={b"comment": b"quantity"},
        )

        out = f.to_arrow_field(dump_json=True)

        assert out.metadata is not None
        assert b"comment" in out.metadata
        assert b"type_json" in out.metadata


class TestFieldFromStrNullability:

    def test_bang_suffix_marks_non_nullable(self) -> None:
        f = Field.from_str("qty!: int64")

        assert f.name == "qty"
        assert f.nullable is False


class TestFieldPrettyFormat:
    """Repr coherence — every Field, flat or nested, renders as a
    uniform ``field: 'name' <dtype>[markers]`` row tree."""

    def test_flat_field_with_known_tag(self) -> None:
        f = Field(
            name="id",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            tags={"primary_key": True},
        )
        assert f.pretty_format() == "field: 'id' int64 not null [PK]"

    def test_custom_tags_surface_in_repr(self) -> None:
        # Previously: caller-defined tags (anything outside the
        # well-known PK / FK / partition_by / ... set) were silently
        # dropped from repr. They must now render as ``name=value``
        # markers so the field round-trips visibly through ``print(f)``.
        f = Field(
            name="region",
            dtype=StringType(),
            tags={"unit": "iso-3166", "description": "geo"},
        )
        formatted = f.pretty_format()
        assert formatted.startswith("field: 'region' string ")
        assert "description='geo'" in formatted
        assert "unit='iso-3166'" in formatted

    def test_boolean_custom_tag_renders_as_bare_flag(self) -> None:
        # Boolean tags round-trip through metadata as ``b"True"``;
        # render them as a bare marker (no ``=value``) to match the
        # PK / FK convention.
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            tags={"reviewed": True},
        )
        assert "[reviewed]" in f.pretty_format()

    def test_nested_struct_walks_inner_fields(self) -> None:
        # Nested types render header + children inline at level + 1.
        # No more ``struct<...>`` bracket frame around the children
        # when the wrapping ``Field`` is the print root — they sit
        # in the same row tree as their parent.
        f = Field(
            name="row",
            dtype=__import__("yggdrasil.data.types.nested", fromlist=["StructType"]).StructType(
                fields=(
                    Field(name="id", dtype=IntegerType(byte_size=8, signed=True), nullable=False),
                    Field(name="email", dtype=StringType()),
                )
            ),
        )
        formatted = f.pretty_format()
        assert formatted.startswith("field: 'row' struct")
        # children indented one level (2 spaces) and using the same
        # ``field: 'name' <dtype>`` row shape.
        assert "\n  field: 'id' int64 not null" in formatted
        assert "\n  field: 'email' string" in formatted
        # No legacy bracket frame.
        assert "struct<" not in formatted
        assert ">" not in formatted
