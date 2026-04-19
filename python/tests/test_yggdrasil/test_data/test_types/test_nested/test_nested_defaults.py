"""Cascading default values for nested and deeply-nested types.

Covers the Python-level ``default_pyobj`` contract plus the Arrow /
Polars / Pandas materialisation paths that build on it.  The goal is
that a non-nullable struct/list/map — however deeply nested — produces
a default value that ``pa.scalar(default, type=self.to_arrow())``
accepts, that polars' ``fill_null`` can broadcast, and that pandas'
object-dtype fill helper doesn't mistake for a per-index mapping.
"""
from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.nested.map import MapType
from yggdrasil.data.types.nested.struct import StructType
from yggdrasil.data.types.primitive import IntegerType, StringType
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


INT64 = IntegerType(byte_size=8, signed=True)


class TestNestedDefaultPyobjCascade(ArrowTestCase):
    """``default_pyobj`` must recurse through children on non-nullable."""

    def test_struct_of_struct_non_null_recurses(self) -> None:
        inner = StructType(
            fields=[Field(name="bar", dtype=INT64, nullable=False)]
        )
        outer = StructType(
            fields=[Field(name="foo", dtype=inner, nullable=False)]
        )

        self.assertEqual(outer.default_pyobj(nullable=False), {"foo": {"bar": 0}})

    def test_struct_of_struct_non_null_with_nullable_child_is_none(self) -> None:
        inner = StructType(
            fields=[Field(name="bar", dtype=INT64, nullable=False)]
        )
        outer = StructType(
            fields=[Field(name="foo", dtype=inner, nullable=True)]
        )

        self.assertEqual(outer.default_pyobj(nullable=False), {"foo": None})

    def test_struct_of_list_non_null_uses_empty_list(self) -> None:
        list_dtype = ArrayType(
            item_field=Field(name="item", dtype=INT64, nullable=True)
        )
        outer = StructType(
            fields=[Field(name="xs", dtype=list_dtype, nullable=False)]
        )

        self.assertEqual(outer.default_pyobj(nullable=False), {"xs": []})

    def test_struct_of_map_non_null_uses_empty_dict(self) -> None:
        map_dtype = MapType.from_key_value(
            key_field=Field(name="k", dtype=StringType(), nullable=False),
            value_field=Field(name="v", dtype=INT64, nullable=True),
        )
        outer = StructType(
            fields=[Field(name="m", dtype=map_dtype, nullable=False)]
        )

        self.assertEqual(outer.default_pyobj(nullable=False), {"m": {}})

    def test_explicit_field_default_wins_over_dtype_cascade(self) -> None:
        outer = StructType(
            fields=[Field(name="n", dtype=INT64, nullable=True, default=42)]
        )

        self.assertEqual(outer.default_pyobj(nullable=False), {"n": 42})

    def test_deeply_nested_struct_list_struct_list(self) -> None:
        leaf_list = ArrayType(
            item_field=Field(name="item", dtype=INT64, nullable=False)
        )
        inner = StructType(
            fields=[Field(name="b", dtype=leaf_list, nullable=False)]
        )
        outer_list = ArrayType(
            item_field=Field(name="item", dtype=inner, nullable=False)
        )
        top = StructType(
            fields=[Field(name="a", dtype=outer_list, nullable=False)]
        )

        # Variable-size lists default to [] even when non-nullable, so the
        # cascade bottoms out at an empty list without needing every leaf's
        # default.  (Fixed-size leaves are exercised below.)
        self.assertEqual(top.default_pyobj(nullable=False), {"a": []})

    def test_fixed_size_list_non_null_materialises_each_slot(self) -> None:
        dtype = ArrayType(
            item_field=Field(name="item", dtype=INT64, nullable=False),
            list_size=3,
        )

        self.assertEqual(dtype.default_pyobj(nullable=False), [0, 0, 0])

    def test_fixed_size_list_of_struct_non_null(self) -> None:
        inner = StructType(
            fields=[Field(name="x", dtype=INT64, nullable=False)]
        )
        dtype = ArrayType(
            item_field=Field(name="item", dtype=inner, nullable=False),
            list_size=2,
        )

        self.assertEqual(
            dtype.default_pyobj(nullable=False), [{"x": 0}, {"x": 0}]
        )

    def test_fixed_size_list_zero_is_empty_list(self) -> None:
        dtype = ArrayType(
            item_field=Field(name="item", dtype=INT64, nullable=True),
            list_size=0,
        )

        self.assertEqual(dtype.default_pyobj(nullable=False), [])


class TestNestedDefaultArrowScalarAndArray(ArrowTestCase):
    """The cascade must produce Arrow-accepted scalars and arrays."""

    def test_struct_of_struct_default_arrow_scalar_roundtrips(self) -> None:
        inner = StructType(
            fields=[Field(name="bar", dtype=INT64, nullable=False)]
        )
        outer = StructType(
            fields=[Field(name="foo", dtype=inner, nullable=False)]
        )

        scalar = outer.default_arrow_scalar(nullable=False)

        self.assertEqual(scalar.type, outer.to_arrow())
        self.assertEqual(scalar.as_py(), {"foo": {"bar": 0}})

    def test_fixed_size_list_default_arrow_scalar(self) -> None:
        pa = self.pa
        dtype = ArrayType(
            item_field=Field(name="item", dtype=INT64, nullable=False),
            list_size=3,
        )

        scalar = dtype.default_arrow_scalar(nullable=False)

        self.assertTrue(pa.types.is_fixed_size_list(scalar.type))
        self.assertEqual(scalar.as_py(), [0, 0, 0])

    def test_struct_of_struct_default_arrow_array_non_null(self) -> None:
        inner = StructType(
            fields=[Field(name="bar", dtype=INT64, nullable=False)]
        )
        outer = StructType(
            fields=[Field(name="foo", dtype=inner, nullable=False)]
        )

        arr = outer.default_arrow_array(nullable=False, size=3)

        self.assertEqual(len(arr), 3)
        self.assertEqual(
            arr.to_pylist(),
            [{"foo": {"bar": 0}}] * 3,
        )

    def test_array_of_struct_default_arrow_array(self) -> None:
        inner = StructType(
            fields=[Field(name="bar", dtype=INT64, nullable=False)]
        )
        dtype = ArrayType(
            item_field=Field(name="item", dtype=inner, nullable=False),
            list_size=2,
        )

        arr = dtype.default_arrow_array(nullable=False, size=2)

        self.assertEqual(
            arr.to_pylist(),
            [[{"bar": 0}, {"bar": 0}]] * 2,
        )

    def test_fill_arrow_nulls_on_struct_array(self) -> None:
        pa = self.pa
        inner = StructType(
            fields=[Field(name="bar", dtype=INT64, nullable=False)]
        )
        outer = StructType(
            fields=[Field(name="foo", dtype=inner, nullable=False)]
        )

        arr = pa.array(
            [{"foo": {"bar": 5}}, None, {"foo": {"bar": 7}}],
            type=outer.to_arrow(),
        )

        filled = outer.fill_arrow_array_nulls(arr, nullable=False)

        self.assertEqual(
            filled.to_pylist(),
            [{"foo": {"bar": 5}}, {"foo": {"bar": 0}}, {"foo": {"bar": 7}}],
        )


class TestNestedDefaultPolars(PolarsTestCase):

    def test_polars_fill_struct_nulls_uses_nested_default(self) -> None:
        pl = self.pl
        inner = StructType(
            fields=[Field(name="bar", dtype=INT64, nullable=False)]
        )
        outer = StructType(
            fields=[Field(name="foo", dtype=inner, nullable=False)]
        )

        s = pl.Series(
            "x",
            [{"foo": {"bar": 5}}, None, {"foo": {"bar": 7}}],
        )

        filled = outer.fill_polars_array_nulls(s, nullable=False)

        self.assertEqual(
            filled.to_list(),
            [{"foo": {"bar": 5}}, {"foo": {"bar": 0}}, {"foo": {"bar": 7}}],
        )


class TestNestedDefaultPandas(PandasTestCase):

    def test_pandas_fill_list_nulls_does_not_treat_default_as_index_map(self) -> None:
        pd = self.pd
        dtype = ArrayType(
            item_field=Field(name="item", dtype=INT64, nullable=True)
        )

        # pandas.Series.fillna(list) raises, and fillna(dict) broadcasts
        # over the index — so the helper has to route nested defaults
        # through a positional assignment instead.
        series = pd.Series([[1, 2], None, [3]], dtype="object")

        filled = dtype.fill_pandas_series_nulls(series, nullable=False)

        self.assertEqual(filled.tolist(), [[1, 2], [], [3]])

    def test_pandas_fill_struct_nulls_broadcasts_nested_default(self) -> None:
        pd = self.pd
        inner = StructType(
            fields=[Field(name="bar", dtype=INT64, nullable=False)]
        )
        outer = StructType(
            fields=[Field(name="foo", dtype=inner, nullable=False)]
        )

        series = pd.Series(
            [{"foo": {"bar": 5}}, None, {"foo": {"bar": 7}}],
            dtype="object",
        )

        filled = outer.fill_pandas_series_nulls(series, nullable=False)

        self.assertEqual(
            filled.tolist(),
            [
                {"foo": {"bar": 5}},
                {"foo": {"bar": 0}},
                {"foo": {"bar": 7}},
            ],
        )


class TestFieldDefaultPyobjCascade(ArrowTestCase):
    """``Field.default_pyobj`` is the cascade entry point used by nested dtypes."""

    def test_explicit_default_wins(self) -> None:
        f = Field(name="x", dtype=INT64, nullable=True, default=9)
        self.assertEqual(f.default_pyobj, 9)

    def test_nullable_without_explicit_default_is_none(self) -> None:
        f = Field(name="x", dtype=INT64, nullable=True)
        self.assertIsNone(f.default_pyobj)

    def test_non_nullable_falls_back_to_dtype_default(self) -> None:
        f = Field(name="x", dtype=INT64, nullable=False)
        self.assertEqual(f.default_pyobj, 0)

    def test_non_nullable_struct_field_cascades(self) -> None:
        inner = StructType(
            fields=[Field(name="bar", dtype=INT64, nullable=False)]
        )
        f = Field(name="s", dtype=inner, nullable=False)
        self.assertEqual(f.default_pyobj, {"bar": 0})
