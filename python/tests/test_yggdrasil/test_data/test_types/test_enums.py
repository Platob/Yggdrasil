"""Tests for :class:`DictionaryType`, :class:`EnumType`, and the typed
:class:`StrEnumType` / :class:`IntEnumType` specializations.

The cases line up the cross-engine contract:

* construction — coercion / de-dup of categories, default value
  types, rejection of nested dictionaries and ``None`` categories;
* parser / dispatch — ``DataType.from_str`` / ``from_dict`` /
  ``from_pytype`` land on the right class and round-trip through
  the dict exporter;
* arrow — encode (raw values → dict-encoded) and identity / re-
  encode of an existing dictionary array via ``cast_arrow_array``;
* polars — same paths through ``cast_polars_series`` so the
  ``pl.Enum`` cast (``strict=False`` lenient semantics) is wired
  up; non-string value types degrade to the value dtype;
* merge — union of categories on widen, intersection on narrow,
  preservation of ``ordered`` only when category tuples agree;
* scalar / pyobj — ``_convert_pyobj`` accepts both the raw value
  and the encoded form, and emits ``None`` for unknowns (lenient)
  / raises (``safe=True``).
"""
from __future__ import annotations

import enum

import pyarrow as pa
import pytest

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.types import (
    DataType,
    DataTypeId,
    DictionaryType,
    EnumType,
    Int32Type,
    IntEnumType,
    IntegerType,
    StrEnumType,
    StringType,
)


# ---------------------------------------------------------------------------
# Construction / __post_init__
# ---------------------------------------------------------------------------


class TestConstruction:

    def test_default_value_type_is_string(self) -> None:
        d = DictionaryType(categories=("a", "b"))
        assert isinstance(d.value_type, StringType)
        assert d.byte_size == 4  # int32 indices

    def test_categories_are_coerced_through_value_type(self) -> None:
        d = DictionaryType(value_type=IntegerType(), categories=("1", 2, 3.0))
        assert d.categories == (1, 2, 3)

    def test_categories_dedup_in_first_seen_order(self) -> None:
        d = DictionaryType(categories=["a", "b", "a", "c", "b"])
        assert d.categories == ("a", "b", "c")

    def test_categories_none_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot contain None"):
            DictionaryType(categories=("a", None, "b"))

    def test_nested_dictionary_rejected(self) -> None:
        inner = DictionaryType(categories=("x",))
        with pytest.raises(TypeError, match="cannot be nested"):
            DictionaryType(value_type=inner, categories=("x",))

    def test_open_dictionary_has_empty_categories(self) -> None:
        d = DictionaryType(value_type=StringType(), categories=())
        assert d.categories == ()
        assert d.cardinality == 0

    def test_strenum_pins_value_type(self) -> None:
        # Even when the caller hands in an IntegerType the class
        # forces a StringType — the contract is "string-valued enum",
        # silently switching backends would lie.
        e = StrEnumType(value_type=IntegerType(), categories=())
        assert isinstance(e.value_type, StringType)

    def test_intenum_pins_value_type(self) -> None:
        e = IntEnumType(value_type=StringType(), categories=())
        assert isinstance(e.value_type, IntegerType)
        assert e.value_type.byte_size == 8

    def test_intenum_accepts_integer_subclass_width(self) -> None:
        e = IntEnumType(value_type=Int32Type(), categories=(1, 2))
        assert isinstance(e.value_type, Int32Type)
        assert e.to_arrow() == pa.dictionary(pa.int32(), pa.int32())


# ---------------------------------------------------------------------------
# Type ids / dispatch / round-trip
# ---------------------------------------------------------------------------


class TestDispatch:

    def test_type_ids(self) -> None:
        assert DictionaryType.class_type_id() is DataTypeId.DICTIONARY
        assert EnumType.class_type_id() is DataTypeId.ENUM
        assert StrEnumType.class_type_id() is DataTypeId.STR_ENUM
        assert IntEnumType.class_type_id() is DataTypeId.INT_ENUM
        # All four sit in the extension band.
        for tid in (
            DataTypeId.DICTIONARY,
            DataTypeId.ENUM,
            DataTypeId.STR_ENUM,
            DataTypeId.INT_ENUM,
        ):
            assert tid.is_extension
            assert tid.is_dictionary_like

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("dictionary", DictionaryType),
            ("categorical", DictionaryType),
            ("enum", EnumType),
        ],
    )
    def test_from_str_bare(self, alias: str, expected: type) -> None:
        dtype = DataType.from_str(alias)
        assert isinstance(dtype, expected)

    def test_from_str_literal_strings_resolve_to_strenum(self) -> None:
        dtype = DataType.from_str("literal('a','b','c')")
        assert isinstance(dtype, StrEnumType)
        assert dtype.categories == ("a", "b", "c")

    def test_from_str_literal_ints_resolve_to_intenum(self) -> None:
        dtype = DataType.from_str("literal(1, 2, 3)")
        assert isinstance(dtype, IntEnumType)
        assert dtype.categories == (1, 2, 3)

    def test_from_str_enum_with_string_args(self) -> None:
        dtype = DataType.from_str("enum('x', 'y')")
        assert isinstance(dtype, StrEnumType)
        assert dtype.categories == ("x", "y")

    @pytest.mark.parametrize(
        "factory",
        [
            # Int64Type used here (rather than IntegerType()) because
            # IntegerType()'s ``__new__`` promotes to a sized subclass
            # only when ``byte_size`` is set; the serialization
            # default is 8, so a None-width input wouldn't round-trip
            # cleanly — that's a pre-existing IntegerType quirk, not
            # an enum/dict one.
            lambda: DictionaryType(categories=("a", "b", "c")),
            lambda: DictionaryType(
                value_type=Int32Type(), categories=(1, 2, 3), ordered=True
            ),
            lambda: EnumType(
                value_type=StringType(),
                categories=("x", "y"),
                name="Foo",
                members={"X": "x", "Y": "y"},
            ),
            lambda: StrEnumType(categories=("red", "blue"), name="Color"),
            lambda: IntEnumType(categories=(1, 2, 3), name="Status"),
        ],
    )
    def test_to_dict_roundtrip(self, factory) -> None:
        dtype = factory()
        out = DataType.from_dict(dtype.to_dict())
        assert out == dtype
        assert type(out) is type(dtype)


# ---------------------------------------------------------------------------
# Pyobj coercion / defaults
# ---------------------------------------------------------------------------


class TestPyobj:

    def test_known_value_passes_through(self) -> None:
        d = DictionaryType(categories=("a", "b", "c"))
        assert d._convert_pyobj("a") == "a"

    def test_unknown_value_lenient_returns_none(self) -> None:
        d = DictionaryType(categories=("a", "b"))
        assert d._convert_pyobj("z") is None

    def test_unknown_value_safe_raises(self) -> None:
        d = DictionaryType(categories=("a", "b"))
        with pytest.raises(ValueError, match="not a member"):
            d._convert_pyobj("z", safe=True)

    def test_open_dictionary_accepts_any_value(self) -> None:
        d = DictionaryType(value_type=IntegerType(), categories=())
        assert d._convert_pyobj("42") == 42

    def test_default_pyobj_nullable(self) -> None:
        assert DictionaryType(categories=("a",)).default_pyobj(nullable=True) is None
        assert IntEnumType(categories=(1, 2)).default_pyobj(nullable=True) is None

    def test_default_pyobj_non_nullable_uses_first_category(self) -> None:
        assert (
            DictionaryType(categories=("a", "b")).default_pyobj(nullable=False)
            == "a"
        )
        assert (
            IntEnumType(categories=(7, 8)).default_pyobj(nullable=False) == 7
        )

    def test_intenum_coerces_string_input(self) -> None:
        e = IntEnumType(categories=(1, 2, 3))
        assert e._convert_pyobj("2") == 2


# ---------------------------------------------------------------------------
# Merge — union on widen, intersection on narrow
# ---------------------------------------------------------------------------


class TestMerge:

    def test_widening_unions_categories(self) -> None:
        a = DictionaryType(categories=("a", "b"))
        b = DictionaryType(categories=("b", "c"))
        merged = a.merge_with(b, upcast=True)
        assert merged.categories == ("a", "b", "c")

    def test_narrowing_intersects_categories(self) -> None:
        a = DictionaryType(categories=("a", "b", "c"))
        b = DictionaryType(categories=("b", "c", "d"))
        merged = a.merge_with(b, downcast=True)
        assert merged.categories == ("b", "c")

    def test_ordered_only_preserved_when_categories_match(self) -> None:
        a = DictionaryType(categories=("a", "b"), ordered=True)
        b = DictionaryType(categories=("a", "b"), ordered=True)
        merged = a.merge_with(b, upcast=True)
        assert merged.ordered is True

        b2 = DictionaryType(categories=("a", "c"), ordered=True)
        merged2 = a.merge_with(b2, upcast=True)
        assert merged2.ordered is False

    def test_enum_merge_preserves_name_when_matching(self) -> None:
        a = EnumType(categories=("a", "b"), name="Foo")
        b = EnumType(categories=("a", "b"), name="Foo")
        merged = a.merge_with(b, upcast=True)
        assert isinstance(merged, EnumType)
        assert merged.name == "Foo"

    def test_enum_merge_drops_name_when_diverging(self) -> None:
        a = EnumType(categories=("a", "b"), name="Foo")
        b = EnumType(categories=("a", "b"), name="Bar")
        merged = a.merge_with(b, upcast=True)
        assert merged.name is None


# ---------------------------------------------------------------------------
# Arrow casts
# ---------------------------------------------------------------------------


class TestArrowCasts(ArrowTestCase):

    def _cast(self, dtype, src, source_dtype):
        return dtype.cast_arrow_array(
            src,
            source=Field("v", source_dtype),
            target=Field("v", dtype),
        )

    def test_string_to_dictionary_encodes_and_nulls_unknowns(self) -> None:
        target = DictionaryType(categories=("a", "b", "c"))
        src = self.pa.array(
            ["a", "b", "c", "a", "unknown", None], type=self.pa.string()
        )
        out = self._cast(target, src, StringType())

        self.assertTrue(self.pa.types.is_dictionary(out.type))
        self.assertEqual(
            out.to_pylist(),
            ["a", "b", "c", "a", None, None],
        )

    def test_dictionary_with_different_order_re_encodes(self) -> None:
        # Source has categories in a different order — output indices
        # must line up against the target's category ordering.
        src = self.pa.DictionaryArray.from_arrays(
            indices=self.pa.array([0, 1, 0], type=self.pa.int32()),
            dictionary=self.pa.array(["b", "a"], type=self.pa.string()),
        )
        target = DictionaryType(categories=("a", "b"))
        out = self._cast(target, src, DictionaryType(categories=("b", "a")))

        # Dictionary is rebuilt against the target's category order.
        self.assertEqual(out.dictionary.to_pylist(), ["a", "b"])
        self.assertEqual(out.to_pylist(), ["b", "a", "b"])

    def test_int_enum_encode(self) -> None:
        target = IntEnumType(categories=(1, 2, 3))
        src = self.pa.array([1, 2, 3, 99, None], type=self.pa.int64())
        out = self._cast(target, src, IntegerType())

        self.assertEqual(out.to_pylist(), [1, 2, 3, None, None])
        self.assertEqual(out.type.value_type, self.pa.int64())

    def test_open_dictionary_round_trips_values(self) -> None:
        target = DictionaryType(value_type=StringType(), categories=())
        src = self.pa.array(["a", "b", "c", None], type=self.pa.string())
        out = self._cast(target, src, StringType())

        # No category gating — all values survive (including ``None``).
        self.assertTrue(self.pa.types.is_dictionary(out.type))
        self.assertEqual(out.to_pylist(), ["a", "b", "c", None])

    def test_handles_arrow_dispatch(self) -> None:
        # ``DictionaryType.from_arrow_type`` returns an open dict
        # (``categories=()``) since a bare type doesn't carry the
        # value set.
        dtype = DataType.from_arrow_type(
            self.pa.dictionary(self.pa.int32(), self.pa.string())
        )
        self.assertIsInstance(dtype, DictionaryType)
        self.assertEqual(dtype.categories, ())

    def test_from_arrow_array_recovers_categories(self) -> None:
        arr = self.pa.DictionaryArray.from_arrays(
            indices=self.pa.array([0, 1, 0, 2], type=self.pa.int32()),
            dictionary=self.pa.array(["a", "b", "c"], type=self.pa.string()),
        )
        dtype = DictionaryType.from_arrow_array(arr)
        self.assertEqual(dtype.categories, ("a", "b", "c"))


# ---------------------------------------------------------------------------
# Python enum.Enum integration
# ---------------------------------------------------------------------------


class _Color(str, enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class _Status(enum.IntEnum):
    OK = 0
    FAIL = 1


class TestFromPyEnum:

    def test_str_enum_promotes_to_strenumtype(self) -> None:
        dtype = DataType.from_pytype(_Color)
        assert isinstance(dtype, StrEnumType)
        assert dtype.name == "_Color"
        assert dtype.categories == ("red", "green", "blue")
        assert dtype.members == {"RED": "red", "GREEN": "green", "BLUE": "blue"}

    def test_int_enum_promotes_to_intenumtype(self) -> None:
        dtype = DataType.from_pytype(_Status)
        assert isinstance(dtype, IntEnumType)
        assert dtype.name == "_Status"
        assert dtype.categories == (0, 1)
        assert dtype.members == {"OK": 0, "FAIL": 1}

    def test_from_pyenum_classmethod(self) -> None:
        # Direct entry point — same shape as the from_pytype path.
        dtype = EnumType.from_pyenum(_Color)
        assert isinstance(dtype, StrEnumType)
        assert dtype.categories == ("red", "green", "blue")


# ---------------------------------------------------------------------------
# Polars casts — only loaded when polars is installed
# ---------------------------------------------------------------------------


pl = pytest.importorskip("polars", reason="polars not installed")


class TestPolarsCasts:

    def test_string_series_to_strenum_uses_pl_enum(self) -> None:
        target = StrEnumType(categories=("a", "b", "c"))
        s = pl.Series("v", ["a", "b", "unknown", None], dtype=pl.String)
        out = target.cast_polars_series(
            s,
            source=Field("v", StringType()),
            target=Field("v", target),
        )
        assert out.to_list() == ["a", "b", None, None]
        assert isinstance(out.dtype, pl.Enum)
        assert out.dtype.categories.to_list() == ["a", "b", "c"]

    def test_intenum_to_polars_degrades_to_int(self) -> None:
        # Polars has no integer-valued Enum type — the dtype falls
        # back to the underlying integer width without category
        # gating (gating happens on the encode path through arrow).
        assert IntEnumType(categories=(1, 2, 3)).to_polars() == pl.Int64
        assert (
            IntEnumType(value_type=Int32Type(), categories=(1, 2)).to_polars()
            == pl.Int32
        )
