"""DataType engine-type conversion caches and singleton behavior.

Covers:
  * :data:`_FROM_ARROW_TYPE_CACHE` — populated on first call, same instance
    returned on repeat, keyed by ``pa.DataType`` equality not identity.
  * ``DataType.__new__`` — per-class singleton on default-arg construction.
  * :func:`_cached_engine_method` decorator — ``to_arrow`` / ``to_polars``
    results cached per-instance via a private attribute.
  * :func:`_arrow_types_compatible` — exact match, identity short-circuit,
    and the view/flat layout bypass (``string_view`` ↔ ``string``, etc.).
  * ``_FROM_POLARS_TYPE_CACHE`` — same caching contract as the Arrow cache
    (gated by the ``polars`` optional dependency).
"""
from __future__ import annotations

import pyarrow as pa
import pytest


# ---------------------------------------------------------------------------
# Arrow type cache
# ---------------------------------------------------------------------------


class TestFromArrowTypeCache:

    def test_cache_populated_on_first_call(self) -> None:
        from yggdrasil.data.types.base import DataType, _FROM_ARROW_TYPE_CACHE

        arrow_type = pa.int64()
        _FROM_ARROW_TYPE_CACHE.pop(arrow_type, None)

        result = DataType.from_arrow_type(arrow_type)
        assert arrow_type in _FROM_ARROW_TYPE_CACHE
        assert _FROM_ARROW_TYPE_CACHE[arrow_type] is result

    def test_same_instance_returned_on_repeat(self) -> None:
        from yggdrasil.data.types.base import DataType

        arrow_type = pa.float64()
        r1 = DataType.from_arrow_type(arrow_type)
        r2 = DataType.from_arrow_type(arrow_type)
        assert r1 is r2

    def test_equal_arrow_types_share_cache_entry(self) -> None:
        """Two separately constructed pa.DataType objects that compare equal
        should collapse to the same cache entry."""
        from yggdrasil.data.types.base import DataType

        t1 = pa.timestamp("us", tz="UTC")
        t2 = pa.timestamp("us", tz="UTC")
        assert t1 == t2

        r1 = DataType.from_arrow_type(t1)
        r2 = DataType.from_arrow_type(t2)
        assert r1 is r2

    def test_distinct_arrow_types_yield_distinct_results(self) -> None:
        from yggdrasil.data.types.base import DataType

        r_i32 = DataType.from_arrow_type(pa.int32())
        r_i64 = DataType.from_arrow_type(pa.int64())
        assert type(r_i32) is not type(r_i64) or r_i32 != r_i64

    def test_unsupported_arrow_type_raises(self) -> None:
        from yggdrasil.data.types.base import DataType

        with pytest.raises(TypeError, match="Unsupported Arrow"):
            DataType.from_arrow_type(pa.month_day_nano_interval())

    def test_subclass_rooted_call_bypasses_cache(self) -> None:
        """``IntegerType.from_arrow_type`` uses its own subclass walk, not
        the module-level cache, so a direct subclass call returning a
        different instance than the base-class cache is fine."""
        from yggdrasil.data.types.base import DataType
        from yggdrasil.data.types.primitive import IntegerType

        arrow_type = pa.int32()
        base_result = DataType.from_arrow_type(arrow_type)
        sub_result = IntegerType.from_arrow_type(arrow_type)
        assert type(base_result) is type(sub_result)


# ---------------------------------------------------------------------------
# DataType singleton via __new__
# ---------------------------------------------------------------------------


class TestDataTypeSingleton:

    def test_default_construction_returns_same_instance(self) -> None:
        from yggdrasil.data.types.primitive import StringType

        a = StringType()
        b = StringType()
        assert a is b

    def test_non_default_construction_may_differ(self) -> None:
        from yggdrasil.data.types.primitive import IntegerType

        a = IntegerType(byte_size=4, signed=True)
        b = IntegerType(byte_size=8, signed=True)
        assert a != b

    def test_instance_classmethod_returns_singleton(self) -> None:
        from yggdrasil.data.types.primitive import BooleanType

        a = BooleanType.instance()
        b = BooleanType.instance()
        assert a is b

    def test_singleton_is_same_as_default_construction(self) -> None:
        from yggdrasil.data.types.primitive import StringType

        assert StringType.instance() is StringType()


# ---------------------------------------------------------------------------
# _cached_engine_method — to_arrow / to_polars caching
# ---------------------------------------------------------------------------


class TestCachedEngineMethod:

    def test_to_arrow_result_is_reused(self) -> None:
        from yggdrasil.data.types.primitive import StringType

        dt = StringType()
        r1 = dt.to_arrow()
        r2 = dt.to_arrow()
        assert r1 is r2

    def test_to_arrow_returns_correct_type(self) -> None:
        from yggdrasil.data.types.primitive import IntegerType

        dt = IntegerType(byte_size=4, signed=True)
        assert dt.to_arrow() == pa.int32()

    def test_timestamp_to_arrow_cached(self) -> None:
        from yggdrasil.data.types.primitive import TimestampType

        dt = TimestampType(unit="us", tz="UTC")
        r1 = dt.to_arrow()
        r2 = dt.to_arrow()
        assert r1 is r2
        assert r1 == pa.timestamp("us", tz="UTC")

    def test_different_instances_have_independent_caches(self) -> None:
        from yggdrasil.data.types.primitive import TimestampType

        dt_utc = TimestampType(unit="us", tz="UTC")
        dt_naive = TimestampType(unit="us", tz=None)
        assert dt_utc.to_arrow() != dt_naive.to_arrow()


# ---------------------------------------------------------------------------
# _arrow_types_compatible — bypass equality
# ---------------------------------------------------------------------------


class TestArrowTypesCompatible:

    def test_identity_shortcircuit(self) -> None:
        from yggdrasil.data.types.base import _arrow_types_compatible

        t = pa.int64()
        assert _arrow_types_compatible(t, t) is True

    def test_equal_types_compatible(self) -> None:
        from yggdrasil.data.types.base import _arrow_types_compatible

        assert _arrow_types_compatible(pa.string(), pa.string()) is True
        assert _arrow_types_compatible(pa.timestamp("us", tz="UTC"), pa.timestamp("us", tz="UTC")) is True

    def test_unequal_types_not_compatible(self) -> None:
        from yggdrasil.data.types.base import _arrow_types_compatible

        assert _arrow_types_compatible(pa.int32(), pa.int64()) is False
        assert _arrow_types_compatible(pa.string(), pa.binary()) is False

    def test_string_view_and_string_are_compatible(self) -> None:
        from yggdrasil.data.types.base import _arrow_types_compatible

        if not hasattr(pa, "string_view"):
            pytest.skip("pyarrow version does not have string_view")
        sv = pa.string_view()
        assert _arrow_types_compatible(sv, pa.string()) is True
        assert _arrow_types_compatible(pa.string(), sv) is True

    def test_binary_view_and_binary_are_compatible(self) -> None:
        from yggdrasil.data.types.base import _arrow_types_compatible

        if not hasattr(pa, "binary_view"):
            pytest.skip("pyarrow version does not have binary_view")
        bv = pa.binary_view()
        assert _arrow_types_compatible(bv, pa.binary()) is True
        assert _arrow_types_compatible(pa.binary(), bv) is True

    def test_list_view_and_list_compatible_same_element(self) -> None:
        from yggdrasil.data.types.base import _arrow_types_compatible

        if not hasattr(pa, "list_view"):
            pytest.skip("pyarrow version does not have list_view")
        lv = pa.list_view(pa.int32())
        ls = pa.list_(pa.int32())
        assert _arrow_types_compatible(lv, ls) is True
        assert _arrow_types_compatible(ls, lv) is True

    def test_list_view_and_list_incompatible_different_element(self) -> None:
        from yggdrasil.data.types.base import _arrow_types_compatible

        if not hasattr(pa, "list_view"):
            pytest.skip("pyarrow version does not have list_view")
        lv_i32 = pa.list_view(pa.int32())
        ls_i64 = pa.list_(pa.int64())
        assert _arrow_types_compatible(lv_i32, ls_i64) is False


# ---------------------------------------------------------------------------
# _FROM_POLARS_TYPE_CACHE (optional — skipped without polars)
# ---------------------------------------------------------------------------


class TestFromPolarsTypeCache:

    def test_cache_populated_and_reused(self) -> None:
        try:
            import polars as pl
        except ImportError:
            pytest.skip("polars not installed")

        from yggdrasil.data.types.base import DataType, _FROM_POLARS_TYPE_CACHE

        dtype = pl.Int64()
        _FROM_POLARS_TYPE_CACHE.pop(dtype, None)

        r1 = DataType.from_polars_type(dtype)
        assert dtype in _FROM_POLARS_TYPE_CACHE
        r2 = DataType.from_polars_type(dtype)
        assert r1 is r2

    def test_distinct_polars_types_yield_distinct_ygg_types(self) -> None:
        try:
            import polars as pl
        except ImportError:
            pytest.skip("polars not installed")

        from yggdrasil.data.types.base import DataType

        r_i32 = DataType.from_polars_type(pl.Int32())
        r_i64 = DataType.from_polars_type(pl.Int64())
        assert r_i32 != r_i64
