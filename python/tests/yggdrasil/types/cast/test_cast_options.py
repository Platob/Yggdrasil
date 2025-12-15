# tests/test_cast_options.py
#
# Real-function tests (no mocks) for:
#   yggdrasil.types.cast.cast_options.CastOptions
#
# Run:
#   pytest -q

from __future__ import annotations

import pytest
import pyarrow as pa

from yggdrasil.types.cast.cast_options import CastOptions, DEFAULT_INSTANCE


def test_default_instance_shape():
    assert isinstance(DEFAULT_INSTANCE, CastOptions)
    # sanity: defaults
    assert DEFAULT_INSTANCE.safe is False
    assert DEFAULT_INSTANCE.add_missing_columns is True
    assert DEFAULT_INSTANCE.strict_match_names is False
    assert DEFAULT_INSTANCE.allow_add_columns is False


def test_safe_init_sets_source_and_target_fields_from_arrow_objects():
    src = pa.field("s", pa.int32(), nullable=True)
    tgt = pa.field("t", pa.int64(), nullable=False)

    opt = CastOptions.safe_init(source_field=src, target_field=tgt, safe=True, eager=True)

    assert opt.safe is True
    assert opt.eager is True
    assert opt.source_field == src
    assert opt.target_field == tgt


def test_check_arg_accepts_field_schema_datatype_and_none():
    # None -> should still return a CastOptions
    opt0 = CastOptions.check_arg(None)
    assert isinstance(opt0, CastOptions)

    # DataType -> should become a Field (name possibly empty depending on convert())
    opt1 = CastOptions.check_arg(pa.int64())
    assert isinstance(opt1, CastOptions)
    assert isinstance(opt1.target_field, pa.Field)
    assert opt1.target_field.type == pa.int64()

    # Field -> stays Field
    f = pa.field("x", pa.string(), nullable=True)
    opt2 = CastOptions.check_arg(f)
    assert opt2.target_field == f

    # Schema -> should convert to a Field (contract of your registry.convert)
    sch = pa.schema([pa.field("a", pa.int32()), pa.field("b", pa.int64())])
    opt3 = CastOptions.check_arg(sch)
    assert isinstance(opt3.target_field, pa.Field)


def test_check_arg_applies_source_and_target_overrides():
    src = pa.field("src", pa.int32(), nullable=True)
    tgt = pa.field("tgt", pa.int32(), nullable=False)

    opt = CastOptions.check_arg(None, source_field=src, target_field=tgt)
    assert opt.source_field == src
    assert opt.target_field == tgt


def test_check_arg_kwargs_trigger_copy_not_in_place():
    base = CastOptions.safe_init(safe=False, target_field=pa.field("x", pa.int32()))
    res = CastOptions.check_arg(base, safe=True)

    # kwargs path does result.copy(**kwargs) -> new instance
    assert res is not base
    assert res.safe is True
    assert res.target_field == base.target_field


def test_copy_boolean_merge_semantics_and_add_missing_columns_override():
    base = CastOptions.safe_init(
        safe=False,
        add_missing_columns=True,
        strict_match_names=False,
        allow_add_columns=False,
        eager=False,
        datetime_patterns=None,
        target_field=pa.field("x", pa.int32()),
    )

    # OR-merge semantics for flags (safe/strict/allow_add/eager)
    c1 = base.copy(safe=True, strict_match_names=True, allow_add_columns=True, eager=True)
    assert c1.safe is True
    assert c1.strict_match_names is True
    assert c1.allow_add_columns is True
    assert c1.eager is True
    # special handling: add_missing_columns defaults to "inherit"
    assert c1.add_missing_columns is True

    # explicit override should work
    c2 = base.copy(add_missing_columns=False)
    assert c2.add_missing_columns is False


def test_copy_datetime_patterns_edge_cases():
    # NOTE: implementation uses `self.datetime_patterns or datetime_patterns`
    # so empty list behaves like "unset" and should allow override.
    base = CastOptions.safe_init(datetime_patterns=[])
    c = base.copy(datetime_patterns=["%Y-%m-%d"])
    assert c.datetime_patterns == ["%Y-%m-%d"]

    base2 = CastOptions.safe_init(datetime_patterns=["%Y"])
    c2 = base2.copy(datetime_patterns=["%Y-%m-%d"])
    # because self.datetime_patterns is truthy, it should win
    assert c2.datetime_patterns == ["%Y"]


def test_check_source_only_sets_when_unset_and_obj_not_none():
    opt = CastOptions.safe_init(target_field=pa.field("x", pa.int32()))
    assert opt.source_field is None

    opt.check_source(pa.field("s", pa.int32()))
    assert opt.source_field.name == "s"

    # should not overwrite
    opt.check_source(pa.field("s2", pa.int64()))
    assert opt.source_field.name == "s"


def test_need_arrow_type_cast_and_nullability_check():
    src = pa.field("a", pa.int32(), nullable=True)

    opt_same = CastOptions.safe_init(source_field=src, target_field=pa.field("a", pa.int32(), nullable=True))
    assert opt_same.need_arrow_type_cast(source_obj=None) is False
    assert opt_same.need_nullability_check(source_obj=None) is False

    opt_diff = CastOptions.safe_init(source_field=src, target_field=pa.field("a", pa.int64(), nullable=True))
    assert opt_diff.need_arrow_type_cast(source_obj=None) is True

    opt_null_tighten = CastOptions.safe_init(source_field=src, target_field=pa.field("a", pa.int32(), nullable=False))
    assert opt_null_tighten.need_nullability_check(source_obj=None) is True

    opt_null_ok = CastOptions.safe_init(
        source_field=pa.field("a", pa.int32(), nullable=False),
        target_field=pa.field("a", pa.int32(), nullable=False),
    )
    assert opt_null_ok.need_nullability_check(source_obj=None) is False


def test_target_field_name_fallbacks():
    opt0 = CastOptions.safe_init(target_field=None)
    assert opt0.target_field_name == ""

    opt1 = CastOptions.safe_init(
        source_field=pa.field("src", pa.int32()),
        target_field=pa.field("", pa.int32()),
    )
    assert opt1.target_field_name == "src"

    opt2 = CastOptions.safe_init(
        source_field=pa.field("src", pa.int32()),
        target_field=pa.field("tgt", pa.int32()),
    )
    assert opt2.target_field_name == "tgt"


def test_child_arrow_field_struct_list_map_and_union_notimplemented():
    # struct: returns indexed child field
    f_struct = pa.field(
        "s",
        pa.struct([pa.field("x", pa.int32()), pa.field("y", pa.string())]),
        nullable=True,
    )
    c0 = CastOptions._child_arrow_field(f_struct, index=0)
    assert c0.name == "x"
    assert c0.type == pa.int32()

    # list-like: returns value_field
    f_list = pa.field("l", pa.list_(pa.field("item", pa.int32())), nullable=True)
    cl = CastOptions._child_arrow_field(f_list, index=0)
    assert isinstance(cl, pa.Field)
    assert cl.name == "item"
    assert cl.type == pa.int32()

    # map: returns synthetic "entries" struct(key, value), nullable=False
    f_map = pa.field("m", pa.map_(pa.string(), pa.int32()), nullable=True)
    cm = CastOptions._child_arrow_field(f_map, index=0)
    assert cm.name == "entries"
    assert cm.nullable is False
    assert pa.types.is_struct(cm.type)
    assert cm.type[0].name == "key"
    assert cm.type[1].name == "value"

    # union: nested-ish but not supported in implementation -> NotImplementedError
    u = pa.union([pa.field("a", pa.int32()), pa.field("b", pa.string())], mode="sparse")
    f_union = pa.field("u", u)
    with pytest.raises(NotImplementedError):
        CastOptions._child_arrow_field(f_union, index=0)


def test_target_arrow_schema_struct_unwrap_and_scalar_wrap():
    # struct target -> schema unwrap
    tf = pa.field("root", pa.struct([pa.field("x", pa.int32()), pa.field("y", pa.string())]))
    opt = CastOptions.safe_init(target_field=tf)
    sch = opt.target_arrow_schema
    assert isinstance(sch, pa.Schema)
    assert [f.name for f in sch] == ["x", "y"]

    # scalar target -> single-field schema
    opt2 = CastOptions.safe_init(target_field=pa.field("z", pa.int64()))
    sch2 = opt2.target_arrow_schema
    assert isinstance(sch2, pa.Schema)
    assert [f.name for f in sch2] == ["z"]


@pytest.mark.skipif(
    pytest.importorskip("polars", reason="polars not installed") is None,  # pragma: no cover
    reason="polars not installed",
)
def test_polars_field_lazy_and_cached_identity():
    # This exercises the real arrow_field_to_polars_field conversion in your codebase.
    opt = CastOptions.safe_init(
        source_field=pa.field("a", pa.int32()),
        target_field=pa.field("b", pa.int64()),
    )

    sp1 = opt.source_polars_field
    tp1 = opt.target_polars_field
    assert sp1 is not None
    assert tp1 is not None

    # Cached: subsequent property access should return same object identity
    assert opt.source_polars_field is sp1
    assert opt.target_polars_field is tp1


@pytest.mark.skipif(
    pytest.importorskip("polars", reason="polars not installed") is None,  # pragma: no cover
    reason="polars not installed",
)
def test_need_polars_type_cast_true_and_false():
    # equal types
    opt_same = CastOptions.safe_init(
        source_field=pa.field("a", pa.int32()),
        target_field=pa.field("a", pa.int32()),
    )
    assert opt_same.need_polars_type_cast(source_obj=None) is False

    # different types
    opt_diff = CastOptions.safe_init(
        source_field=pa.field("a", pa.int32()),
        target_field=pa.field("a", pa.int64()),
    )
    assert opt_diff.need_polars_type_cast(source_obj=None) is True


@pytest.mark.skipif(
    pytest.importorskip("pyspark", reason="pyspark not installed") is None,  # pragma: no cover
    reason="pyspark not installed",
)
def test_spark_field_lazy_cached_and_need_spark_type_cast():
    # This one is intentionally spicy: it will catch two common regressions:
    #  1) caching field written to the wrong attribute
    #  2) need_spark_type_cast comparing the same thing to itself
    opt = CastOptions.safe_init(
        source_field=pa.field("a", pa.int32()),
        target_field=pa.field("a", pa.int64()),
    )

    ss1 = opt.source_spark_field
    ts1 = opt.target_spark_field
    assert ss1 is not None
    assert ts1 is not None

    # Cached identity
    assert opt.source_spark_field is ss1
    assert opt.target_spark_field is ts1

    # Different source/target types should require cast
    assert opt.need_spark_type_cast(source_obj=None) is True


@pytest.mark.skipif(
    pytest.importorskip("pyspark", reason="pyspark not installed") is None,  # pragma: no cover
    reason="pyspark not installed",
)
def test_target_spark_schema_roundtrip_presence():
    # target_spark_schema should be non-None when target_arrow_schema is non-None
    opt = CastOptions.safe_init(target_field=pa.field("z", pa.int64()))
    assert opt.target_arrow_schema is not None
    assert opt.target_spark_schema is not None

    # if no target_field => None
    opt2 = CastOptions.safe_init(target_field=None)
    assert opt2.target_spark_schema is None
