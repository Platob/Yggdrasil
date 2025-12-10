import pyarrow as pa
import pytest

polars = pytest.importorskip("polars")

from yggdrasil.libs.polarslib import (
    arrow_type_to_polars_type,
    arrow_field_to_polars_field,
    polars_type_to_arrow_type,
    polars_field_to_arrow_field,
)


def test_arrow_to_polars_primitive_bool_int_string():
    assert arrow_type_to_polars_type(pa.bool_()) == polars.Boolean()

    assert arrow_type_to_polars_type(pa.int64()) == polars.Int64()
    assert arrow_type_to_polars_type(pa.uint32()) == polars.UInt32()

    # Arrow string/large_string -> Polars Utf8
    assert arrow_type_to_polars_type(pa.string()) == polars.Utf8()
    assert arrow_type_to_polars_type(pa.large_string()) == polars.Utf8()


def test_arrow_to_polars_list_and_struct():
    inner = pa.int64()
    list_type = pa.list_(inner)
    pl_list = arrow_type_to_polars_type(list_type)
    assert isinstance(pl_list, polars.List)
    assert pl_list.inner == polars.Int64()

    struct_type = pa.struct(
        [
            pa.field("a", pa.int32()),
            pa.field("b", pa.string()),
        ]
    )
    pl_struct = arrow_type_to_polars_type(struct_type)
    # Struct dtype
    assert isinstance(pl_struct, polars.Struct)
    # Fields should map correctly
    field_names = [f.name for f in pl_struct.fields]
    assert field_names == ["a", "b"]
    dtypes = [f.dtype for f in pl_struct.fields]
    assert dtypes[0] == polars.Int32()
    assert dtypes[1] == polars.Utf8()


def test_arrow_to_polars_timestamp_and_duration():
    ts = pa.timestamp("us", tz="UTC")
    pl_ts = arrow_type_to_polars_type(ts)
    assert isinstance(pl_ts, polars.Datetime)
    assert pl_ts.time_unit == "us"
    assert pl_ts.time_zone == "UTC"

    dur = pa.duration("ms")
    pl_dur = arrow_type_to_polars_type(dur)
    assert isinstance(pl_dur, polars.Duration)
    assert pl_dur.time_unit == "ms"


def test_arrow_to_polars_map_type_is_list_of_struct():
    # Arrow map(key_type, item_type)
    m = pa.map_(pa.string(), pa.int64())
    pl_dtype = arrow_type_to_polars_type(m)

    # Represented as List(Struct(key, value))
    assert isinstance(pl_dtype, polars.List)
    inner = pl_dtype.inner
    assert isinstance(inner, polars.Struct)

    field_names = [f.name for f in inner.fields]
    assert field_names == ["key", "value"]
    key_field, value_field = inner.fields
    assert key_field.dtype == polars.Utf8()
    assert value_field.dtype == polars.Int64()


def test_arrow_field_to_polars_field_and_back_roundtrip():
    arrow_field = pa.field("x", pa.int64(), nullable=False)

    pl_field = arrow_field_to_polars_field(arrow_field)
    # Depending on Polars version, this is either pl.Field or (name, dtype)
    if hasattr(polars, "Field") and isinstance(pl_field, polars.Field):
        assert pl_field.name == "x"
        assert pl_field.dtype == polars.Int64()
        pl_repr = pl_field
    else:
        name, dtype = pl_field
        assert name == "x"
        assert dtype == polars.Int64()
        pl_repr = (name, dtype)

    arrow_field2 = polars_field_to_arrow_field(pl_repr)
    assert isinstance(arrow_field2, pa.Field)
    assert arrow_field2.name == "x"
    # We don't assert nullable equality strictly, since implementation
    # defaults to nullable=True
    assert arrow_field2.type == pa.int64()


def test_polars_to_arrow_primitive_roundtrip():
    for pl_type, arrow_type in [
        (polars.Boolean, pa.bool_()),
        (polars.Int64, pa.int64()),
        (polars.UInt32, pa.uint32()),
        (polars.Float64, pa.float64()),
        (polars.Utf8, pa.large_string()),
        (polars.Date, pa.date32()),
    ]:
        # class
        assert polars_type_to_arrow_type(pl_type) == arrow_type
        # instance
        assert polars_type_to_arrow_type(pl_type()) == arrow_type


def test_polars_to_arrow_list_and_struct():
    # List<Int64>
    pl_list = polars.List(polars.Int64)
    arrow_list = polars_type_to_arrow_type(pl_list)
    assert pa.types.is_list(arrow_list)
    assert arrow_list.value_type == pa.int64()

    # Struct with fields
    Field = getattr(polars, "Field", None)
    if Field is not None:
        pl_struct = polars.Struct(
            [Field("a", polars.Int32), Field("b", polars.Utf8)]
        )
    else:
        pl_struct = polars.Struct({"a": polars.Int32, "b": polars.Utf8})

    arrow_struct = polars_type_to_arrow_type(pl_struct)
    assert pa.types.is_struct(arrow_struct)
    assert [f.name for f in arrow_struct] == ["a", "b"]
    assert arrow_struct.field("a").type == pa.int32()
    assert arrow_struct.field("b").type == pa.large_string()


def test_polars_to_arrow_datetime_and_duration():
    pl_dt = polars.Datetime(time_unit="ms", time_zone="UTC")
    arrow_dt = polars_type_to_arrow_type(pl_dt)
    assert pa.types.is_timestamp(arrow_dt)
    assert arrow_dt.unit == "ms"
    assert arrow_dt.tz == "UTC"

    pl_dur = polars.Duration(time_unit="us")
    arrow_dur = polars_type_to_arrow_type(pl_dur)
    assert pa.types.is_duration(arrow_dur)
    assert arrow_dur.unit == "us"


def test_arrow_to_polars_unsupported_raises():
    dec = pa.decimal128(10, 2)
    with pytest.raises(TypeError):
        arrow_type_to_polars_type(dec)


def test_polars_to_arrow_unsupported_raises():
    class FakeDtype:
        pass

    with pytest.raises(TypeError):
        polars_type_to_arrow_type(FakeDtype())
