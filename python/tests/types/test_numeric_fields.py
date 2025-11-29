import pyarrow as pa
import pytest

from yggdrasil.libs.pandaslib import pandas
from yggdrasil.libs.polarslib import polars
from yggdrasil.types.fields.scalar.decimal_field import DecimalField
from yggdrasil.types.fields.scalar.floating_field import FloatingField

requires_polars_and_pandas = pytest.mark.skipif(
    polars is None or pandas is None, reason="polars and pandas are required",
)


def test_floating_field_to_arrow_and_python():
    field = FloatingField("measure", bytesize=4, nullable=False, metadata={"source": "sensor"})

    arrow_field = field.to_arrow()
    assert arrow_field.name == "measure"
    assert arrow_field.type == pa.float32()
    assert arrow_field.metadata_bytes == {b"source": b"sensor"}
    assert arrow_field.metadata == {"source": "sensor"}

    python_field = field.to_python()
    assert python_field.type is float
    assert python_field.metadata == {"source": "sensor"}


@requires_polars_and_pandas
@pytest.mark.parametrize("bytesize,polars_type,pandas_dtype", [
    (2, polars.Float16 if polars is not None and hasattr(polars, "Float16") else None, "float16"),
    (4, polars.Float32 if polars is not None else None, "float32"),
    (8, polars.Float64 if polars is not None else None, "float64"),
])
def test_floating_field_other_backends(bytesize, polars_type, pandas_dtype):
    field = FloatingField("f", bytesize=bytesize, metadata={"v": 1})

    polars_field = field.to_polars()
    assert polars_field.type == polars_type

    pandas_field = field.to_pandas()
    assert pandas_field.type == pandas_dtype


@pytest.mark.parametrize("precision,scale", [(10, 2), (20, 4)])
def test_decimal_field_arrow_conversion(precision, scale):
    field = DecimalField("amount", precision=precision, scale=scale, metadata={"currency": "usd"})
    arrow_field = field.to_arrow()

    assert arrow_field.type == pa.decimal128(precision, scale)
    assert arrow_field.metadata_bytes == {b"currency": b"usd"}
    assert arrow_field.metadata == {"currency": "usd"}

    python_field = field.to_python()
    assert python_field.type is float
    assert python_field.metadata == {"currency": "usd"}
