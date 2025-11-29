import pyarrow as pa
import pytest

from yggdrasil.libs.pandaslib import pandas
from yggdrasil.libs.polarslib import polars
from yggdrasil.libs.sparklib import pyspark
from yggdrasil.types.fields.scalar.binary_field import BinaryField
from yggdrasil.types.fields.scalar.integer_field import IntegerField
from yggdrasil.types.fields.scalar.string_field import StringField

requires_polars_and_pandas = pytest.mark.skipif(
    polars is None or pandas is None, reason="polars and pandas are required",
)


@requires_polars_and_pandas
@pytest.mark.parametrize("large", [False, True])
def test_string_field_conversions(large):
    field = StringField("text", large=large, nullable=False, metadata={"k": "v"})

    python_field = field.to_python()
    assert python_field.name == "text"
    assert python_field.type is str
    assert python_field.nullable is False
    assert python_field.metadata == {"k": "v"}

    arrow_field = field.to_arrow()
    expected_dtype = pa.large_string() if large else pa.string()
    assert arrow_field.type == expected_dtype
    assert arrow_field.nullable is False
    assert arrow_field.metadata_bytes == {b"k": b"v"}

    polars_field = field.to_polars()
    assert polars_field.name == "text"
    assert polars_field.type == polars.Utf8
    assert polars_field.nullable is False
    assert polars_field.metadata == {"k": "v"}

    pandas_field = field.to_pandas()
    assert pandas_field.name == "text"
    assert isinstance(pandas_field.type, pandas.StringDtype)
    assert pandas_field.nullable is False
    assert pandas_field.metadata == {"k": "v"}


@requires_polars_and_pandas
@pytest.mark.parametrize("bytesize,arrow_type,polars_type,pandas_dtype", [
    (1, pa.int8(), polars.Int8 if polars is not None else None, pandas.Int8Dtype() if pandas is not None else None),
    (2, pa.int16(), polars.Int16 if polars is not None else None, pandas.Int16Dtype() if pandas is not None else None),
    (4, pa.int32(), polars.Int32 if polars is not None else None, pandas.Int32Dtype() if pandas is not None else None),
    (8, pa.int64(), polars.Int64 if polars is not None else None, pandas.Int64Dtype() if pandas is not None else None),
])
def test_integer_field_mappings(bytesize, arrow_type, polars_type, pandas_dtype):
    field = IntegerField("num", bytesize=bytesize, metadata={"signed": True})

    arrow_field = field.to_arrow()
    assert arrow_field.type == arrow_type
    assert arrow_field.metadata_bytes == {b"signed": b"True"}

    polars_field = field.to_polars()
    assert polars_field.type == polars_type

    pandas_field = field.to_pandas()
    assert isinstance(pandas_field.type, type(pandas_dtype))

    if pyspark is not None:
        spark_field = field.to_spark()
        spark_type = spark_field.type
        expected = {
            1: pyspark.sql.types.ByteType,
            2: pyspark.sql.types.ShortType,
            4: pyspark.sql.types.IntegerType,
            8: pyspark.sql.types.LongType,
        }[bytesize]
        assert isinstance(spark_type, expected)


@pytest.mark.parametrize("arrow_type,expected_large", [
    (pa.binary(), False),
    (pa.large_binary(), True),
])
def test_parse_arrow_binary_fields(arrow_type, expected_large):
    arrow_field = pa.field("payload", arrow_type, nullable=True, metadata={b"key": b"value"})
    parsed = BinaryField.parse(arrow_field)

    assert isinstance(parsed, BinaryField)
    assert parsed.name == "payload"
    assert parsed.nullable is True
    assert parsed.metadata == {"key": "value"}
    assert parsed._large is expected_large


@pytest.mark.parametrize("arrow_type,bytesize", [
    (pa.int16(), 2),
    (pa.int64(), 8),
])
def test_parse_arrow_integer_fields(arrow_type, bytesize):
    arrow_field = pa.field("num", arrow_type, nullable=False, metadata={b"bit": b"size"})
    parsed = IntegerField.parse(arrow_field)

    assert isinstance(parsed, IntegerField)
    assert parsed.name == "num"
    assert parsed.nullable is False
    assert parsed.metadata == {"bit": "size"}
    assert parsed._bytesize == bytesize
