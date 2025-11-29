import pyarrow as pa
import pytest

from yggdrasil.libs.pandaslib import pandas
from yggdrasil.libs.polarslib import polars
from yggdrasil.libs.sparklib import pyspark
from yggdrasil.types.fields.scalar.binary_field import BinaryField

requires_polars_and_pandas = pytest.mark.skipif(
    polars is None or pandas is None, reason="polars and pandas are required",
)


def test_binary_field_metadata_and_aliases():
    field = BinaryField("payload", large=True, nullable=False, metadata={"flag": True, "note": "hi"})

    python_field = field.to_python()
    assert python_field.name == "payload"
    assert python_field.type is bytes
    assert python_field.nullable is False
    assert python_field.metadata == {"flag": True, "note": "hi"}

    arrow_field = field.to_arrow()
    assert arrow_field.name == "payload"
    assert arrow_field.type == pa.large_binary()
    assert arrow_field.nullable is False
    assert arrow_field.metadata_bytes == {b"flag": b"true", b"note": b"hi"}
    assert arrow_field.metadata == {"flag": "true", "note": "hi"}

    if pyspark is not None:
        spark_field = field.to_spark()
        assert spark_field.name == "payload"
        assert spark_field.nullable is False
        assert spark_field.metadata_bytes == {b"flag": b"true", b"note": b"hi"}


@requires_polars_and_pandas
@pytest.mark.parametrize("large", [False, True])
def test_binary_field_polars_and_pandas(large):
    field = BinaryField("bytes", large=large, metadata={"k": "v"})

    polars_field = field.to_polars()
    assert polars_field.name == "bytes"
    assert polars_field.type == polars.Binary
    assert polars_field.metadata == {"k": "v"}

    pandas_field = field.to_pandas()
    assert pandas_field.name == "bytes"
    assert pandas_field.type is object
    assert pandas_field.metadata == {"k": "v"}
