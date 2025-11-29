import pyarrow as pa
import pytest

from yggdrasil.libs.pandaslib import pandas
from yggdrasil.libs.polarslib import polars
from yggdrasil.libs.sparklib import pyspark
from yggdrasil.types.fields.scalar.date_field import DateField
from yggdrasil.types.fields.scalar.time_field import TimeField
from yggdrasil.types.fields.scalar.timestamp_field import TimestampField

requires_polars_and_pandas = pytest.mark.skipif(
    polars is None or pandas is None, reason="polars and pandas are required",
)


def test_date_field_arrow_metadata():
    field = DateField("date", nullable=False, metadata={"k": "v"})
    arrow_field = field.to_arrow()

    assert arrow_field.type == pa.date32()
    assert arrow_field.nullable is False
    assert arrow_field.metadata_bytes == {b"k": b"v"}
    assert arrow_field.metadata == {"k": "v"}

    python_field = field.to_python()
    assert python_field.type == "date"
    assert python_field.metadata == {"k": "v"}


def test_time_field_arrow_metadata():
    field = TimeField("time", unit="us", metadata={"unit": "us"})
    arrow_field = field.to_arrow()

    assert arrow_field.type == pa.time64("us")
    assert arrow_field.metadata_bytes == {b"unit": b"us"}
    assert arrow_field.metadata == {"unit": "us"}

    python_field = field.to_python()
    assert python_field.type == "time"
    assert python_field.metadata == {"unit": "us"}


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_timestamp_field_arrow_and_python(tz):
    field = TimestampField("ts", unit="ms", tz=tz, metadata={"desc": "event"})
    arrow_field = field.to_arrow()

    assert arrow_field.type == pa.timestamp("ms", tz=tz)
    assert arrow_field.metadata_bytes == {b"desc": b"event"}
    assert arrow_field.metadata == {"desc": "event"}

    python_field = field.to_python()
    assert python_field.type == "datetime"
    assert python_field.metadata == {"desc": "event"}

    if pyspark is not None:
        spark_field = field.to_spark()
        assert spark_field.type.__class__ == pyspark.sql.types.TimestampType
        assert spark_field.metadata_bytes == {b"desc": b"event"}


@requires_polars_and_pandas
def test_temporal_fields_polars_and_pandas():
    date_field = DateField("d")
    time_field = TimeField("t")
    ts_field = TimestampField("ts", tz="UTC")

    assert date_field.to_polars().type == polars.Date
    assert time_field.to_polars().type == polars.Time
    assert ts_field.to_polars().type == polars.Datetime("ns", time_zone="UTC")

    assert date_field.to_pandas().type == "datetime64[ns]"
    assert time_field.to_pandas().type == "timedelta64[ns]"
    assert isinstance(ts_field.to_pandas().type, pandas.ArrowDtype)
