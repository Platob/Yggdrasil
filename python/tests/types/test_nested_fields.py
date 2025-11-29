import pyarrow as pa

from yggdrasil.libs.sparklib import pyspark
from yggdrasil.types.fields.nested.list_field import ListField
from yggdrasil.types.fields.nested.map_field import MapField
from yggdrasil.types.fields.nested.struct_field import StructField
from yggdrasil.types.fields.scalar.integer_field import IntegerField
from yggdrasil.types.fields.scalar.string_field import StringField


def test_struct_field_to_arrow_and_python():
    child_name = StringField("name")
    child_age = IntegerField("age", bytesize=4, nullable=False)

    field = StructField("person", [child_name, child_age], nullable=False, metadata={"level": 1})

    arrow_field = field.to_arrow()
    expected_type = pa.struct([child_name.to_arrow().inner, child_age.to_arrow().inner])

    assert arrow_field.name == "person"
    assert arrow_field.type == expected_type
    assert arrow_field.nullable is False
    assert arrow_field.metadata_bytes == {b"level": b"1"}

    python_field = field.to_python()
    assert python_field.type is dict
    assert python_field.nullable is False
    assert python_field.metadata == {"level": 1}
    assert [child.name for child in python_field.fields] == ["name", "age"]

    if pyspark is not None:
        spark_field = field.to_spark()
        assert spark_field.name == "person"
        assert isinstance(spark_field.type, pyspark.sql.types.StructType)


def test_list_field_to_arrow_and_python():
    value_field = IntegerField("value", bytesize=2)
    field = ListField("values", value_field, metadata={"description": "numbers"})

    arrow_field = field.to_arrow()
    assert arrow_field.type == pa.list_(pa.int16())
    assert arrow_field.metadata_bytes == {b"description": b"numbers"}

    python_field = field.to_python()
    assert python_field.type is list
    assert python_field.metadata == {"description": "numbers"}
    assert python_field.value_field.name == "value"

    if pyspark is not None:
        spark_field = field.to_spark()
        assert isinstance(spark_field.type, pyspark.sql.types.ArrayType)


def test_map_field_to_arrow_and_python():
    key_field = StringField("key")
    value_field = IntegerField("value")
    field = MapField("mapping", key_field, value_field, metadata={"hint": True})

    arrow_field = field.to_arrow()
    expected_type = pa.map_(pa.string(), pa.int64())
    assert arrow_field.type == expected_type
    assert arrow_field.metadata_bytes == {b"hint": b"true"}

    python_field = field.to_python()
    assert python_field.type is dict
    assert python_field.metadata == {"hint": True}
    assert python_field.key_field.name == "key"
    assert python_field.value_field.name == "value"

    if pyspark is not None:
        spark_field = field.to_spark()
        assert isinstance(spark_field.type, pyspark.sql.types.MapType)
