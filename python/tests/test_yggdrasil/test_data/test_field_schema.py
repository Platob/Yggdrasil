from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.field import Field
from yggdrasil.data.schema import Schema



def test_field_from_to_pyspark() -> None:
    T = pytest.importorskip("pyspark.sql.types")
    field = Field.from_pyspark(name="id", dtype=T.IntegerType(), nullable=False)

    assert field.name == "id"
    assert field.arrow_type == pa.int32()
    assert field.nullable is False

    spark_field = field.to_pyspark_field()
    assert spark_field.name == "id"
    assert isinstance(spark_field.dataType, T.IntegerType)
    assert spark_field.nullable is False


def test_schema_from_to_pyspark() -> None:
    T = pytest.importorskip("pyspark.sql.types")
    spark_schema = T.StructType(
        [
            T.StructField("id", T.IntegerType(), False),
            T.StructField("name", T.StringType(), True),
        ]
    )

    schema = Schema.from_pyspark(spark_schema)

    assert schema.names == ("id", "name")
    assert schema["id"].arrow_type == pa.int32()
    assert schema["id"].nullable is False
    assert schema["name"].arrow_type == pa.string()

    restored = schema.to_spark_schema()
    assert restored.fieldNames() == ["id", "name"]
    assert isinstance(restored["id"].dataType, T.IntegerType)
    assert restored["id"].nullable is False


def test_schema_cast_table_and_unstructured() -> None:
    schema = Schema.from_fields(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("name", pa.string(), nullable=True),
        ]
    )

    input_table = pa.table({"id": ["1", "2"], "name": ["a", "b"]})
    casted_table = schema.cast_table(input_table)

    assert casted_table.schema.field("id").type == pa.int64()
    assert casted_table.column("id").to_pylist() == [1, 2]

    casted_from_rows = schema.cast_unstructured(
        [{"id": "3", "name": "c"}, {"id": "4", "name": "d"}]
    )
    assert isinstance(casted_from_rows, pa.Table)
    assert casted_from_rows.column("id").to_pylist() == [3, 4]
