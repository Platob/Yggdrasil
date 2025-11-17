"""Unit tests for the spark_utils module."""

import pytest
import pyarrow as pa

# Check if Spark is available
has_spark = False
try:
    import pyspark.sql as spark_sql
    import pyspark.sql.types as spark_types
    has_spark = True
except ImportError:
    # Create mock classes for type hints
    class MockSparkTypes:
        class DataType:
            pass
        class StructField:
            pass
    spark_types = MockSparkTypes()

# Skip all tests if Spark is not available
pytestmark = pytest.mark.skipif(not has_spark, reason="PySpark not installed")

# Only import from spark_utils if Spark is available
if has_spark:
    from yggdrasil.libutils.spark_utils import (
        ARROW_TYPE_TO_SPARK_TYPE,
        spark_to_arrow_type,
        cast_nested_spark_field,
        safe_spark_dataframe
    )


@pytest.fixture
def spark_session():
    """Create a local Spark session for testing."""
    if not has_spark:
        pytest.skip("PySpark not installed")
        return None

    from pyspark.sql import SparkSession

    # Try to create a SparkSession with minimal resources
    try:
        session = (
            SparkSession.builder
            .master("local[1]")
            .appName("yggdrasil-test")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.driver.memory", "1g")
            .config("spark.executor.memory", "1g")
            .getOrCreate()
        )
    except Exception as e:
        pytest.skip(f"Failed to create SparkSession: {e}")
        return None

    yield session

    # Clean up after test
    if session:
        try:
            session.stop()
        except:
            pass


@pytest.mark.skipif(not has_spark, reason="PySpark not installed")
class TestArrowTypeToSparkType:
    """Test the ARROW_TYPE_TO_SPARK_TYPE mapping."""

    def test_primitive_type_mappings(self):
        """Test that primitive Arrow types map to the expected Spark types."""
        # Check boolean mapping
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.bool_()], spark_types.BooleanType)

        # Check string mappings
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.utf8()], spark_types.StringType)
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.large_string()], spark_types.StringType)

        # Check binary mappings
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.binary()], spark_types.BinaryType)
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.large_binary()], spark_types.BinaryType)

        # Check integer mappings
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.int8()], spark_types.ByteType)
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.int16()], spark_types.IntegerType)
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.int32()], spark_types.IntegerType)
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.int64()], spark_types.LongType)

        # Check float mappings
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.float32()], spark_types.FloatType)
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.float64()], spark_types.DoubleType)

        # Check date/time mappings
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.date32()], spark_types.DateType)
        assert isinstance(ARROW_TYPE_TO_SPARK_TYPE[pa.date64()], spark_types.TimestampType)


@pytest.mark.skipif(not has_spark, reason="PySpark not installed")
class TestSparkToArrowType:
    """Test the spark_to_arrow_type function."""

    def test_primitive_types(self):
        """Test conversion of primitive Spark types to Arrow types."""
        # Boolean type
        assert spark_to_arrow_type(spark_types.BooleanType()) == pa.bool_()

        # Integer types
        assert spark_to_arrow_type(spark_types.ByteType()) == pa.int8()
        assert spark_to_arrow_type(spark_types.ShortType()) == pa.int16()
        assert spark_to_arrow_type(spark_types.IntegerType()) == pa.int32()
        assert spark_to_arrow_type(spark_types.LongType()) == pa.int64()

        # Float types
        assert spark_to_arrow_type(spark_types.FloatType()) == pa.float32()
        assert spark_to_arrow_type(spark_types.DoubleType()) == pa.float64()

        # String and binary types
        assert spark_to_arrow_type(spark_types.StringType()) == pa.utf8()
        assert spark_to_arrow_type(spark_types.BinaryType()) == pa.binary()

        # Date and timestamp types
        assert spark_to_arrow_type(spark_types.DateType()) == pa.date32()
        assert spark_to_arrow_type(spark_types.TimestampType()) == pa.timestamp('us', "UTC")
        assert spark_to_arrow_type(spark_types.TimestampNTZType()) == pa.timestamp('us')

    def test_decimal_type(self):
        """Test conversion of Spark decimal type to Arrow decimal type."""
        decimal_type = spark_types.DecimalType(10, 2)
        arrow_type = spark_to_arrow_type(decimal_type)
        assert pa.types.is_decimal(arrow_type)
        assert arrow_type.precision == 10
        assert arrow_type.scale == 2

    def test_array_type(self):
        """Test conversion of Spark array type to Arrow list type."""
        # Array of integers
        array_type = spark_types.ArrayType(spark_types.IntegerType())
        arrow_type = spark_to_arrow_type(array_type)

        assert pa.types.is_list(arrow_type)
        assert arrow_type.value_type == pa.int32()

        # Nested array
        nested_array_type = spark_types.ArrayType(
            spark_types.ArrayType(spark_types.StringType())
        )
        arrow_nested_type = spark_to_arrow_type(nested_array_type)

        assert pa.types.is_list(arrow_nested_type)
        assert pa.types.is_list(arrow_nested_type.value_type)
        assert arrow_nested_type.value_type.value_type == pa.utf8()

    def test_map_type(self):
        """Test conversion of Spark map type to Arrow map type."""
        # Map with string keys and integer values
        map_type = spark_types.MapType(
            spark_types.StringType(),
            spark_types.IntegerType()
        )
        arrow_type = spark_to_arrow_type(map_type)

        assert pa.types.is_map(arrow_type)
        assert arrow_type.key_type == pa.utf8()
        assert arrow_type.item_type == pa.int32()

    def test_struct_type(self):
        """Test conversion of Spark struct type to Arrow struct type."""
        # Simple struct with two fields
        struct_type = spark_types.StructType([
            spark_types.StructField("name", spark_types.StringType(), True),
            spark_types.StructField("age", spark_types.IntegerType(), False)
        ])
        arrow_type = spark_to_arrow_type(struct_type)

        assert pa.types.is_struct(arrow_type)
        assert len(arrow_type) == 2

        name_field = arrow_type.field("name")
        age_field = arrow_type.field("age")

        assert name_field.type == pa.utf8()
        assert name_field.nullable

        assert age_field.type == pa.int32()
        assert not age_field.nullable

    def test_nested_struct_type(self):
        """Test conversion of nested Spark struct type to Arrow struct type."""
        # Create a nested struct type
        address_struct = spark_types.StructType([
            spark_types.StructField("street", spark_types.StringType(), True),
            spark_types.StructField("city", spark_types.StringType(), True)
        ])

        person_struct = spark_types.StructType([
            spark_types.StructField("name", spark_types.StringType(), True),
            spark_types.StructField("address", address_struct, True)
        ])

        arrow_type = spark_to_arrow_type(person_struct)

        assert pa.types.is_struct(arrow_type)
        assert len(arrow_type) == 2

        address_field = arrow_type.field("address")
        assert pa.types.is_struct(address_field.type)
        assert len(address_field.type) == 2
        assert address_field.type.field("street").type == pa.utf8()
        assert address_field.type.field("city").type == pa.utf8()

    def test_unsupported_type(self):
        """Test that TypeError is raised for unsupported types."""
        class UnsupportedType(spark_types.DataType):
            pass

        with pytest.raises(TypeError):
            spark_to_arrow_type(UnsupportedType())


@pytest.mark.skipif(not has_spark, reason="PySpark not installed")
class TestCastNestedSparkField:
    """Test the cast_nested_spark_field function."""

    def test_primitive_field_cast(self, spark_session):
        """Test casting a primitive field."""
        if spark_session is None:
            pytest.skip("Failed to create Spark session")

        # Create a test DataFrame
        df = spark_session.createDataFrame(
            [(1, "test")],
            "id INT, name STRING"
        )

        # Define source and target fields
        source_field = spark_types.StructField("id", spark_types.IntegerType(), True)
        target_field = spark_types.StructField("id_long", spark_types.LongType(), True)

        # Cast the column
        result = cast_nested_spark_field(
            df["id"],
            source_field=source_field,
            target_field=target_field
        )

        # Apply the cast and check the result
        casted_df = df.select(result)
        assert casted_df.schema[0].name == "id_long"
        assert isinstance(casted_df.schema[0].dataType, spark_types.LongType)

    def test_struct_field_cast(self, spark_session):
        """Test casting a struct field."""
        if spark_session is None:
            pytest.skip("Failed to create Spark session")

        # Create a test DataFrame with a struct
        df = spark_session.createDataFrame(
            [(1, {"first": "John", "last": "Doe"})],
            "id INT, name STRUCT<first: STRING, last: STRING>"
        )

        # Define source struct field
        source_struct = spark_types.StructType([
            spark_types.StructField("first", spark_types.StringType(), True),
            spark_types.StructField("last", spark_types.StringType(), True)
        ])
        source_field = spark_types.StructField("name", source_struct, True)

        # Define target struct field with different field order and added field
        target_struct = spark_types.StructType([
            spark_types.StructField("last", spark_types.StringType(), True),
            spark_types.StructField("first", spark_types.StringType(), True),
            spark_types.StructField("middle", spark_types.StringType(), True)
        ])
        target_field = spark_types.StructField("person", target_struct, True)

        # Cast the column
        result = cast_nested_spark_field(
            df["name"],
            source_field=source_field,
            target_field=target_field
        )

        # Apply the cast and check the result
        casted_df = df.select(result)
        assert casted_df.schema[0].name == "person"
        assert isinstance(casted_df.schema[0].dataType, spark_types.StructType)
        assert len(casted_df.schema[0].dataType.fields) == 3
        assert casted_df.schema[0].dataType.fieldNames() == ["last", "first", "middle"]


@pytest.mark.skipif(not has_spark, reason="PySpark not installed")
class TestSafeSparkDataFrame:
    """Test the safe_spark_dataframe function."""

    def test_with_spark_dataframe(self, spark_session):
        """Test that Spark DataFrames are returned as-is."""
        if spark_session is None:
            pytest.skip("Failed to create Spark session")

        df = spark_session.createDataFrame(
            [(1, "test")],
            "id INT, name STRING"
        )
        result = safe_spark_dataframe(df, spark_session)
        assert result is df

    def test_with_arrow_table(self, spark_session):
        """Test conversion from PyArrow Table."""
        if spark_session is None:
            pytest.skip("Failed to create Spark session")

        # Create an Arrow table
        table = pa.table({
            'id': pa.array([1, 2, 3], type=pa.int32()),
            'name': pa.array(['a', 'b', 'c'], type=pa.utf8())
        })

        # Convert to Spark DataFrame
        result = safe_spark_dataframe(table, spark_session)

        # Check the result
        assert isinstance(result, spark_sql.DataFrame)
        assert result.count() == 3
        assert result.schema.names == ['id', 'name']


if __name__ == "__main__":
    pytest.main()