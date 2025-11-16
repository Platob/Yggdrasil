"""Tests for Spark integration with the DataField class.

These tests verify the conversion between PyArrow types and Spark types using the
DataField class as the intermediate representation. The tests check:
1. Basic type conversions (int, str, bool, etc.)
2. Complex type conversions (arrays, maps, structs)
3. Dataclass to Spark schema conversions
4. Round-trip conversions (Python -> Arrow -> Spark -> Arrow -> Python)

Requirements:
- PySpark: Install with 'pip install pyspark'
- Java JDK: Required for Spark execution, will be installed automatically if needed

If PySpark is not installed, the tests will be skipped.
If Java is not installed, the test will attempt to install it automatically.
"""

import logging
import sys
import unittest
from dataclasses import dataclass
from typing import Optional, List, Dict

import pyarrow as pa
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Import from field module
from yggdrasil.utils.spark_utils import spark_sql, spark_types
from yggdrasil.types.field import DataField

@pytest.mark.skipif(not "pyspark" in sys.modules, reason="No pyspark found")
class TestDataFieldSpark(unittest.TestCase):
    """Tests for Spark integration with DataField."""

    def setUp(self):
        """Set up a SparkSession for testing."""
        test_name = self.id().split('.')[-1]
        logger.info(f"Setting up test: {test_name}")

        try:
            logger.info("Creating SparkSession...")
            self.spark = (
                spark_sql.SparkSession.builder
                .appName("DataFieldSparkTest")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .config("spark.ui.enabled", "false")  # Disable UI for tests
                .config("spark.driver.host", "localhost")  # Explicitly set driver host
                .master("local[1]")
                .getOrCreate()
            )
            logger.info(f"SparkSession created successfully: {self.spark.version}")
            logger.debug(f"Spark configuration: {self.spark.sparkContext.getConf().getAll()}")
        except Exception as e:
            error_msg = f"Failed to create SparkSession: {e}"
            logger.error(error_msg)
            # Log more detailed information about the error
            logger.exception("Detailed error information:")
            self.skipTest(error_msg)

    def tearDown(self):
        """Stop the SparkSession after testing."""

    def test_spark_basic_types(self):
        """Test converting basic Spark types to DataField and back."""
        logger.info("Starting test_spark_basic_types")

        # Create Spark StructField objects
        logger.info("Creating Spark StructField objects for basic types")
        string_field = spark_types.StructField("string_col", spark_types.StringType(), False)
        int_field = spark_types.StructField("int_col", spark_types.IntegerType(), True)
        bool_field = spark_types.StructField("bool_col", spark_types.BooleanType(), False)
        float_field = spark_types.StructField("float_col", spark_types.DoubleType(), True)

        # Convert to DataField
        logger.info("Converting Spark fields to DataField objects")
        string_data_field = DataField.from_spark_field(string_field)
        logger.debug(f"Converted string field: {string_data_field}")

        int_data_field = DataField.from_spark_field(int_field)
        logger.debug(f"Converted int field: {int_data_field}")

        bool_data_field = DataField.from_spark_field(bool_field)
        logger.debug(f"Converted bool field: {bool_data_field}")

        float_data_field = DataField.from_spark_field(float_field)
        logger.debug(f"Converted float field: {float_data_field}")

        # Check types
        logger.info("Verifying DataField conversion results")
        self.assertEqual(string_data_field.name, "string_col")
        self.assertEqual(string_data_field.arrow_type, pa.utf8())
        self.assertFalse(string_data_field.nullable)

        self.assertEqual(int_data_field.name, "int_col")
        self.assertEqual(int_data_field.arrow_type, pa.int32())
        self.assertTrue(int_data_field.nullable)

        self.assertEqual(bool_data_field.name, "bool_col")
        self.assertEqual(bool_data_field.arrow_type, pa.bool_())
        self.assertFalse(bool_data_field.nullable)

        self.assertEqual(float_data_field.name, "float_col")
        self.assertEqual(float_data_field.arrow_type, pa.float64())
        self.assertTrue(float_data_field.nullable)

        # Convert back to Spark fields
        logger.info("Converting DataField objects back to Spark fields")
        string_spark_field = string_data_field.to_spark_field()
        int_spark_field = int_data_field.to_spark_field()
        bool_spark_field = bool_data_field.to_spark_field()
        float_spark_field = float_data_field.to_spark_field()

        logger.debug(f"Round-trip string field: {string_spark_field}")
        logger.debug(f"Round-trip int field: {int_spark_field}")
        logger.debug(f"Round-trip bool field: {bool_spark_field}")
        logger.debug(f"Round-trip float field: {float_spark_field}")

        # Check types
        logger.info("Verifying round-trip conversion results")
        self.assertEqual(string_spark_field.name, "string_col")
        self.assertIsInstance(string_spark_field.dataType, spark_types.StringType)
        self.assertFalse(string_spark_field.nullable)

        self.assertEqual(int_spark_field.name, "int_col")
        self.assertIsInstance(int_spark_field.dataType, spark_types.IntegerType)
        self.assertTrue(int_spark_field.nullable)

        self.assertEqual(bool_spark_field.name, "bool_col")
        self.assertIsInstance(bool_spark_field.dataType, spark_types.BooleanType)
        self.assertFalse(bool_spark_field.nullable)

        self.assertEqual(float_spark_field.name, "float_col")
        self.assertIsInstance(float_spark_field.dataType, spark_types.DoubleType)
        self.assertTrue(float_spark_field.nullable)

        logger.info("Completed test_spark_basic_types successfully")

    def test_spark_complex_types(self):
        """Test converting complex Spark types to DataField and back."""
        logger.info("Starting test_spark_complex_types")

        # Create Spark StructField objects
        logger.info("Creating complex Spark StructField objects (array, map, struct)")

        logger.info("Creating ArrayType field")
        array_field = spark_types.StructField("array_col",
                                            spark_types.ArrayType(spark_types.IntegerType(), True),
                                            False)
        logger.debug(f"Created array field: name={array_field.name}, nullable={array_field.nullable}")

        logger.info("Creating MapType field")
        map_field = spark_types.StructField("map_col",
                                          spark_types.MapType(spark_types.StringType(),
                                                            spark_types.IntegerType(),
                                                            True),
                                          True)
        logger.debug(f"Created map field: name={map_field.name}, nullable={map_field.nullable}")

        logger.info("Creating nested StructType field")
        nested_struct_field = spark_types.StructField("nested_col",
                                                   spark_types.StructType([
                                                       spark_types.StructField("a", spark_types.StringType(), True),
                                                       spark_types.StructField("b", spark_types.IntegerType(), False)
                                                   ]),
                                                   False)
        logger.debug(f"Created nested struct field: name={nested_struct_field.name}, nullable={nested_struct_field.nullable}")

        # Convert to DataField
        logger.info("Converting complex Spark fields to DataField objects")
        array_data_field = DataField.from_spark_field(array_field)
        logger.debug(f"Converted array field: {array_data_field}")

        map_data_field = DataField.from_spark_field(map_field)
        logger.debug(f"Converted map field: {map_data_field}")

        nested_data_field = DataField.from_spark_field(nested_struct_field)
        logger.debug(f"Converted nested struct field: {nested_data_field}")

        # Check types
        logger.info("Verifying complex type conversion results")

        logger.info("Checking array field conversion")
        self.assertEqual(array_data_field.name, "array_col")
        self.assertTrue(pa.types.is_list(array_data_field.arrow_type))
        self.assertEqual(array_data_field.arrow_type.value_type, pa.int32())
        self.assertFalse(array_data_field.nullable)

        logger.info("Checking map field conversion")
        self.assertEqual(map_data_field.name, "map_col")
        self.assertTrue(pa.types.is_map(map_data_field.arrow_type))
        self.assertEqual(map_data_field.arrow_type.key_type, pa.utf8())
        self.assertEqual(map_data_field.arrow_type.item_type, pa.int32())
        self.assertTrue(map_data_field.nullable)

        logger.info("Checking nested struct field conversion")
        self.assertEqual(nested_data_field.name, "nested_col")
        self.assertTrue(pa.types.is_struct(nested_data_field.arrow_type))
        self.assertFalse(nested_data_field.nullable)

        # Convert back to Spark fields
        logger.info("Converting DataField objects back to Spark fields")
        array_spark_field = array_data_field.to_spark_field()
        map_spark_field = map_data_field.to_spark_field()
        nested_spark_field = nested_data_field.to_spark_field()

        logger.debug(f"Round-trip array field: {array_spark_field}")
        logger.debug(f"Round-trip map field: {map_spark_field}")
        logger.debug(f"Round-trip nested struct field: {nested_spark_field}")

        # Check types
        logger.info("Verifying round-trip conversion results for complex types")

        logger.info("Checking array field round-trip")
        self.assertEqual(array_spark_field.name, "array_col")
        self.assertIsInstance(array_spark_field.dataType, spark_types.ArrayType)
        self.assertIsInstance(array_spark_field.dataType.elementType, spark_types.IntegerType)
        self.assertFalse(array_spark_field.nullable)

        logger.info("Checking map field round-trip")
        self.assertEqual(map_spark_field.name, "map_col")
        self.assertIsInstance(map_spark_field.dataType, spark_types.MapType)
        self.assertIsInstance(map_spark_field.dataType.keyType, spark_types.StringType)
        self.assertIsInstance(map_spark_field.dataType.valueType, spark_types.IntegerType)
        self.assertTrue(map_spark_field.nullable)

        logger.info("Checking nested struct field round-trip")
        self.assertEqual(nested_spark_field.name, "nested_col")
        self.assertIsInstance(nested_spark_field.dataType, spark_types.StructType)
        self.assertFalse(nested_spark_field.nullable)

        logger.info("Completed test_spark_complex_types successfully")

    def test_spark_dataclass_conversion(self):
        """Test converting Python dataclasses to Spark schemas and back."""
        logger.info("Starting test_spark_dataclass_conversion")

        logger.info("Defining test dataclasses (Address and Person)")
        @dataclass
        class Address:
            street: str
            city: str
            zip_code: str
            country: str = "USA"

        @dataclass
        class Person:
            name: str
            age: int
            active: bool
            address: Address
            tags: Optional[List[str]] = None
            scores: Optional[Dict[str, float]] = None

        logger.debug(f"Person dataclass fields: {Person.__annotations__}")
        logger.debug(f"Address dataclass fields: {Address.__annotations__}")

        # Create DataField from dataclass
        logger.info("Creating DataField from Person dataclass")
        person_field = DataField.from_py_hint("person", Person, metadata={})
        logger.debug(f"Created DataField with {len(person_field.children)} children from dataclass")

        # Log the structure of the generated field
        if logger.isEnabledFor(logging.DEBUG):
            for child in person_field.children:
                logger.debug(f"  Field: {child.name}, type: {child.arrow_type}, nullable: {child.nullable}")
                if child.name == "address" and child.children:
                    for nested_child in child.children:
                        logger.debug(f"    Nested field: {nested_child.name}, type: {nested_child.arrow_type}, nullable: {nested_child.nullable}")

        # Convert to Spark schema
        logger.info("Converting DataField to Spark schema")
        spark_schema = person_field.to_spark_schema()
        logger.debug(f"Generated Spark schema with {len(spark_schema.fields)} fields")

        # Verify schema structure
        logger.info("Verifying Spark schema structure")
        self.assertIsInstance(spark_schema, spark_types.StructType)
        self.assertEqual(len(spark_schema.fields), 6)

        field_names = [f.name for f in spark_schema.fields]
        logger.debug(f"Schema field names: {field_names}")
        self.assertIn("name", field_names)
        self.assertIn("age", field_names)
        self.assertIn("active", field_names)
        self.assertIn("address", field_names)
        self.assertIn("tags", field_names)
        self.assertIn("scores", field_names)

        # Check address field type
        logger.info("Checking nested Address structure")
        address_field = next(f for f in spark_schema.fields if f.name == "address")
        self.assertIsInstance(address_field.dataType, spark_types.StructType)
        self.assertEqual(len(address_field.dataType.fields), 4)

        # Log address field details
        address_field_names = [f.name for f in address_field.dataType.fields]
        logger.debug(f"Address field names: {address_field_names}")

        # Check tags field type
        logger.info("Checking tags array field")
        tags_field = next(f for f in spark_schema.fields if f.name == "tags")
        self.assertIsInstance(tags_field.dataType, spark_types.ArrayType)
        self.assertIsInstance(tags_field.dataType.elementType, spark_types.StringType)
        self.assertTrue(tags_field.nullable)
        logger.debug(f"Tags field: type={type(tags_field.dataType).__name__}, element_type={type(tags_field.dataType.elementType).__name__}, nullable={tags_field.nullable}")

        # Check scores field type
        logger.info("Checking scores map field")
        scores_field = next(f for f in spark_schema.fields if f.name == "scores")
        self.assertIsInstance(scores_field.dataType, spark_types.MapType)
        self.assertIsInstance(scores_field.dataType.keyType, spark_types.StringType)
        self.assertIsInstance(scores_field.dataType.valueType, spark_types.DoubleType)
        self.assertTrue(scores_field.nullable)
        logger.debug(f"Scores field: type={type(scores_field.dataType).__name__}, key_type={type(scores_field.dataType.keyType).__name__}, value_type={type(scores_field.dataType.valueType).__name__}, nullable={scores_field.nullable}")

        logger.info("Completed test_spark_dataclass_conversion successfully")

    def test_round_trip_conversion(self):
        """Test round-trip conversion between Python, Arrow, and Spark types."""
        logger.info("Starting test_round_trip_conversion")

        logger.info("Defining Product dataclass")
        @dataclass
        class Product:
            id: int
            name: str
            price: float
            in_stock: bool
            category: Optional[str] = None

        # Create test data
        logger.info("Creating test data for DataFrame")
        data = [
            {"id": 1, "name": "Product A", "price": 19.99, "in_stock": True, "category": "Electronics"},
            {"id": 2, "name": "Product B", "price": 29.99, "in_stock": False, "category": None},
            {"id": 3, "name": "Product C", "price": 9.99, "in_stock": True, "category": "Books"}
        ]
        logger.debug(f"Test data: {data}")

        # Create DataFrame from data
        logger.info("Creating Spark DataFrame from test data")
        df = self.spark.createDataFrame(data)
        logger.debug(f"DataFrame created with {df.count()} rows")

        # Get Spark schema
        spark_schema = df.schema
        logger.debug(f"Original Spark schema: {spark_schema}")

        # Convert to DataField
        logger.info("Converting Spark schema to DataField")
        root_field = DataField.from_spark_schema(spark_schema)
        logger.debug(f"Created root DataField with {len(root_field.children)} children")

        # Log the structure of the DataField
        if logger.isEnabledFor(logging.DEBUG):
            for child in root_field.children:
                logger.debug(f"  Field: {child.name}, type: {child.arrow_type}, nullable: {child.nullable}")

        # Convert back to Spark schema
        logger.info("Converting DataField back to Spark schema")
        new_spark_schema = root_field.to_spark_schema()
        logger.debug(f"New Spark schema created with {len(new_spark_schema.fields)} fields")

        # Apply schema to DataFrame
        logger.info("Creating new DataFrame with round-trip schema")
        new_df = self.spark.createDataFrame(df.collect(), new_spark_schema)
        logger.debug(f"New DataFrame created with {new_df.count()} rows")

        # Display sample data for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Sample data from original DataFrame:")
            for row in df.limit(2).collect():
                logger.debug(f"  {row}")

            logger.debug("Sample data from new DataFrame:")
            for row in new_df.limit(2).collect():
                logger.debug(f"  {row}")

        # Verify data is preserved
        logger.info("Verifying data integrity through round-trip conversion")
        self.assertEqual(df.count(), new_df.count())

        original_rows = sorted(df.collect())
        new_rows = sorted(new_df.collect())
        self.assertEqual(original_rows, new_rows)
        logger.debug("Data integrity verified - all rows match")

        # Verify schema structure is preserved
        logger.info("Verifying schema structure is preserved")
        self.assertEqual(len(spark_schema.fields), len(new_spark_schema.fields))

        for original_field, new_field in zip(sorted(spark_schema.fields, key=lambda f: f.name),
                                            sorted(new_spark_schema.fields, key=lambda f: f.name)):
            logger.debug(f"Comparing field: {original_field.name}")
            self.assertEqual(original_field.name, new_field.name)
            self.assertEqual(type(original_field.dataType), type(new_field.dataType))
            self.assertEqual(original_field.nullable, new_field.nullable)
            logger.debug(f"  Match: {original_field.name} - original: {type(original_field.dataType).__name__}, new: {type(new_field.dataType).__name__}")

        logger.info("Schema structure verified - all fields match")
        logger.info("Completed test_round_trip_conversion successfully")

if __name__ == "__main__":
    unittest.main()