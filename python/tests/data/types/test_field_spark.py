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

import unittest
import os
import sys
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

import pyarrow as pa
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Skip tests if pyspark is not installed
logger.info("Checking if PySpark is installed")
try:
    import pyspark
    import pyspark.sql.types as spark_types
    from pyspark.sql import SparkSession
    HAVE_SPARK = True
    logger.info(f"PySpark is installed (version {pyspark.__version__})")
except ImportError:
    HAVE_SPARK = False
    logger.warning("PySpark is not installed. Spark tests will be skipped.")
    logger.debug("ImportError while importing PySpark. Install with: pip install pyspark")

# Import from field module
logger.info("Importing DataField from yggdrasil.types.field")
from yggdrasil.types.field import DataField

# Check Java availability and install if needed
logger.info("Checking Java availability")
try:
    logger.debug("Importing Java utilities from yggdrasil.utils.java")
    from yggdrasil.utils.java import is_java_installed, install_java, get_java_home

    HAVE_JAVA = False
    AUTO_INSTALL_JAVA = True  # Set to False to disable auto-installation
    logger.debug(f"AUTO_INSTALL_JAVA is set to {AUTO_INSTALL_JAVA}")

    # Check if Java is installed
    logger.info("Checking if Java is already installed")
    java_installed, java_version = is_java_installed()
    if java_installed:
        HAVE_JAVA = True
        logger.info(f"Java is installed (version {java_version})")

        # Log JAVA_HOME if available
        java_home = get_java_home()
        if java_home:
            logger.debug(f"JAVA_HOME is set to {java_home}")
        else:
            logger.debug("JAVA_HOME environment variable is not set")

    elif HAVE_SPARK and AUTO_INSTALL_JAVA:
        # Try to install Java if PySpark is available
        logger.info("Java not found. Attempting to install Java 17...")
        logger.debug("Calling install_java() from yggdrasil.utils.java")

        java_home = install_java(version="17", set_env=True, persist_env=False)
        if java_home:
            HAVE_JAVA = True
            logger.info(f"Successfully installed Java at {java_home}")
            logger.debug("JAVA_HOME environment variable has been set for this process")
        else:
            logger.warning("Failed to install Java. Spark tests will be skipped.")
            logger.debug("install_java() returned None - installation failed")
    else:
        reason = "AUTO_INSTALL_JAVA is disabled" if not AUTO_INSTALL_JAVA else "PySpark is not installed"
        logger.warning(f"Java is not installed and won't be auto-installed ({reason}). Spark tests will be skipped.")
except ImportError:
    logger.warning("Java installation utilities not available. Spark tests may be skipped if Java is not installed.")
    logger.debug("Could not import Java utilities from yggdrasil.utils.java")

    # Check if JAVA_HOME is set
    java_home = os.environ.get('JAVA_HOME', '')
    HAVE_JAVA = java_home is not None and os.path.exists(java_home)

    if HAVE_JAVA:
        logger.info(f"Java is available at JAVA_HOME: {java_home}")
    else:
        logger.warning("JAVA_HOME is not set or does not exist. Spark tests will be skipped.")

# Skip all Spark tests if either pyspark or Java is not available
SKIP_SPARK_TESTS = not HAVE_SPARK or not HAVE_JAVA
if SKIP_SPARK_TESTS:
    if not HAVE_SPARK:
        SKIP_REASON = "PySpark is not installed"
    else:
        SKIP_REASON = "Java is not available"

    logger.warning(f"Spark tests will be SKIPPED. Reason: {SKIP_REASON}")
else:
    SKIP_REASON = ""
    logger.info("All dependencies available - Spark tests will RUN")
    logger.debug(f"PySpark version: {pyspark.__version__}, Java available: {HAVE_JAVA}")

@pytest.mark.skipif(SKIP_SPARK_TESTS, reason=SKIP_REASON)
class TestDataFieldSpark(unittest.TestCase):
    """Tests for Spark integration with DataField."""

    def setUp(self):
        """Set up a SparkSession for testing."""
        test_name = self.id().split('.')[-1]
        logger.info(f"Setting up test: {test_name}")

        if SKIP_SPARK_TESTS:
            logger.warning(f"Skipping test {test_name}: {SKIP_REASON}")
            self.skipTest(SKIP_REASON)

        try:
            logger.info("Creating SparkSession...")
            self.spark = (
                SparkSession.builder
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
        test_name = self.id().split('.')[-1]
        if not SKIP_SPARK_TESTS and hasattr(self, 'spark') and self.spark is not None:
            logger.info(f"Stopping SparkSession for test: {test_name}")
            self.spark.stop()
            logger.info("SparkSession stopped successfully")
        else:
            logger.debug(f"No SparkSession to stop for test: {test_name}")

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

    def test_spark_installation_required(self):
        """Test that Spark methods properly raise ImportError when Spark is not installed."""
        logger.info("Starting test_spark_installation_required")

        # This test doesn't require an actual SparkSession, so we can run it regardless
        # of whether Spark is installed

        # If HAVE_SPARK is False, we can test the ImportError behavior
        if not HAVE_SPARK:
            logger.info("PySpark not installed - testing ImportError behavior")

            # Create a test field
            logger.debug("Creating a test DataField")
            field = DataField(
                name="test_field",
                arrow_type=pa.int64(),
                nullable=False,
                metadata=None,
                children=None
            )

            # Test methods that require Spark
            logger.info("Testing to_spark_field raises ImportError")
            with self.assertRaises(ImportError) as cm:
                field.to_spark_field()
            logger.debug(f"ImportError message: {cm.exception}")

            logger.info("Testing _arrow_to_spark_type raises ImportError")
            with self.assertRaises(ImportError) as cm:
                field._arrow_to_spark_type(pa.int64())
            logger.debug(f"ImportError message: {cm.exception}")

            logger.info("Testing from_spark_field raises ImportError")
            with self.assertRaises(ImportError) as cm:
                DataField.from_spark_field(None)
            logger.debug(f"ImportError message: {cm.exception}")

            logger.info("Testing _spark_to_arrow_type raises ImportError")
            with self.assertRaises(ImportError) as cm:
                DataField._spark_to_arrow_type(None)
            logger.debug(f"ImportError message: {cm.exception}")

            logger.info("Testing to_spark_schema raises ImportError")
            with self.assertRaises(ImportError) as cm:
                field.to_spark_schema()
            logger.debug(f"ImportError message: {cm.exception}")

            logger.info("Testing from_spark_schema raises ImportError")
            with self.assertRaises(ImportError) as cm:
                DataField.from_spark_schema(None)
            logger.debug(f"ImportError message: {cm.exception}")

            logger.info("All Spark-related methods correctly raise ImportError when PySpark is not installed")

        else:
            # PySpark is installed, so we can test the recursive conversion directly
            # without needing a SparkSession

            # Only run this test if Java is not available, otherwise we'll run the full test suite
            if not HAVE_JAVA:
                logger.info("PySpark installed but Java not available - testing recursive type conversion directly")

                # Test simple type conversion
                from pyspark.sql.types import IntegerType, StringType, BooleanType

                logger.info("Testing basic type conversions")
                logger.debug("Testing IntegerType conversion")
                self.assertEqual(DataField._spark_to_arrow_type(IntegerType()), pa.int32())

                logger.debug("Testing StringType conversion")
                self.assertEqual(DataField._spark_to_arrow_type(StringType()), pa.utf8())

                logger.debug("Testing BooleanType conversion")
                self.assertEqual(DataField._spark_to_arrow_type(BooleanType()), pa.bool_())

                # Test complex types with recursion
                from pyspark.sql.types import ArrayType, MapType, StructType, StructField

                # Array type
                logger.info("Testing ArrayType conversion")
                array_type = ArrayType(IntegerType())
                arrow_array_type = DataField._spark_to_arrow_type(array_type)
                logger.debug(f"ArrayType converted to {arrow_array_type}")
                self.assertTrue(pa.types.is_list(arrow_array_type))
                self.assertEqual(arrow_array_type.value_type, pa.int32())

                # Map type
                logger.info("Testing MapType conversion")
                map_type = MapType(StringType(), IntegerType())
                arrow_map_type = DataField._spark_to_arrow_type(map_type)
                logger.debug(f"MapType converted to {arrow_map_type}")
                self.assertTrue(pa.types.is_map(arrow_map_type))
                self.assertEqual(arrow_map_type.key_type, pa.utf8())
                self.assertEqual(arrow_map_type.item_type, pa.int32())

                # Struct type
                logger.info("Testing StructType conversion")
                struct_type = StructType([
                    StructField("name", StringType(), False),
                    StructField("age", IntegerType(), True),
                    StructField("active", BooleanType(), False)
                ])
                logger.debug(f"Created StructType with fields: {[f.name for f in struct_type.fields]}")
                arrow_struct_type = DataField._spark_to_arrow_type(struct_type)
                logger.debug(f"StructType converted to arrow struct with {len(arrow_struct_type)} fields")
                self.assertTrue(pa.types.is_struct(arrow_struct_type))
                self.assertEqual(len(arrow_struct_type), 3)

                # Verify field names and types
                logger.info("Verifying struct field names and types")
                self.assertEqual(arrow_struct_type[0].name, "name")
                self.assertEqual(arrow_struct_type[1].name, "age")
                self.assertEqual(arrow_struct_type[2].name, "active")

                self.assertEqual(arrow_struct_type[0].type, pa.utf8())
                self.assertEqual(arrow_struct_type[1].type, pa.int32())
                self.assertEqual(arrow_struct_type[2].type, pa.bool_())

                # Verify nullable flags
                logger.info("Verifying struct field nullable flags")
                self.assertFalse(arrow_struct_type[0].nullable)
                self.assertTrue(arrow_struct_type[1].nullable)
                self.assertFalse(arrow_struct_type[2].nullable)

                # Test nested complex types
                logger.info("Testing deeply nested complex type conversion")
                nested_type = ArrayType(
                    MapType(
                        StringType(),
                        StructType([
                            StructField("x", IntegerType(), False),
                            StructField("y", BooleanType(), True)
                        ])
                    )
                )
                logger.debug("Created nested type: Array<Map<String, Struct<x:Int, y:Bool>>>")

                arrow_nested_type = DataField._spark_to_arrow_type(nested_type)
                logger.debug(f"Nested type converted to {arrow_nested_type}")

                # Verify the nested structure
                self.assertTrue(pa.types.is_list(arrow_nested_type))
                self.assertTrue(pa.types.is_map(arrow_nested_type.value_type))
                self.assertTrue(pa.types.is_struct(arrow_nested_type.value_type.item_type))
                logger.debug("Nested structure verified correctly")

                logger.info("All recursive type conversions tested successfully")
            else:
                logger.info("Both PySpark and Java are available - skipping direct type conversion test")
                self.skipTest("Java is available, skipping direct type conversion test")


if __name__ == "__main__":
    unittest.main()