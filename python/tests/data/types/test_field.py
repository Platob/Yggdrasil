"""Unit tests for the field module."""

import unittest
import datetime as dt
import decimal as dec
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, get_type_hints

import pyarrow as pa

from yggdrasil.types.field import (
    DataField,
    merge_dicts,
    safe_str,
    safe_bool,
    safe_metadata_str,
    annotation_args_to_metadata,
)

# For Python 3.9+ compatibility
try:
    from typing import Annotated
except ImportError:
    try:
        from typing_extensions import Annotated
    except ImportError:
        # Use dummy implementation from the module
        from yggdrasil.types.field import Annotated


class TestDataField(unittest.TestCase):
    """Test suite for the DataField class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_name = "test_field"
        self.test_comment = "Test field comment"
        self.test_metadata = {"key1": "value1", "key2": "value2"}

        # Basic test DataField instances
        self.int_field = DataField(
            name="int_field",
            arrow_type=pa.int64(),
            nullable=False,
            comment="Integer field",
            partition_key=False,
            metadata=None,
            children=None
        )

        self.str_field = DataField(
            name="str_field",
            arrow_type=pa.utf8(),
            nullable=True,
            comment="String field",
            partition_key=False,
            metadata={"encoding": "utf8"},
            children=None
        )
    def test_merge_dicts(self):
        """Test merge_dicts function."""
        # Test merging empty dicts
        self.assertEqual(merge_dicts(), {})
        self.assertEqual(merge_dicts({}, {}), {})
        
        # Test merging with None
        self.assertEqual(merge_dicts(None), {})
        self.assertEqual(merge_dicts(None, {"a": 1}), {"a": 1})
        
        # Test merging multiple dicts
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 3, "c": 4}
        d3 = {"d": 5}
        
        self.assertEqual(merge_dicts(d1, d2), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(merge_dicts(d1, d2, d3), {"a": 1, "b": 3, "c": 4, "d": 5})
        
        # Check that original dicts are not modified
        self.assertEqual(d1, {"a": 1, "b": 2})
        self.assertEqual(d2, {"b": 3, "c": 4})
        self.assertEqual(d3, {"d": 5})

    def test_safe_str(self):
        """Test safe_str function."""
        # Test with None and empty values
        self.assertIsNone(safe_str(None))
        self.assertIsNone(safe_str(""))
        self.assertEqual(safe_str(None, "default"), "default")
        
        # Test with strings
        self.assertEqual(safe_str("hello"), "hello")
        
        # Test with bytes
        self.assertEqual(safe_str(b"hello"), "hello")
        
        # Test with other types
        self.assertEqual(safe_str(123), "123")
        self.assertEqual(safe_str(True), "True")
        
    def test_safe_bool(self):
        """Test safe_bool function."""
        # Test with None and empty values
        self.assertTrue(safe_bool(None, True))
        self.assertFalse(safe_bool(None, False))
        self.assertFalse(safe_bool("", False))
        
        # Test with boolean values
        self.assertTrue(safe_bool(True, False))
        self.assertTrue(safe_bool(False, True))  # The implementation converts False to string "False"
        
        # Test with string values
        self.assertTrue(safe_bool("true", False))
        self.assertTrue(safe_bool("True", False))
        self.assertTrue(safe_bool("t", False))
        self.assertTrue(safe_bool("1", False))
        self.assertFalse(safe_bool("false", True))
        self.assertFalse(safe_bool("False", True))
        self.assertFalse(safe_bool("f", True))
        self.assertFalse(safe_bool("0", True))
        
    def test_safe_metadata_str(self):
        """Test safe_metadata_str function."""
        # Test with None and empty values
        self.assertEqual(safe_metadata_str(None), {})
        self.assertEqual(safe_metadata_str({}), {})
        
        # Test with valid metadata
        metadata = {"key1": "value1", "key2": b"value2", "key3": 123}
        expected = {"key1": "value1", "key2": "value2", "key3": "123"}
        self.assertEqual(safe_metadata_str(metadata), expected)
        
        # Test with None keys or values
        metadata = {"key1": None, None: "value2", "key3": "value3"}
        expected = {"key3": "value3"}
        self.assertEqual(safe_metadata_str(metadata), expected)
        
    def test_annotation_args_to_metadata(self):
        """Test annotation_args_to_metadata function."""
        # Test with empty args
        self.assertEqual(annotation_args_to_metadata([]), {})
        
        # Test with tuples
        args = [
            ("key1", "value1"),
            ("key2", b"value2"),
            ("key3", 123),
            ("invalid",)  # Too few elements
        ]
        expected = {"key1": "value1", "key2": "value2", "key3": "123"}
        self.assertEqual(annotation_args_to_metadata(args), expected)
        
        # Test with dictionaries
        args = [
            {"dict_key1": "value1", "dict_key2": b"value2"},
            {"dict_key3": 123}
        ]
        expected = {
            "dict_key1": "value1", 
            "dict_key2": "value2", 
            "dict_key3": "123"
        }
        self.assertEqual(annotation_args_to_metadata(args), expected)
        
        # Test with mixed types
        args = [
            ("tuple_key", "tuple_value"),
            {"dict_key": "dict_value"}
        ]
        expected = {"tuple_key": "tuple_value", "dict_key": "dict_value"}
        self.assertEqual(annotation_args_to_metadata(args), expected)
    def test_from_py_hint_basic_types(self):
        """Test DataField.from_py_hint with basic Python types."""
        # Test with basic types
        types_to_test = {
            bool: pa.bool_(),
            int: pa.int64(),
            float: pa.float64(),
            str: pa.utf8(),
            bytes: pa.binary(),
            dt.datetime: pa.timestamp("us"),
            dt.date: pa.date32(),
            dec.Decimal: pa.decimal128(38, 18),
        }
        
        for py_type, expected_arrow_type in types_to_test.items():
            field = DataField.from_py_hint(hint=py_type, name="test_field")
            self.assertEqual(field.name, "test_field")
            self.assertEqual(field.arrow_type, expected_arrow_type)
            self.assertFalse(field.nullable)
            self.assertIsNone(field.children)
    
    def test_from_py_hint_optional_types(self):
        """Test DataField.from_py_hint with Optional[T] and Union types."""
        # Test Optional[int] == Union[int, None]
        field = DataField.from_py_hint(hint=Optional[int], name="optional_int")
        self.assertEqual(field.arrow_type, pa.int64())
        self.assertTrue(field.nullable)
        
        # Test Union[int, None] explicitly
        field = DataField.from_py_hint(hint=Union[int, type(None)], name="union_int_none")
        self.assertEqual(field.arrow_type, pa.int64())
        self.assertTrue(field.nullable)
        
        
        # Test overriding nullable
        field = DataField.from_py_hint(hint=Optional[int], name="non_null_optional", nullable=False)
        self.assertEqual(field.arrow_type, pa.int64())
        self.assertFalse(field.nullable)
        
    def test_from_py_hint_with_arrow_type(self):
        """Test DataField.from_py_hint with PyArrow types."""
        # Test with PyArrow types directly
        arrow_types = [
            pa.int8(),
            pa.int16(),
            pa.int32(),
            pa.int64(),
            pa.uint8(),
            pa.uint16(),
            pa.uint32(),
            pa.uint64(),
            pa.float32(),
            pa.float64(),
            pa.string(),
            pa.binary(),
            pa.bool_(),
            pa.timestamp("s"),
            pa.date32(),
            pa.decimal128(10, 2)
        ]
        
        for arrow_type in arrow_types:
            field = DataField.from_py_hint(hint=arrow_type, name="arrow_type_field")
            self.assertEqual(field.name, "arrow_type_field")
            self.assertEqual(field.arrow_type, arrow_type)
            self.assertFalse(field.nullable)
            self.assertIsNone(field.children)
            
    def test_from_py_hint_with_metadata(self):
        """Test DataField.from_py_hint with metadata."""
        # Test with comment
        field = DataField.from_py_hint(
            hint=int, 
            name="int_field", 
            comment="Test comment"
        )
        self.assertEqual(field.comment, "Test comment")
        
        # Test with partition_key
        field = DataField.from_py_hint(
            hint=int, 
            name="int_field", 
            partition_key=True
        )
        self.assertTrue(field.partition_key)
        
        # Test with custom metadata
        metadata = {"encoding": "utf8", "description": "Test description"}
        field = DataField.from_py_hint(
            hint=str, 
            name="str_field", 
            metadata=metadata.copy()
        )
        self.assertEqual(field.metadata, metadata)
        
        # Test with timestamp unit and timezone
        metadata = {"unit": "ms", "timezone": "UTC"}
        field = DataField.from_py_hint(
            hint=dt.datetime, 
            name="timestamp_field", 
            metadata=metadata.copy()
        )
        self.assertEqual(field.arrow_type, pa.timestamp("ms", "UTC"))
        
        # Test with decimal precision and scale
        metadata = {"precision": "20", "scale": "5"}
        field = DataField.from_py_hint(
            hint=dec.Decimal, 
            name="decimal_field", 
            metadata=metadata.copy()
        )
        self.assertEqual(field.arrow_type, pa.decimal128(20, 5))

    def test_from_py_hint_list_type(self):
        """Test DataField.from_py_hint with List[T] types."""
        # Test List[int]
        field = DataField.from_py_hint(hint=List[int], name="int_list")
        self.assertEqual(field.name, "int_list")
        self.assertTrue(pa.types.is_list(field.arrow_type))
        self.assertFalse(field.nullable)
        
        # Check the list's item type
        self.assertEqual(len(field.children), 1)
        item_field = field.children[0]
        self.assertEqual(item_field.name, "item")
        self.assertEqual(item_field.arrow_type, pa.int64())
        
        # Test List[str]
        field = DataField.from_py_hint(hint=List[str], name="str_list")
        self.assertTrue(pa.types.is_list(field.arrow_type))
        item_field = field.children[0]
        self.assertEqual(item_field.arrow_type, pa.utf8())
        
        # Test List without type parameter (defaults to str)
        field = DataField.from_py_hint(hint=list, name="generic_list")
        self.assertTrue(pa.types.is_list(field.arrow_type))
        item_field = field.children[0]
        self.assertEqual(item_field.arrow_type, pa.utf8())
        
        # Test fixed-size list
        field = DataField.from_py_hint(
            hint=List[int], 
            name="fixed_size_list",
            metadata={"fixed_size": "3"}
        )
        self.assertTrue(pa.types.is_list(field.arrow_type))
        self.assertEqual(field.arrow_type.list_size, 3)
        item_field = field.children[0]
        self.assertEqual(item_field.arrow_type, pa.int64())
        
        # Test nested lists: List[List[int]]
        field = DataField.from_py_hint(hint=List[List[int]], name="nested_list")
        self.assertTrue(pa.types.is_list(field.arrow_type))
        
        outer_item_field = field.children[0]
        self.assertEqual(outer_item_field.name, "item")
        self.assertTrue(pa.types.is_list(outer_item_field.arrow_type))
        
        inner_item_field = outer_item_field.children[0]
        self.assertEqual(inner_item_field.name, "item")
        self.assertEqual(inner_item_field.arrow_type, pa.int64())
    
    def test_from_py_hint_dict_type(self):
        """Test DataField.from_py_hint with Dict[K, V] types."""
        # Test Dict[str, int]
        field = DataField.from_py_hint(hint=Dict[str, int], name="str_int_dict")
        self.assertEqual(field.name, "str_int_dict")
        self.assertTrue(pa.types.is_map(field.arrow_type))
        self.assertFalse(field.nullable)
        
        # Check the map's key and value types
        self.assertEqual(len(field.children), 2)
        key_field = field.children[0]
        value_field = field.children[1]
        
        self.assertEqual(key_field.name, "key")
        self.assertEqual(key_field.arrow_type, pa.utf8())
        
        self.assertEqual(value_field.name, "value")
        self.assertEqual(value_field.arrow_type, pa.int64())
        
        # Test Dict[int, str]
        field = DataField.from_py_hint(hint=Dict[int, str], name="int_str_dict")
        self.assertTrue(pa.types.is_map(field.arrow_type))
        
        key_field = field.children[0]
        value_field = field.children[1]
        
        self.assertEqual(key_field.arrow_type, pa.int64())
        self.assertEqual(value_field.arrow_type, pa.utf8())
        
        # Test Dict without type parameters (defaults to str, str)
        field = DataField.from_py_hint(hint=dict, name="generic_dict")
        self.assertTrue(pa.types.is_map(field.arrow_type))
        
        key_field = field.children[0]
        value_field = field.children[1]
        
        self.assertEqual(key_field.arrow_type, pa.utf8())
        self.assertEqual(value_field.arrow_type, pa.utf8())
        
        # Test keys_sorted parameter
        field = DataField.from_py_hint(
            hint=Dict[str, int], 
            name="sorted_dict",
            metadata={"keys_sorted": "true"}
        )
        self.assertTrue(pa.types.is_map(field.arrow_type))
        self.assertTrue(field.arrow_type.keys_sorted)
        
        # Test nested dict: Dict[str, Dict[str, int]]
        field = DataField.from_py_hint(hint=Dict[str, Dict[str, int]], name="nested_dict")
        self.assertTrue(pa.types.is_map(field.arrow_type))
        
        key_field = field.children[0]
        value_field = field.children[1]
        
        self.assertEqual(key_field.arrow_type, pa.utf8())
        self.assertTrue(pa.types.is_map(value_field.arrow_type))
        
        inner_key_field = value_field.children[0]
        inner_value_field = value_field.children[1]
        
        self.assertEqual(inner_key_field.arrow_type, pa.utf8())
        self.assertEqual(inner_value_field.arrow_type, pa.int64())

    def test_from_py_hint_annotated_types(self):
        """Test DataField.from_py_hint with Annotated types."""
        # Test Annotated with simple metadata
        field = DataField.from_py_hint(
            hint=Annotated[int, "metadata"], 
            name="annotated_int"
        )
        self.assertEqual(field.name, "annotated_int")
        self.assertEqual(field.arrow_type, pa.int64())
        self.assertFalse(field.nullable)
        
        # Test Annotated with tuple metadata
        field = DataField.from_py_hint(
            hint=Annotated[str, ("encoding", "utf8")], 
            name="annotated_str"
        )
        self.assertEqual(field.arrow_type, pa.utf8())
        self.assertEqual(field.metadata, {"encoding": "utf8"})
        
        # Test Annotated with dict metadata
        field = DataField.from_py_hint(
            hint=Annotated[
                float, 
                {"precision": "double", "description": "A double-precision float"}
            ], 
            name="annotated_float"
        )
        self.assertEqual(field.arrow_type, pa.float64())
        self.assertEqual(
            field.metadata, 
            {"precision": "double", "description": "A double-precision float"}
        )
        
        # Test Annotated with multiple metadata items
        field = DataField.from_py_hint(
            hint=Annotated[
                dt.datetime, 
                ("unit", "ms"), 
                {"timezone": "UTC"}
            ], 
            name="annotated_datetime"
        )
        self.assertEqual(field.arrow_type, pa.timestamp("ms", "UTC"))
        
        # Test Annotated with comment and partition_key in metadata
        field = DataField.from_py_hint(
            hint=Annotated[
                str, 
                {"comment": "Test comment", "partition_key": "true"}
            ], 
            name="annotated_partitioned_str"
        )
        self.assertEqual(field.comment, "Test comment")
        self.assertTrue(field.partition_key)
        
        # Test Annotated with decimal precision and scale
        field = DataField.from_py_hint(
            hint=Annotated[
                dec.Decimal, 
                {"precision": "10", "scale": "2"}
            ], 
            name="annotated_decimal"
        )
        self.assertTrue(pa.types.is_decimal(field.arrow_type))
        self.assertEqual(field.arrow_type.precision, 10)
        self.assertEqual(field.arrow_type.scale, 2)
        
        # Test combining Annotated and Optional
        field = DataField.from_py_hint(
            hint=Optional[Annotated[int, ("description", "Optional integer")]],
            name="optional_annotated_int"
        )
        self.assertEqual(field.arrow_type, pa.int64())
        self.assertTrue(field.nullable)
        self.assertEqual(field.metadata, {"description": "Optional integer"})

    def test_dataclass_with_defaults(self):
        """Test DataField.from_py_hint with dataclasses with default values."""
        @dataclass
        class Person:
            name: str
            age: int = 30
            active: bool = True
        
        # Create a field from the dataclass type
        field = DataField.from_py_hint(hint=Person, name="person")
        
        # Check the field properties
        self.assertEqual(field.name, "person")
        self.assertTrue(pa.types.is_struct(field.arrow_type))
        self.assertFalse(field.nullable)
        
        # Check children fields
        self.assertEqual(len(field.children), 3)
        
        name_field = field.children[0]
        self.assertEqual(name_field.name, "name")
        self.assertEqual(name_field.arrow_type, pa.utf8())
        
        age_field = field.children[1]
        self.assertEqual(age_field.name, "age")
        self.assertEqual(age_field.arrow_type, pa.int64())
        
        active_field = field.children[2]
        self.assertEqual(active_field.name, "active")
        self.assertEqual(active_field.arrow_type, pa.bool_())

    def test_nested_dataclasses(self):
        """Test DataField.from_py_hint with nested dataclasses."""
        @dataclass
        class Address:
            street: str
            city: str
            zip_code: str
        
        @dataclass
        class Person:
            name: str
            age: int
            address: Address
        
        # Create a field from the outer dataclass type
        field = DataField.from_py_hint(hint=Person, name="person")
        
        # Check the field properties
        self.assertEqual(field.name, "person")
        self.assertTrue(pa.types.is_struct(field.arrow_type))
        
        # Check children fields
        self.assertEqual(len(field.children), 3)
        
        # Check the nested address field
        address_field = field.children[2]
        self.assertEqual(address_field.name, "address")
        self.assertTrue(pa.types.is_struct(address_field.arrow_type))
        
        # Check the nested address field's children
        self.assertEqual(len(address_field.children), 3)
        
        street_field = address_field.children[0]
        self.assertEqual(street_field.name, "street")
        self.assertEqual(street_field.arrow_type, pa.utf8())
        
        city_field = address_field.children[1]
        self.assertEqual(city_field.name, "city")
        self.assertEqual(city_field.arrow_type, pa.utf8())
        
        zip_field = address_field.children[2]
        self.assertEqual(zip_field.name, "zip_code")
        self.assertEqual(zip_field.arrow_type, pa.utf8())

    def test_annotated_dataclass_fields(self):
        """Test DataField.from_py_hint with dataclasses using Annotated fields."""
        @dataclass
        class AnnotatedPerson:
            name: Annotated[str, {"description": "Person name"}]
            age: Annotated[int, {"description": "Person age"}]
            email: Annotated[str, {"description": "Email address", "pattern": r"^.+@.+\..+$"}]
        
        # Create a field from the dataclass type
        field = DataField.from_py_hint(hint=AnnotatedPerson, name="annotated_person")
        
        # Check the field properties
        self.assertEqual(field.name, "annotated_person")
        self.assertTrue(pa.types.is_struct(field.arrow_type))
        
        # Check children fields and their metadata
        self.assertEqual(len(field.children), 3)
        
        name_field = field.children[0]
        self.assertEqual(name_field.name, "name")
        self.assertEqual(name_field.arrow_type, pa.utf8())
        self.assertEqual(name_field.metadata, {"description": "Person name"})
        
        age_field = field.children[1]
        self.assertEqual(age_field.name, "age")
        self.assertEqual(age_field.arrow_type, pa.int64())
        self.assertEqual(age_field.metadata, {"description": "Person age"})
        
        email_field = field.children[2]
        self.assertEqual(email_field.name, "email")
        self.assertEqual(email_field.arrow_type, pa.utf8())
        self.assertEqual(email_field.metadata, {"description": "Email address", "pattern": r"^.+@.+\..+$"})

    def test_to_arrow_field(self):
        """Test DataField.to_arrow_field method."""
        # Test with a basic field
        arrow_field = self.int_field.to_arrow_field()
        self.assertIsInstance(arrow_field, pa.Field)
        self.assertEqual(arrow_field.name, "int_field")
        self.assertEqual(arrow_field.type, pa.int64())
        self.assertFalse(arrow_field.nullable)
        
        # Test with a field that has metadata
        arrow_field = self.str_field.to_arrow_field()
        self.assertIsInstance(arrow_field, pa.Field)
        self.assertEqual(arrow_field.name, "str_field")
        self.assertEqual(arrow_field.type, pa.utf8())
        self.assertTrue(arrow_field.nullable)
        self.assertEqual(arrow_field.metadata, {b"encoding": b"utf8"})
        
        # Test with a nested field (struct)
        @dataclass
        class Person:
            name: str
            age: int
        
        field = DataField.from_py_hint(hint=Person, name="person")
        arrow_field = field.to_arrow_field()
        
        self.assertIsInstance(arrow_field, pa.Field)
        self.assertEqual(arrow_field.name, "person")
        self.assertTrue(pa.types.is_struct(arrow_field.type))

    def test_to_arrow_schema(self):
        """Test DataField.to_arrow_schema method."""
        # Test with a struct field (which has children)
        @dataclass
        class Person:
            name: str
            age: int
        
        field = DataField.from_py_hint(hint=Person, name="person")
        schema = field.to_arrow_schema()
        
        self.assertIsInstance(schema, pa.Schema)
        self.assertEqual(len(schema.names), 2)
        self.assertEqual(schema.names, ["name", "age"])
        self.assertEqual(schema.types, [pa.utf8(), pa.int64()])
        
        # Test with a field that has no children
        schema = self.int_field.to_arrow_schema()
        self.assertEqual(len(schema.names), 0)
        
        # Test with a field that has metadata
        field = DataField.from_py_hint(
            hint=Person, 
            name="person_with_metadata", 
            metadata={"version": "1.0"}
        )
        schema = field.to_arrow_schema()
        
        self.assertEqual(schema.metadata, {b"version": b"1.0"})
        
    def test_to_arrow_schema_no_children(self):
        """Test DataField.to_arrow_schema with no children."""
        # Create a field without children
        field = DataField.from_py_hint(hint=int, name="int_field")
        
        # The schema should be empty
        schema = field.to_arrow_schema()
        self.assertEqual(len(schema.names), 0)
    def test_equality_method(self):
        """Test DataField.__eq__ method."""
        # Create two identical fields
        field1 = DataField(
            name="test_field",
            arrow_type=pa.int64(),
            nullable=False,
            comment="Test comment",
            partition_key=False,
            metadata=None,
            children=None
        )
        
        field2 = DataField(
            name="test_field",
            arrow_type=pa.int64(),
            nullable=False,
            comment="Test comment",
            partition_key=False,
            metadata=None,
            children=None
        )
        
        # They should be equal
        self.assertEqual(field1, field2)
        
        # Create a field with different name
        field3 = DataField(
            name="different_name",
            arrow_type=pa.int64(),
            nullable=False,
            comment="Test comment",
            partition_key=False,
            metadata=None,
            children=None
        )
        
        # They should not be equal
        self.assertNotEqual(field1, field3)
        
        # Create a field with different arrow_type
        field4 = DataField(
            name="test_field",
            arrow_type=pa.float64(),
            nullable=False,
            comment="Test comment",
            partition_key=False,
            metadata=None,
            children=None
        )
        
        # They should not be equal
        self.assertNotEqual(field1, field4)
        
        # Note: The equality check only considers name and arrow_type
        # Create a field with different nullable value
        field5 = DataField(
            name="test_field",
            arrow_type=pa.int64(),
            nullable=True,  # Different from field1
            comment="Test comment",
            partition_key=False,
            metadata=None,
            children=None
        )
        
        # They should still be equal because only name and arrow_type are considered
        self.assertEqual(field1, field5)

    def test_hash_method(self):
        """Test DataField.__hash__ method."""
        # Create two identical fields
        field1 = DataField(
            name="test_field",
            arrow_type=pa.int64(),
            nullable=False,
            comment="Test comment",
            partition_key=False,
            metadata=None,
            children=None
        )
        
        field2 = DataField(
            name="test_field",
            arrow_type=pa.int64(),
            nullable=False,
            comment="Test comment",
            partition_key=False,
            metadata=None,
            children=None
        )
        
        # Their hashes should be equal
        self.assertEqual(hash(field1), hash(field2))
        
        # Create a field with different name
        field3 = DataField(
            name="different_name",
            arrow_type=pa.int64(),
            nullable=False,
            comment="Test comment",
            partition_key=False,
            metadata=None,
            children=None
        )
        
        # Their hashes should be different
        self.assertNotEqual(hash(field1), hash(field3))
        
        # Create a field with different arrow_type
        field4 = DataField(
            name="test_field",
            arrow_type=pa.float64(),
            nullable=False,
            comment="Test comment",
            partition_key=False,
            metadata=None,
            children=None
        )
        
        # Their hashes should be different
        self.assertNotEqual(hash(field1), hash(field4))
        
        # Add fields to a set to test hash functionality
        field_set = {field1, field2, field3, field4}
        self.assertEqual(len(field_set), 3)  # field1 and field2 are duplicates

if __name__ == "__main__":
    unittest.main()
