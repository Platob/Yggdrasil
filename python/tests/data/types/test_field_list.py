"""Test for DataField from_py_hint with list types."""

import unittest
import pyarrow as pa
from typing import List

from yggdrasil.types.field import DataField

class TestDataFieldList(unittest.TestCase):
    def test_from_py_hint_list_type(self):
        """Test DataField.from_py_hint with List[T] types."""
        # Test List[int]
        field = DataField.from_py_hint(hint=List[int], name="int_list", metadata={})
        self.assertEqual(field.name, "int_list")
        self.assertTrue(pa.types.is_list(field.arrow_type))
        self.assertFalse(field.nullable)
        
        # Check the list's item type
        self.assertEqual(len(field.children), 1)
        item_field = field.children[0]
        self.assertEqual(item_field.name, "item")
        self.assertEqual(item_field.arrow_type, pa.int64())
        
        # Test fixed-size list
        field = DataField.from_py_hint(
            hint=List[int], 
            name="fixed_size_list",
            metadata={"fixed_size": "3"}
        )
        self.assertTrue(pa.types.is_fixed_size_list(field.arrow_type))
        self.assertEqual(field.arrow_type.list_size, 3)
        item_field = field.children[0]
        self.assertEqual(item_field.arrow_type, pa.int64())

if __name__ == "__main__":
    unittest.main()
