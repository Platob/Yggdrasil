"""Unit tests for the fake_module module."""

import pytest
import sys
from types import ModuleType

# Import fake_module functionality with proper error handling
try:
    from yggdrasil.libutils.fake_module import (
        FakeObject,
        FakeModule,
        make_fake_module
    )
except ImportError:
    pytest.skip("Failed to import fake_module", allow_module_level=True)


class TestFakeObject:
    """Test the FakeObject class."""

    def test_init(self):
        """Test initialization of FakeObject."""
        # Test with name
        obj = FakeObject("test_name")
        assert obj._name == "test_name"
        assert obj._inner == {}

        # Test without name
        obj = FakeObject()
        assert obj._name is None
        assert obj._inner == {}

    def test_getattr(self):
        """Test attribute access creates new FakeObjects with proper names."""
        obj = FakeObject("parent")
        child = obj.child
        assert isinstance(child, FakeObject)
        assert child._name == "parent.child"

        # Test nested attribute access
        grandchild = obj.child.grandchild
        assert isinstance(grandchild, FakeObject)
        assert grandchild._name == "parent.child.grandchild"

        # Test cache behavior (should return same object)
        assert obj.child is obj.child

    def test_setattr(self):
        """Test setting attributes on FakeObject."""
        obj = FakeObject("test")

        # Set a regular attribute
        obj.value = 42
        assert obj._inner["value"] == 42
        assert obj.value == 42

        # Set a callable
        def dummy_func():
            return "dummy"

        obj.func = dummy_func
        assert obj._inner["func"] == dummy_func
        assert obj.func() == "dummy"

    def test_call(self):
        """Test calling behavior of FakeObject."""
        obj = FakeObject("test_callable")

        # Calling with no args should return self
        assert obj() is obj

        # Calling with args should raise ImportError
        with pytest.raises(ImportError):
            obj(1, 2, 3)

        with pytest.raises(ImportError):
            obj(key="value")

    def test_boolean_operations(self):
        """Test boolean behavior of FakeObject."""
        obj = FakeObject("test")

        # Should be falsy
        assert not obj

        # Length should be 0
        assert len(obj) == 0

        # Int conversion should be 0
        assert int(obj) == 0

    def test_string_representation(self):
        """Test string representations of FakeObject."""
        obj = FakeObject("test")

        # String representation
        assert str(obj) == "None"

        # Repr
        assert repr(obj) == "<FakeNone>"

    def test_equality(self):
        """Test equality comparison of FakeObject."""
        obj = FakeObject("test")

        # Should be equal to None
        assert obj == None  # noqa: E711

        # Should not be equal to other values
        assert obj != 0
        assert obj != ""
        assert obj != False  # noqa: E712

    def test_getitem(self):
        """Test item access on FakeObject."""
        obj = FakeObject("test")

        # Access item
        item = obj["key"]
        assert isinstance(item, FakeObject)
        assert item._name == "test[key]"

        # Ensure cached
        assert obj["key"] is item

        # Test with no name
        obj2 = FakeObject()
        assert obj2["key"]._name == "[key]"

    def test_iteration(self):
        """Test iteration over FakeObject."""
        obj = FakeObject("test")

        # Should iterate once with None
        items = list(obj)
        assert items == [None]


class TestFakeModule:
    """Test the FakeModule class."""

    def test_init(self):
        """Test initialization of FakeModule."""
        # Test with non-strict mode
        mod = FakeModule("test_module", strict=False)
        assert mod.__name__ == "test_module"
        assert mod.__is_fake_module__ is True
        assert mod._fake_strict is False

        # Test with strict mode
        mod = FakeModule("test_module", strict=True)
        assert mod._fake_strict is True

    def test_getattr_non_strict(self):
        """Test attribute access in non-strict mode."""
        mod = FakeModule("test_module", strict=False)

        # Should return FakeObject
        attr = mod.some_attr
        assert isinstance(attr, FakeObject)
        assert attr._name == "some_attr"

        # Test nested attribute
        nested = mod.some_attr.nested
        assert isinstance(nested, FakeObject)
        assert nested._name == "some_attr.nested"

    def test_getattr_strict(self):
        """Test attribute access in strict mode."""
        mod = FakeModule("test_module", strict=True)

        # Should return None
        assert mod.some_attr is None

    def test_setattr(self):
        """Test setting attributes on FakeModule."""
        mod = FakeModule("test_module")

        # Set a normal attribute
        mod.attr = 42
        assert mod.attr == 42

        # Set a callable
        def func():
            return "hello"

        mod.func = func
        assert mod.func() == "hello"


class TestMakeFakeModule:
    """Test the make_fake_module function."""

    def setup_method(self):
        """Setup: Remove any existing test modules from sys.modules."""
        # Remove any existing test modules
        keys_to_remove = [key for key in sys.modules.keys()
                         if key.startswith('test_fake_') or key == 'nonexistent_module']
        for key in keys_to_remove:
            if key in sys.modules:
                del sys.modules[key]

    def test_new_module_creation(self):
        """Test creating a new fake module."""
        # Create a module that doesn't exist
        mod = make_fake_module("test_fake_new", inject=True)

        # Check it's a FakeModule
        assert isinstance(mod, FakeModule)
        assert mod.__name__ == "test_fake_new"
        assert mod.__is_fake_module__ is True

        # Check it's in sys.modules
        assert "test_fake_new" in sys.modules
        assert sys.modules["test_fake_new"] is mod

    def test_existing_module_return(self):
        """Test that existing modules are returned as-is."""
        # First create a module
        orig_mod = make_fake_module("test_fake_existing", inject=True)

        # Now try to make it again
        second_mod = make_fake_module("test_fake_existing")

        # Should be the same object
        assert second_mod is orig_mod

    def test_strict_mode(self):
        """Test creating a module in strict mode."""
        mod = make_fake_module("test_fake_strict", strict=True)

        # Attributes should be None
        assert mod.anything is None

    def test_no_inject(self):
        """Test creating a module without injecting it into sys.modules."""
        mod = make_fake_module("nonexistent_module", inject=False)

        # Should not be in sys.modules
        assert "nonexistent_module" not in sys.modules


if __name__ == "__main__":
    pytest.main()