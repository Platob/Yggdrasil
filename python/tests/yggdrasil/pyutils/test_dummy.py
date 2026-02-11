# test_dummy.py
from __future__ import annotations

import types
import pytest
import importlib

from yggdrasil.pyutils.dummy import Dummy  # noqa: F401


def test_from_name_builds_path_strs_and_iterables() -> None:
    d1 = Dummy.from_name("a", "b", to_class=False, try_import=False)
    assert isinstance(d1, Dummy)
    assert d1._module_path == ("a", "b")

    d2 = Dummy.from_name(["a", "b"], to_class=False, try_import=False)
    assert isinstance(d2, Dummy)
    assert d2._module_path == ("a", "b")

    d3 = Dummy.from_name("a", ["b", "c"], to_class=False, try_import=False)
    assert isinstance(d3, Dummy)
    assert d3._module_path == ("a", "b", "c")


def test_from_name_to_class_false_try_import_true_imports_module() -> None:
    m = Dummy.from_name("importlib", to_class=False, try_import=True)
    assert isinstance(m, types.ModuleType)
    assert getattr(m, "__name__", "") == "importlib"


def test_from_name_to_class_false_try_import_true_missing_returns_dummy() -> None:
    d = Dummy.from_name("definitely_not_a_real_module_12345", to_class=False, try_import=True)
    assert isinstance(d, Dummy)
    assert d._module_path == ("definitely_not_a_real_module_12345",)


def test_from_name_to_class_true_try_import_false_always_dummy() -> None:
    d = Dummy.from_name("importlib", "machinery", to_class=True, try_import=False)
    assert isinstance(d, Dummy)
    assert d._module_path == ("importlib", "machinery")


def test_from_name_to_class_true_resolves_attr_on_module() -> None:
    # "importlib.machinery" is a module; "SourceFileLoader" is an attribute
    obj = Dummy.from_name("importlib", "machinery", "SourceFileLoader", to_class=True, try_import=True)
    assert obj is importlib.machinery.SourceFileLoader  # type: ignore[name-defined]


def test_from_name_to_class_true_missing_attr_returns_dummy() -> None:
    d = Dummy.from_name("importlib", "machinery", "DefinitelyNotThere", to_class=True, try_import=True)
    assert isinstance(d, Dummy)
    assert d._module_path == ("importlib", "machinery", "DefinitelyNotThere")


def test_dotted_and_getattr_chain() -> None:
    d = Dummy(("a", "b"))
    assert d.dotted() == "a.b"

    d2 = d.some_attr.other
    assert isinstance(d2, Dummy)
    assert d2._module_path == ("a", "b", "some_attr", "other")
    assert d2.dotted() == "a.b.some_attr.other"


def test_materialize_import_module_success() -> None:
    d = Dummy(("importlib",))
    obj = d.materialize()
    assert isinstance(obj, types.ModuleType)
    assert obj.__name__ == "importlib"


def test_materialize_import_dotted_module_success() -> None:
    d = Dummy(("importlib", "machinery"))
    obj = d.materialize()
    assert isinstance(obj, types.ModuleType)
    assert obj.__name__ == "importlib.machinery"


def test_materialize_fallback_import_root_then_getattr_success() -> None:
    # "importlib.SourceFileLoader" isn't a module, so it should import root and getattr
    d = Dummy(("importlib", "machinery", "SourceFileLoader"))
    obj = d.materialize()
    assert obj is importlib.machinery.SourceFileLoader  # type: ignore[name-defined]


def test_materialize_missing_module_raise_error_true() -> None:
    d = Dummy(("definitely_not_a_real_module_12345",))
    with pytest.raises(ModuleNotFoundError) as ei:
        d.materialize(raise_error=True)
    msg = str(ei.value)
    assert "Missing optional dependency" in msg
    assert "definitely_not_a_real_module_12345" in msg


def test_materialize_missing_module_raise_error_false_returns_dummy() -> None:
    d = Dummy(("definitely_not_a_real_module_12345",))
    out = d.materialize(raise_error=False)
    assert isinstance(out, Dummy)
    assert out._module_path == ("definitely_not_a_real_module_12345",)


def test_materialize_missing_attr_raise_error_true() -> None:
    d = Dummy(("importlib", "machinery", "DefinitelyNotThere"))
    with pytest.raises(ModuleNotFoundError) as ei:
        d.materialize(raise_error=True)
    msg = str(ei.value)
    assert "Missing optional dependency" in msg
    assert "attribute 'DefinitelyNotThere' not found" in msg


def test_materialize_missing_attr_raise_error_false_returns_dummy() -> None:
    d = Dummy(("importlib", "machinery", "DefinitelyNotThere"))
    out = d.materialize(raise_error=False)
    assert isinstance(out, Dummy)
    assert out._module_path == ("importlib", "machinery", "DefinitelyNotThere")


@pytest.mark.parametrize(
    "op",
    [
        lambda d: d(),            # __call__
        lambda d: bool(d),        # __bool__
        lambda d: iter(d),        # __iter__
        lambda d: len(d),         # __len__
        lambda d: ("x" in d),     # __contains__
        lambda d: d["k"],         # __getitem__
        lambda d: d.__setitem__("k", 1),  # __setitem__
        lambda d: d.__delitem__("k"),     # __delitem__
    ],
)
def test_runtime_ops_raise_module_not_found(op) -> None:
    d = Dummy(("missing_pkg", "thing"))
    with pytest.raises(ModuleNotFoundError) as ei:
        op(d)
    msg = str(ei.value)
    assert "Missing optional dependency" in msg
    assert "missing_pkg" in msg


def test_repr() -> None:
    d = Dummy(("a", "b", "c"))
    assert repr(d) == "<Dummy missing 'a.b.c'>"
