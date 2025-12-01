# tests/test_embedded_function.py

import json
import math
import os
import types

import pytest
from yggdrasil.ser import (
    EmbeddedFunction,
    DependencyInfo,
    DependencyCheckResult,
)

MATH_MOD = math
JSON_MOD = json


def top_level_add(x, y):
    return x + y


def uses_globals_and_modules(x: float) -> float:
    # use global module so dependency inference can see it
    return MATH_MOD.sqrt(x)


def uses_multiple_modules(x: int) -> str:
    v = MATH_MOD.sqrt(x)
    return JSON_MOD.dumps({"value": v})


def test_from_callable_rejects_bound_method():
    class Foo:
        def bar(self):
            return 1

    f = Foo().bar
    with pytest.raises(ValueError):
        EmbeddedFunction.from_callable(f)


def test_from_callable_supports_top_level_function():
    emb = EmbeddedFunction.from_callable(top_level_add)

    assert emb.name == "top_level_add"
    assert "def top_level_add" in emb.source
    assert callable(emb)


def test_from_callable_supports_local_function_without_nonlocals():
    def outer():
        def inner(x):
            return x + 1
        return inner

    inner_fn = outer()
    emb = EmbeddedFunction.from_callable(inner_fn)

    # Should have extracted a valid def for inner
    assert "def inner" in emb.source
    assert emb(10) == 11


def test_from_callable_rejects_local_function_with_nonlocals():
    def make_multiplier(factor):
        def inner(x):
            return x * factor  # captures nonlocal
        return inner

    f = make_multiplier(3)
    with pytest.raises(ValueError):
        EmbeddedFunction.from_callable(f)


def test_call_executes_correctly_and_caches_compiled_function():
    emb = EmbeddedFunction.from_callable(top_level_add)

    result1 = emb(2, 3)
    assert result1 == 5

    compiled_before = emb._compiled  # type: ignore[attr-defined]
    assert isinstance(compiled_before, types.FunctionType)

    result2 = emb(10, -4)
    assert result2 == 6
    assert emb._compiled is compiled_before


def test_list_dependencies_single_module_shape_and_content():
    emb = EmbeddedFunction.from_callable(uses_globals_and_modules)

    deps = emb.list_dependencies()
    assert isinstance(deps, list)
    assert all(isinstance(d, DependencyInfo) for d in deps)

    roots = {d.root_module for d in deps}
    subs = {d.submodule for d in deps}

    # math should show up as both root and submodule
    assert "math" in roots
    assert "math" in subs


def test_list_dependencies_multiple_modules_shape_and_content():
    emb = EmbeddedFunction.from_callable(uses_multiple_modules)

    deps = emb.list_dependencies()
    roots = {d.root_module for d in deps}
    subs = {d.submodule for d in deps}

    # we expect at least math and json to show up
    assert "math" in roots
    assert "json" in roots
    assert "math" in subs
    assert "json" in subs


def test_list_dependencies_paths_are_directories_or_none():
    emb = EmbeddedFunction.from_callable(uses_multiple_modules)

    deps = emb.list_dependencies()
    for d in deps:
        if d.root_path is not None:
            assert os.path.isdir(d.root_path)


def test_check_dependencies_reports_importable_using_submodule():
    emb = EmbeddedFunction.from_callable(uses_multiple_modules)

    status = emb.check_dependencies()

    # keys are submodule values
    assert "math" in status
    assert "json" in status

    for sub, info in status.items():
        assert isinstance(info, DependencyCheckResult)
        assert info.root_module in {"math", "json", "ser"}
        assert info.importable is True
        assert info.error is None


def test_build_creates_independent_function_object():
    emb = EmbeddedFunction.from_callable(top_level_add)

    fn1 = emb.build()
    fn2 = emb.build()

    assert fn1(1, 2) == 3
    assert fn2(10, -1) == 9
    assert fn1 is not fn2


def test_call_matches_build():
    emb = EmbeddedFunction.from_callable(uses_globals_and_modules)

    via_call = emb(9.0)
    via_build = emb.build()(9.0)

    assert via_call == via_build


def test_package_root_is_captured_for_module_defined_function():
    emb = EmbeddedFunction.from_callable(top_level_add)

    # For functions defined in this test module, we expect some package root dir
    assert emb.package_root is not None
    assert os.path.isdir(emb.package_root)

    this_file = os.path.abspath(__file__)
    common = os.path.commonpath([this_file, os.path.abspath(emb.package_root)])
    # package_root should be an ancestor of this test file
    assert common == os.path.abspath(emb.package_root)
