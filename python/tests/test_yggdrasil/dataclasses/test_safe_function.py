"""Tests for :mod:`yggdrasil.dataclasses.safe_function`."""
from __future__ import annotations

import unittest

from yggdrasil.dataclasses.safe_function import (
    check_function_args,
    checkargs,
    describe_signature,
    format_signature,
)


def _annotated(name: str = "alice", count: int = 3) -> str:
    """Greet someone N times."""
    return f"hi {name}" * count


def _unannotated(x, y=7):
    return (x, y)


def _with_varargs(prefix: str, *nums: int, **opts: bool) -> tuple:
    return (prefix, nums, opts)


class TestDescribeSignature(unittest.TestCase):

    def test_captures_annotations_and_defaults(self):
        meta = describe_signature(_annotated)

        self.assertEqual(meta["qualname"], "_annotated")
        self.assertEqual(meta["return"], "str")
        params = {p["name"]: p for p in meta["parameters"]}
        self.assertEqual(params["name"]["annotation"], "str")
        self.assertEqual(params["name"]["default"], "'alice'")
        self.assertEqual(params["count"]["annotation"], "int")
        self.assertEqual(params["count"]["default"], "3")
        self.assertEqual(params["count"]["kind"], "POSITIONAL_OR_KEYWORD")

    def test_omits_missing_annotations_and_defaults(self):
        meta = describe_signature(_unannotated)
        params = {p["name"]: p for p in meta["parameters"]}

        self.assertNotIn("annotation", params["x"])
        self.assertNotIn("default", params["x"])
        self.assertNotIn("annotation", params["y"])
        self.assertEqual(params["y"]["default"], "7")
        self.assertIsNone(meta["return"])

    def test_handles_var_positional_and_var_keyword(self):
        meta = describe_signature(_with_varargs)
        params = {p["name"]: p for p in meta["parameters"]}

        self.assertEqual(params["nums"]["kind"], "VAR_POSITIONAL")
        self.assertEqual(params["nums"]["annotation"], "int")
        self.assertEqual(params["opts"]["kind"], "VAR_KEYWORD")
        self.assertEqual(params["opts"]["annotation"], "bool")


class TestFormatSignature(unittest.TestCase):

    def test_renders_full_signature(self):
        self.assertEqual(
            format_signature(describe_signature(_annotated)),
            "_annotated(name: str = 'alice', count: int = 3) -> str",
        )

    def test_unannotated_signature(self):
        self.assertEqual(
            format_signature(describe_signature(_unannotated)),
            "_unannotated(x, y = 7)",
        )


class TestCheckFunctionArgs(unittest.TestCase):

    def test_coerces_annotated_kwargs(self):
        args, kwargs = check_function_args(
            _annotated, (), {"name": "bob", "count": "9"},
        )
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"name": "bob", "count": 9})
        self.assertIsInstance(kwargs["count"], int)

    def test_coerces_annotated_positional_args(self):
        # ``name`` is str, ``count`` is int — positional inputs coerce too.
        args, kwargs = check_function_args(_annotated, ("bob", "9"), {})
        self.assertEqual(args, ("bob", 9))
        self.assertEqual(kwargs, {})

    def test_skips_unannotated_parameters(self):
        args, kwargs = check_function_args(
            _unannotated, ("raw",), {"y": "9"},
        )
        self.assertEqual(args, ("raw",))
        self.assertEqual(kwargs, {"y": "9"})

    def test_empty_input_short_circuits(self):
        # Empty input must not import yggdrasil — it short-circuits.
        self.assertEqual(check_function_args(_annotated, (), {}), ((), {}))

    def test_var_positional_annotation_applies_per_element(self):
        args, kwargs = check_function_args(
            _with_varargs, ("p", "1", "2", "3"), {"flag": "true"},
        )
        # prefix stays str; *nums coerce element-wise to int;
        # **opts coerce element-wise to bool.
        self.assertEqual(args, ("p", 1, 2, 3))
        self.assertEqual(kwargs, {"flag": True})

    def test_excess_positional_passes_through(self):
        # No matching parameter, no *args catch — pass through so the
        # natural TypeError fires on call rather than being swallowed.
        args, kwargs = check_function_args(_annotated, ("bob", "9", "extra"), {})
        self.assertEqual(args, ("bob", 9, "extra"))

    def test_unknown_keyword_with_no_var_keyword_passes_through(self):
        args, kwargs = check_function_args(_annotated, (), {"bogus": "x"})
        self.assertEqual(kwargs, {"bogus": "x"})


class TestCheckargsDecorator(unittest.TestCase):

    def test_decorator_coerces_each_call(self):
        @checkargs
        def add(a: int, b: int) -> int:
            return a + b

        self.assertEqual(add("2", "3"), 5)
        self.assertEqual(add(a="10", b="4"), 14)

    def test_decorator_preserves_signature_via_functools_wraps(self):
        import inspect

        @checkargs
        def add(a: int, b: int = 5) -> int:
            """Docstring stays."""
            return a + b

        self.assertEqual(add.__name__, "add")
        self.assertEqual(add.__doc__, "Docstring stays.")
        # __wrapped__ exposes the underlying function (functools.wraps).
        self.assertTrue(hasattr(add, "__wrapped__"))
        sig = inspect.signature(add)
        self.assertEqual(list(sig.parameters), ["a", "b"])

    def test_decorator_skips_unannotated_parameters(self):
        @checkargs
        def passthrough(x, y):
            return (x, y)

        # No annotations → no coercion; strings stay strings.
        self.assertEqual(passthrough("1", "2"), ("1", "2"))


if __name__ == "__main__":
    unittest.main()
