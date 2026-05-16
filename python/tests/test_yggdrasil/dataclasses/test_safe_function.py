"""Tests for :mod:`yggdrasil.dataclasses.safe_function`.

PEP 563 (``from __future__ import annotations``) is intentionally
*not* enabled here: the suite exercises annotation resolution and
needs the type references to be live names in the module globals so
``inspect.get_annotations(eval_str=True)`` can resolve them.
Optional engine imports (pyarrow / polars / pandas) are gated with
``try/except`` and the test classes that use them carry
``@unittest.skipUnless`` so the file still loads on a base install.
"""
import datetime as dt
import inspect
import unittest
from typing import Optional

from yggdrasil.dataclasses.safe_function import (
    _annotation_to_str,
    _canonical_module_path,
    _expand_alias,
    _resolve_str_annotation,
    check_function_args,
    checkargs,
    describe_signature,
    format_signature,
)


# Optional engine imports — module-level so eval_str can resolve
# ``pa.Table`` / ``pl.DataFrame`` / ``pd.DataFrame`` against the test
# module globals at annotation-resolution time.
try:
    import pyarrow as pa  # type: ignore[import-not-found]
except ImportError:
    pa = None  # type: ignore[assignment]

try:
    import polars as pl  # type: ignore[import-not-found]
except ImportError:
    pl = None  # type: ignore[assignment]

try:
    import pandas as pd  # type: ignore[import-not-found]
except ImportError:
    pd = None  # type: ignore[assignment]


def _have(name: str) -> bool:
    return globals().get(name) is not None


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


# ---------------------------------------------------------------------------
# Tricky type hints — primitives, Optional, datetime, list/set, identity
# ---------------------------------------------------------------------------

class TestSafeFunctionPrimitives(unittest.TestCase):
    """Built-in scalar coercion targets routed through yggdrasil.convert."""

    def test_int_from_string(self):
        @checkargs
        def f(n: int) -> int:
            return n + 1
        self.assertEqual(f("41"), 42)
        # Identity path: an int passed in stays an int.
        self.assertEqual(f(41), 42)

    def test_float_from_string(self):
        @checkargs
        def f(x: float) -> float:
            return x * 2
        self.assertEqual(f("3.14"), 6.28)

    def test_bool_from_truthy_string_forms(self):
        @checkargs
        def f(flag: bool) -> bool:
            return flag
        # Common truthy/falsy strings parse via yggdrasil.convert.
        for truthy in ("true", "yes", "1", "True"):
            with self.subTest(truthy=truthy):
                self.assertTrue(f(truthy))
        for falsy in ("false", "no", "0", "False"):
            with self.subTest(falsy=falsy):
                self.assertFalse(f(falsy))

    def test_date_from_iso_string(self):
        @checkargs
        def f(d: dt.date) -> dt.date:
            return d
        self.assertEqual(f("2024-01-15"), dt.date(2024, 1, 15))

    def test_datetime_from_iso_string_is_utc_normalized(self):
        @checkargs
        def f(ts: dt.datetime) -> dt.datetime:
            return ts
        out = f("2024-01-15T10:00:00")
        self.assertEqual(out, dt.datetime(2024, 1, 15, 10, 0, tzinfo=dt.timezone.utc))

    def test_datetime_from_unix_timestamp(self):
        @checkargs
        def f(ts: dt.datetime) -> dt.datetime:
            return ts
        # Unix epoch seconds round-trip cleanly via convert.
        out = f(1700000000)
        self.assertEqual(out.tzinfo, dt.timezone.utc)
        self.assertEqual(out, dt.datetime(2023, 11, 14, 22, 13, 20, tzinfo=dt.timezone.utc))


class TestSafeFunctionContainers(unittest.TestCase):

    def test_list_int_identity(self):
        @checkargs
        def f(xs: list[int]) -> int:
            return sum(xs)
        self.assertEqual(f([1, 2, 3]), 6)

    def test_set_int_from_list(self):
        @checkargs
        def f(xs: set[int]) -> int:
            return len(xs)
        self.assertEqual(f([1, 2, 2, 3]), 3)


class TestSafeFunctionOptional(unittest.TestCase):
    """Optional[T] and None propagation."""

    def test_optional_int_accepts_none(self):
        @checkargs
        def f(n: Optional[int]) -> Optional[int]:
            return n
        self.assertIsNone(f(None))

    def test_optional_int_coerces_string(self):
        @checkargs
        def f(n: Optional[int]) -> Optional[int]:
            return n
        self.assertEqual(f("5"), 5)
        self.assertIsInstance(f("5"), int)


# ---------------------------------------------------------------------------
# Tricky type hints — dataframe engines (pa.Table / pl.DataFrame / pd.DataFrame)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_have("pa"), "pyarrow not installed")
class TestSafeFunctionArrow(unittest.TestCase):

    def test_pa_table_identity(self):
        @checkargs
        def f(t: pa.Table) -> int:
            return t.num_rows
        tbl = pa.table({"x": [1, 2, 3]})
        self.assertEqual(f(tbl), 3)

    def test_pa_record_batch_coerced_to_table(self):
        @checkargs
        def f(t: pa.Table) -> int:
            return t.num_rows
        batch = pa.record_batch([pa.array([1, 2, 3])], names=["x"])
        # convert lifts a RecordBatch to a single-batch Table.
        self.assertEqual(f(batch), 3)

    def test_pa_table_coerced_to_record_batch(self):
        @checkargs
        def f(b: pa.RecordBatch) -> int:
            return b.num_rows
        tbl = pa.table({"x": [1, 2, 3, 4]})
        self.assertEqual(f(tbl), 4)


@unittest.skipUnless(_have("pa") and _have("pl"), "pyarrow or polars not installed")
class TestSafeFunctionArrowPolars(unittest.TestCase):

    def test_polars_df_coerced_to_pa_table(self):
        @checkargs
        def consume(t: pa.Table) -> int:
            return t.num_rows
        out = consume(pl.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual(out, 3)

    def test_pa_table_coerced_to_polars_df(self):
        @checkargs
        def consume(df: pl.DataFrame) -> tuple:
            return df.shape
        out = consume(pa.table({"x": [1, 2, 3]}))
        self.assertEqual(out, (3, 1))


@unittest.skipUnless(_have("pa") and _have("pd"), "pyarrow or pandas not installed")
class TestSafeFunctionArrowPandas(unittest.TestCase):

    def test_pandas_df_coerced_to_pa_table(self):
        @checkargs
        def consume(t: pa.Table) -> int:
            return t.num_rows
        out = consume(pd.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual(out, 3)

    def test_pa_table_coerced_to_pandas_df(self):
        @checkargs
        def consume(df: pd.DataFrame) -> tuple:
            return df.shape
        out = consume(pa.table({"x": [1, 2, 3]}))
        self.assertEqual(out, (3, 1))


@unittest.skipUnless(_have("pl") and _have("pd"), "polars or pandas not installed")
class TestSafeFunctionPolarsPandas(unittest.TestCase):

    def test_polars_df_coerced_to_pandas_df(self):
        @checkargs
        def consume(df: pd.DataFrame) -> tuple:
            return df.shape
        out = consume(pl.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual(out, (3, 1))

    def test_pandas_df_coerced_to_polars_df(self):
        @checkargs
        def consume(df: pl.DataFrame) -> tuple:
            return df.shape
        out = consume(pd.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual(out, (3, 1))


@unittest.skipUnless(_have("pa") and _have("pl"), "pyarrow + polars required")
class TestCheckFunctionArgsAcrossDataframes(unittest.TestCase):
    """Round-trip check_function_args against dataframe-shaped signatures."""

    def test_check_function_args_coerces_dataframe_kwargs(self):
        def consume(t: pa.Table, n: int) -> None: ...

        args, kwargs = check_function_args(
            consume, (), {"t": pl.DataFrame({"x": [1, 2]}), "n": "7"},
        )
        self.assertIsInstance(kwargs["t"], pa.Table)
        self.assertEqual(kwargs["t"].num_rows, 2)
        self.assertEqual(kwargs["n"], 7)
        self.assertIsInstance(kwargs["n"], int)

    def test_check_function_args_coerces_dataframe_positional(self):
        def consume(t: pa.Table, n: int) -> None: ...

        args, kwargs = check_function_args(
            consume, (pl.DataFrame({"x": [1, 2]}), "7"), {},
        )
        self.assertIsInstance(args[0], pa.Table)
        self.assertEqual(args[1], 7)


# ---------------------------------------------------------------------------
# Mixed-engine signatures — every parameter coerced independently
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    _have("pa") and _have("pl") and _have("pd"),
    "pyarrow / polars / pandas all required",
)
class TestSafeFunctionMixedEngines(unittest.TestCase):

    def test_each_engine_parameter_coerced_independently(self):
        @checkargs
        def heavy(
            arrow: pa.Table,
            polars_df: pl.DataFrame,
            pandas_df: pd.DataFrame,
            cutoff: dt.date,
            limit: int,
        ) -> dict:
            return {
                "arrow_rows": arrow.num_rows,
                "polars_shape": polars_df.shape,
                "pandas_shape": pandas_df.shape,
                "cutoff": cutoff,
                "limit": limit,
            }

        # Cross every parameter with a non-matching input type so the
        # decorator actually has to convert each one.
        out = heavy(
            arrow=pl.DataFrame({"x": [1, 2, 3]}),         # pl → pa
            polars_df=pa.table({"x": [4, 5]}),            # pa → pl
            pandas_df=pl.DataFrame({"x": [6, 7, 8, 9]}),  # pl → pd
            cutoff="2024-01-15",                          # str → date
            limit="50",                                   # str → int
        )

        self.assertEqual(out["arrow_rows"], 3)
        self.assertEqual(out["polars_shape"], (2, 1))
        self.assertEqual(out["pandas_shape"], (4, 1))
        self.assertEqual(out["cutoff"], dt.date(2024, 1, 15))
        self.assertEqual(out["limit"], 50)
        self.assertIsInstance(out["limit"], int)


# ---------------------------------------------------------------------------
# Forgiving annotation resolution — fast aliases + graceful fallback
# ---------------------------------------------------------------------------

class TestExpandAlias(unittest.TestCase):
    """Short-prefix aliases resolve to fully-qualified module paths."""

    def test_pyarrow_aliases(self):
        self.assertEqual(_expand_alias("pa.Table"), "pyarrow.Table")
        self.assertEqual(_expand_alias("pa.RecordBatch"), "pyarrow.RecordBatch")

    def test_polars_aliases(self):
        self.assertEqual(_expand_alias("pl.DataFrame"), "polars.DataFrame")
        self.assertEqual(_expand_alias("pl.LazyFrame"), "polars.LazyFrame")

    def test_pandas_alias(self):
        self.assertEqual(_expand_alias("pd.DataFrame"), "pandas.DataFrame")
        self.assertEqual(_expand_alias("pd.Series"), "pandas.Series")

    def test_numpy_alias(self):
        self.assertEqual(_expand_alias("np.ndarray"), "numpy.ndarray")

    def test_pyspark_alias(self):
        self.assertEqual(
            _expand_alias("ps.sql.DataFrame"), "pyspark.sql.DataFrame",
        )

    def test_unknown_prefix_passes_through(self):
        # Non-aliased names are returned untouched.
        self.assertEqual(_expand_alias("int"), "int")
        self.assertEqual(_expand_alias("my.custom.Type"), "my.custom.Type")


class TestResolveStrAnnotation(unittest.TestCase):
    """:func:`_resolve_str_annotation` walks eval → typing → alias → import."""

    @unittest.skipUnless(_have("pa"), "pyarrow not installed")
    def test_resolves_fast_alias_without_caller_globals(self):
        # No ``func_globals`` passed — has to fall through to the
        # alias-prefix + importlib path.
        import pyarrow as pa_real
        self.assertIs(_resolve_str_annotation("pa.Table"), pa_real.Table)

    @unittest.skipUnless(_have("pl"), "polars not installed")
    def test_resolves_polars_short_form(self):
        import polars as pl_real
        self.assertIs(_resolve_str_annotation("pl.DataFrame"), pl_real.DataFrame)

    @unittest.skipUnless(_have("pd"), "pandas not installed")
    def test_resolves_pandas_short_form(self):
        import pandas as pd_real
        self.assertIs(_resolve_str_annotation("pd.DataFrame"), pd_real.DataFrame)

    def test_resolves_typing_generic_without_caller_globals(self):
        from typing import Optional
        # ``Optional[int]`` would NameError in an empty globals, but
        # the typing fallback handles it.
        self.assertEqual(_resolve_str_annotation("Optional[int]"), Optional[int])

    def test_unresolvable_returns_original_string(self):
        # No fast alias, no module, no typing match — return the input.
        out = _resolve_str_annotation("ThisTypeReallyDoesNotExist")
        self.assertEqual(out, "ThisTypeReallyDoesNotExist")
        self.assertIsInstance(out, str)


@unittest.skipUnless(_have("pa"), "pyarrow not installed")
class TestCheckargsFallbackPaths(unittest.TestCase):
    """@checkargs is forgiving: unresolvable annotations + convert failures pass through."""

    def test_short_alias_resolves_when_caller_missing_import(self):
        # Build a function whose ``__globals__`` do NOT have ``pa`` —
        # eval_str will NameError on ``pa.Table``, and our fallback
        # has to kick in via the alias prefix + importlib lookup.
        ns: dict = {"__builtins__": __builtins__}
        exec(
            "def consume(t: 'pa.Table') -> int:\n"
            "    return t.num_rows\n",
            ns,
        )
        consume = ns["consume"]
        # Sanity: ``pa`` really isn't in the function's globals.
        self.assertNotIn("pa", consume.__globals__)

        import pyarrow as pa_real
        tbl = pa_real.table({"x": [1, 2, 3]})
        self.assertEqual(checkargs(consume)(tbl), 3)

    def test_unresolvable_annotation_passes_value_through(self):
        # ``ThisTypeDoesNotExist`` isn't a real type and isn't aliased.
        # The decorator must not raise — pass the value through as-is.
        @checkargs
        def f(x: "ThisTypeDoesNotExist") -> object:  # noqa: F821 — intentional
            return x

        self.assertEqual(f("hello"), "hello")
        self.assertEqual(f(42), 42)

    def test_convert_failure_passes_value_through(self):
        # ``Union[int, str]`` has no registered converter; convert
        # raises TypeError and our wrapper falls through cleanly.
        from typing import Union

        @checkargs
        def f(x: Union[int, str]) -> object:
            return x

        self.assertEqual(f("hello"), "hello")
        self.assertEqual(f(42), 42)


# ---------------------------------------------------------------------------
# Real-world use cases — enums, dataclasses, nested generics, methods, async
# ---------------------------------------------------------------------------

from enum import Enum
from dataclasses import dataclass


class _Color(Enum):
    RED = "red"
    GREEN = "green"


@dataclass
class _Point:
    x: int
    y: int


class TestSafeFunctionEnumsAndDataclasses(unittest.TestCase):

    def test_enum_from_value_string(self):
        @checkargs
        def pick(c: _Color) -> str:
            return c.name
        # yggdrasil.convert maps the value string to the enum member.
        self.assertEqual(pick("red"), "RED")

    def test_enum_from_name_string(self):
        @checkargs
        def pick(c: _Color) -> str:
            return c.name
        # And the member name works too.
        self.assertEqual(pick("RED"), "RED")

    def test_dataclass_from_dict(self):
        @checkargs
        def origin_distance(p: _Point) -> int:
            return p.x + p.y
        # convert builds the dataclass from a dict, coercing each field.
        self.assertEqual(origin_distance({"x": "3", "y": "4"}), 7)

    def test_dataclass_identity_passes_through(self):
        @checkargs
        def consume(p: _Point) -> _Point:
            return p
        p = _Point(1, 2)
        self.assertIs(consume(p), p)


class TestSafeFunctionNestedGenerics(unittest.TestCase):

    def test_list_of_list(self):
        from typing import List

        @checkargs
        def f(rows: List[List[int]]) -> int:
            return sum(v for row in rows for v in row)
        # Outer list, inner list — both preserved; identity for already-int values.
        self.assertEqual(f([[1, 2], [3, 4]]), 10)

    def test_list_of_dict_with_value_coercion(self):
        from typing import List, Dict

        @checkargs
        def f(rows: List[Dict[str, int]]) -> int:
            return sum(r["v"] for r in rows)
        # Values arrive as strings, get coerced to int via the nested generic.
        self.assertEqual(f([{"v": "1"}, {"v": "2"}]), 3)

    def test_tuple_with_mixed_element_types(self):
        from typing import Tuple

        @checkargs
        def f(row: Tuple[int, str, float]) -> tuple:
            return (type(row[0]).__name__, row[1], type(row[2]).__name__)
        self.assertEqual(f((1, "a", 2.0)), ("int", "a", "float"))


class TestSafeFunctionDefaultsAndKindedParams(unittest.TestCase):

    def test_defaults_are_not_coerced_when_caller_omits_them(self):
        ledger: list = []

        @checkargs
        def f(name: str, count: int = 5) -> None:
            ledger.append((name, count, type(count).__name__))

        # ``count`` defaults to int 5 — the wrapper must not run the
        # default through convert (would still be int → int but the
        # principle stands: only caller-supplied args get coerced).
        f("alice")
        self.assertEqual(ledger[-1], ("alice", 5, "int"))

    def test_keyword_only_param_is_coerced(self):
        @checkargs
        def f(*, n: int) -> int:
            return n
        self.assertEqual(f(n="42"), 42)

    def test_positional_only_param_is_coerced(self):
        @checkargs
        def f(n: int, /) -> int:
            return n
        self.assertEqual(f("42"), 42)

    def test_var_positional_per_element_coercion(self):
        @checkargs
        def f(*nums: int) -> int:
            return sum(nums)
        self.assertEqual(f("1", "2", "3"), 6)

    def test_var_keyword_per_element_coercion(self):
        @checkargs
        def f(**flags: bool) -> int:
            return sum(1 for v in flags.values() if v)
        # Each kwarg value coerces to bool individually.
        self.assertEqual(f(a="true", b="false", c="yes"), 2)


class TestSafeFunctionMethods(unittest.TestCase):
    """``@checkargs`` on bound / static / class methods."""

    def test_instance_method_self_passes_through(self):
        class Thing:
            @checkargs
            def double(self, n: int) -> int:
                return n * 2

        # ``self`` has no annotation — it passes through; ``n`` coerces.
        self.assertEqual(Thing().double("21"), 42)

    def test_classmethod(self):
        class Thing:
            @classmethod
            @checkargs
            def add(cls, a: int, b: int) -> int:
                return a + b

        self.assertEqual(Thing.add("2", "3"), 5)

    def test_staticmethod(self):
        class Thing:
            @staticmethod
            @checkargs
            def mul(a: int, b: int) -> int:
                return a * b

        self.assertEqual(Thing.mul("4", "5"), 20)


class TestSafeFunctionAsync(unittest.TestCase):

    def test_async_function_is_wrapped_in_async_wrapper(self):
        import asyncio

        @checkargs
        async def add(a: int, b: int) -> int:
            return a + b

        self.assertTrue(inspect.iscoroutinefunction(add))
        result = asyncio.run(add("3", "4"))
        self.assertEqual(result, 7)

    def test_async_function_preserves_signature(self):
        @checkargs
        async def f(name: str, count: int = 1) -> str:
            """Docstring stays."""
            return name * count

        self.assertEqual(f.__name__, "f")
        self.assertEqual(f.__doc__, "Docstring stays.")
        sig = inspect.signature(f)
        self.assertEqual(list(sig.parameters), ["name", "count"])


class TestSafeFunctionMisc(unittest.TestCase):

    def test_return_value_passes_through_unchanged(self):
        @checkargs
        def f(n: int) -> object:
            return {"n": n, "tag": object()}
        out = f("10")
        self.assertEqual(out["n"], 10)
        self.assertIsInstance(out["tag"], object)

    def test_double_decoration_is_idempotent(self):
        @checkargs
        @checkargs
        def f(n: int) -> int:
            return n + 1

        # No double coercion (would matter for types whose convert isn't
        # idempotent), and one level of __wrapped__ peels back to the
        # original function.
        self.assertEqual(f("4"), 5)
        # ``f`` is the second wrapper; ``f.__wrapped__`` is the
        # original underlying function (the inner @checkargs got
        # unwrapped during the second application).
        self.assertEqual(f.__wrapped__.__name__, "f")

    def test_functools_partial_target_works(self):
        import functools

        def f(name: str, count: int) -> str:
            return name * count

        wrapped = checkargs(f)
        bound = functools.partial(wrapped, count="3")
        self.assertEqual(bound(name="ab"), "ababab")

    def test_lambda_can_be_wrapped(self):
        # No annotations on a bare lambda; values pass through.
        wrapped = checkargs(lambda x, y: x + y)
        self.assertEqual(wrapped("a", "b"), "ab")
        self.assertEqual(wrapped(1, 2), 3)


class TestCanonicalModulePath(unittest.TestCase):
    """Non-builtin types render with their shortest canonical import path."""

    def test_builtin_returns_empty(self):
        self.assertEqual(_canonical_module_path(int), "")
        self.assertEqual(_canonical_module_path(str), "")
        self.assertEqual(_canonical_module_path(list), "")

    def test_stdlib_module(self):
        self.assertEqual(_canonical_module_path(dt.date), "datetime")
        self.assertEqual(_canonical_module_path(dt.datetime), "datetime")

    def test_pathlib_path(self):
        from pathlib import Path
        self.assertEqual(_canonical_module_path(Path), "pathlib")

    @unittest.skipUnless(_have("pa"), "pyarrow not installed")
    def test_pyarrow_table_collapses_to_top_level(self):
        # Real __module__ is ``pyarrow.lib``; we want the top-level
        # ``pyarrow.Table`` users actually type.
        self.assertEqual(_canonical_module_path(pa.Table), "pyarrow")
        self.assertEqual(_canonical_module_path(pa.RecordBatch), "pyarrow")
        self.assertEqual(_canonical_module_path(pa.Array), "pyarrow")

    @unittest.skipUnless(_have("pl"), "polars not installed")
    def test_polars_dataframe_collapses_to_top_level(self):
        # Real __module__ is ``polars.dataframe.frame``.
        self.assertEqual(_canonical_module_path(pl.DataFrame), "polars")
        self.assertEqual(_canonical_module_path(pl.LazyFrame), "polars")
        self.assertEqual(_canonical_module_path(pl.Series), "polars")

    @unittest.skipUnless(_have("pd"), "pandas not installed")
    def test_pandas_dataframe_collapses_to_top_level(self):
        # Real __module__ is ``pandas.core.frame``.
        self.assertEqual(_canonical_module_path(pd.DataFrame), "pandas")
        self.assertEqual(_canonical_module_path(pd.Series), "pandas")


class TestAnnotationToStr(unittest.TestCase):
    """:func:`_annotation_to_str` renders builtins bare, non-builtins fully-qualified."""

    def test_builtin_types_render_bare_name(self):
        self.assertEqual(_annotation_to_str(int), "int")
        self.assertEqual(_annotation_to_str(str), "str")
        self.assertEqual(_annotation_to_str(bool), "bool")
        self.assertEqual(_annotation_to_str(list), "list")

    def test_stdlib_types_render_with_module(self):
        self.assertEqual(_annotation_to_str(dt.date), "datetime.date")
        self.assertEqual(_annotation_to_str(dt.datetime), "datetime.datetime")

    @unittest.skipUnless(_have("pa") and _have("pl") and _have("pd"),
                         "pa / pl / pd all required")
    def test_dataframe_types_render_with_top_level_module(self):
        self.assertEqual(_annotation_to_str(pa.Table), "pyarrow.Table")
        self.assertEqual(_annotation_to_str(pl.DataFrame), "polars.DataFrame")
        self.assertEqual(_annotation_to_str(pd.DataFrame), "pandas.DataFrame")

    def test_typing_generic_round_trips_via_repr(self):
        from typing import Optional, List
        self.assertEqual(_annotation_to_str(Optional[int]), "typing.Optional[int]")
        self.assertEqual(_annotation_to_str(List[int]), "typing.List[int]")

    def test_none_and_empty_become_none(self):
        self.assertIsNone(_annotation_to_str(None))
        self.assertIsNone(_annotation_to_str(inspect.Parameter.empty))

    def test_string_annotation_passes_through(self):
        # PEP 563 strings the resolver couldn't crack stay verbatim.
        self.assertEqual(_annotation_to_str("Unresolved"), "Unresolved")


@unittest.skipUnless(_have("pa") and _have("pl") and _have("pd"),
                     "pa / pl / pd all required")
class TestDescribeSignatureCanonicalPaths(unittest.TestCase):
    """End-to-end: signature metadata for a mixed-engine function carries full paths."""

    def test_format_signature_uses_canonical_paths(self):
        def f(
            t: pa.Table,
            df: pl.DataFrame,
            pdf: pd.DataFrame,
            cutoff: dt.date,
            n: int,
        ) -> pa.Table:
            return t

        # qualname carries the enclosing test scope; assert on the
        # parameter / return shape rather than the leading qualname.
        rendered = format_signature(describe_signature(f))
        self.assertTrue(
            rendered.endswith(
                "(t: pyarrow.Table, df: polars.DataFrame, pdf: pandas.DataFrame, "
                "cutoff: datetime.date, n: int) -> pyarrow.Table"
            ),
            rendered,
        )

    def test_describe_signature_metadata_carries_canonical_paths(self):
        def f(t: pa.Table, n: int) -> pl.DataFrame:
            return pl.from_arrow(t)
        meta = describe_signature(f)
        params = {p["name"]: p for p in meta["parameters"]}
        self.assertEqual(params["t"]["annotation"], "pyarrow.Table")
        self.assertEqual(params["n"]["annotation"], "int")
        self.assertEqual(meta["return"], "polars.DataFrame")


if __name__ == "__main__":
    unittest.main()
