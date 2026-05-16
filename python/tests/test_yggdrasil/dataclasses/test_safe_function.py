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
import unittest
from typing import Optional

from yggdrasil.dataclasses.safe_function import (
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


if __name__ == "__main__":
    unittest.main()
