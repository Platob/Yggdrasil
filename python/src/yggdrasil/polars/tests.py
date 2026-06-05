"""Unittest base class for Polars tests.

Quick start
-----------
::

    from yggdrasil.testing import PolarsTestCase
    import polars as pl

    class TestMyFilter(PolarsTestCase):
        def test_filter(self):
            df = self.df({"id": [1, 2, 3], "val": ["a", "b", "c"]})
            result = df.filter(pl.col("id") > 1)
            self.assertFrameEqual(result, {"id": [2, 3], "val": ["b", "c"]})

Auto-install
------------
Uses :func:`yggdrasil.environ.runtime_import_module` to load ``polars``.
Set ``auto_install = True`` on the subclass or export
``YGG_TEST_AUTO_INSTALL=1`` to install it automatically; otherwise a
missing ``polars`` skips the class with an install hint.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from yggdrasil.environ import runtime_import_module

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

__all__ = ["PolarsTestCase"]


def _auto_install(class_flag: bool | None) -> bool:
    if class_flag is not None:
        return class_flag
    return os.environ.get("YGG_TEST_AUTO_INSTALL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


class PolarsTestCase(unittest.TestCase):
    """Base class for Polars integration tests.

    Attributes
    ----------
    pl : module
        The imported ``polars`` module. Populated by ``setUpClass``.
    tmp_path : pathlib.Path
        Per-test scratch directory.
    """

    auto_install: ClassVar[bool | None] = None

    pl: ClassVar[Any]  # polars module
    tmp_path: Path

    # --- lifecycle ------------------------------------------------------
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        install = _auto_install(cls.auto_install)
        try:
            cls.pl = runtime_import_module("polars", install=install)
        except ImportError:
            raise unittest.SkipTest(
                "'polars' is not installed. "
                "Install it with: pip install polars  "
                "or: pip install 'ygg[polars]'  "
                "(or set YGG_TEST_AUTO_INSTALL=1 to auto-install)"
            )

    def setUp(self) -> None:
        super().setUp()
        self.tmp_path = Path(tempfile.mkdtemp(prefix="ygg-polars-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)
        super().tearDown()

    # --- convenience constructors --------------------------------------
    def df(self, data: Any, schema: Any = None) -> "pl.DataFrame":
        """Shorthand for ``pl.DataFrame(data, schema=schema)``."""
        return self.pl.DataFrame(data, schema=schema)

    def lazy(self, data: Any, schema: Any = None) -> "pl.LazyFrame":
        """Shorthand for ``pl.LazyFrame(data, schema=schema)``."""
        return self.pl.LazyFrame(data, schema=schema)

    def series(self, name: str, values: Any) -> "pl.Series":
        """Shorthand for ``pl.Series(name, values)``."""
        return self.pl.Series(name, values)

    # --- Arrow interop -------------------------------------------------
    def arrow_to_polars(self, table: "pa.Table") -> "pl.DataFrame":
        """Convert a ``pa.Table`` to a Polars DataFrame (zero-copy)."""
        return self.pl.from_arrow(table)

    def polars_to_arrow(self, df: "pl.DataFrame") -> "pa.Table":
        """Convert a Polars DataFrame to a ``pa.Table`` (zero-copy)."""
        return df.to_arrow()

    # --- I/O helpers ---------------------------------------------------
    def write_parquet(self, df: "pl.DataFrame", path: Path | str) -> Path:
        """Write ``df`` to ``path`` as Parquet and return the path."""
        path = Path(path)
        df.write_parquet(path)
        return path

    def read_parquet(self, path: Path | str) -> "pl.DataFrame":
        """Read a Parquet file as a Polars DataFrame."""
        return self.pl.read_parquet(Path(path))

    # --- assertions ----------------------------------------------------
    def assertFrameEqual(
        self,
        actual: "pl.DataFrame | pl.LazyFrame",
        expected: "pl.DataFrame | pl.LazyFrame | dict[str, Any] | list[dict[str, Any]]",
        *,
        ordered: bool = True,
        check_dtypes: bool = True,
        check_column_order: bool = True,
        **kwargs: Any,
    ) -> None:
        """Assert two Polars frames are equal.

        Parameters
        ----------
        actual : pl.DataFrame | pl.LazyFrame
        expected : pl.DataFrame | pl.LazyFrame | dict | list[dict]
            If a dict or list of dicts is given, it's coerced via
            ``pl.DataFrame``.
        ordered : bool, default True
            If False, both sides are sorted by all columns before compare.
        check_dtypes : bool, default True
        check_column_order : bool, default True
            If False, columns are reordered to match ``expected``.
        kwargs :
            Additional args forwarded to ``polars.testing.assert_frame_equal``.
        """
        pl = self.pl

        if isinstance(actual, pl.LazyFrame):
            actual = actual.collect()
        if isinstance(expected, pl.LazyFrame):
            expected = expected.collect()
        if not isinstance(expected, pl.DataFrame):
            expected = pl.DataFrame(expected)

        left, right = actual, expected

        if not check_column_order:
            left = left.select(right.columns)

        if not ordered:
            cols = right.columns
            left = left.sort(cols)
            right = right.sort(cols)

        from polars.testing import assert_frame_equal
        try:
            assert_frame_equal(left, right, check_dtypes=check_dtypes, **kwargs)
        except AssertionError as exc:
            self.fail(f"DataFrames differ:\n{exc}")

    def assertSeriesEqual(
        self,
        actual: "pl.Series",
        expected: "pl.Series | list[Any]",
        *,
        check_dtype: bool = True,
        **kwargs: Any,
    ) -> None:
        """Assert two Polars Series are equal."""
        pl = self.pl
        if not isinstance(expected, pl.Series):
            expected = pl.Series(actual.name, expected)

        from polars.testing import assert_series_equal
        try:
            assert_series_equal(
                actual, expected, check_dtypes=check_dtype, **kwargs,
            )
        except AssertionError as exc:
            self.fail(f"Series differ:\n{exc}")

    def assertSchemaEqual(
        self,
        actual: "pl.DataFrame | pl.LazyFrame",
        expected_fields: list[tuple[str, Any]],
    ) -> None:
        """Assert a frame has exactly the given ``(name, dtype)`` fields.

        ``dtype`` may be a Polars dtype class or its string name.
        """
        schema = actual.schema
        got = [(name, str(dtype)) for name, dtype in schema.items()]
        want = [(n, str(t)) for n, t in expected_fields]
        self.assertEqual(got, want, "Polars schemas differ")