"""Unittest base class for pandas tests.

Quick start
-----------
::

    from yggdrasil.testing import PandasTestCase

    class TestMyTransform(PandasTestCase):
        def test_groupby_sum(self):
            df = self.df({"k": ["a", "a", "b"], "v": [1, 2, 3]})
            result = df.groupby("k", as_index=False)["v"].sum()
            self.assertFrameEqual(result, {"k": ["a", "b"], "v": [3, 3]})

Auto-install
------------
Uses :func:`yggdrasil.environ.runtime_import_module` to load ``pandas``.
Set ``auto_install = True`` on the subclass or export
``YGG_TEST_AUTO_INSTALL=1`` to install it automatically; otherwise a
missing ``pandas`` skips the class with an install hint.
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
    import pandas as pd
    import pyarrow as pa

__all__ = ["PandasTestCase"]


def _auto_install(class_flag: bool | None) -> bool:
    if class_flag is not None:
        return class_flag
    return os.environ.get("YGG_TEST_AUTO_INSTALL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


class PandasTestCase(unittest.TestCase):
    """Base class for pandas integration tests.

    Attributes
    ----------
    pd : module
        The imported ``pandas`` module. Populated by ``setUpClass``.
    tmp_path : pathlib.Path
        Per-test scratch directory.
    """

    auto_install: ClassVar[bool | None] = None

    pd: ClassVar[Any]  # pandas module
    tmp_path: Path

    # --- lifecycle ------------------------------------------------------
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        install = _auto_install(cls.auto_install)
        try:
            cls.pd = runtime_import_module("pandas", install=install)
        except ImportError:
            raise unittest.SkipTest(
                "'pandas' is not installed. "
                "Install it with: pip install pandas  "
                "or: pip install 'ygg[pandas]'  "
                "(or set YGG_TEST_AUTO_INSTALL=1 to auto-install)"
            )

    def setUp(self) -> None:
        super().setUp()
        self.tmp_path = Path(tempfile.mkdtemp(prefix="ygg-pandas-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)
        super().tearDown()

    # --- convenience constructors --------------------------------------
    def df(self, data: Any, **kwargs: Any) -> "pd.DataFrame":
        """Shorthand for ``pd.DataFrame(data, **kwargs)``."""
        return self.pd.DataFrame(data, **kwargs)

    def series(self, data: Any, **kwargs: Any) -> "pd.Series":
        """Shorthand for ``pd.Series(data, **kwargs)``."""
        return self.pd.Series(data, **kwargs)

    # --- Arrow interop -------------------------------------------------
    def arrow_to_pandas(self, table: "pa.Table") -> "pd.DataFrame":
        """Convert a ``pa.Table`` to a pandas DataFrame."""
        return table.to_pandas()

    def pandas_to_arrow(self, df: "pd.DataFrame") -> "pa.Table":
        """Convert a pandas DataFrame to a ``pa.Table``.

        Loads pyarrow on demand via ``runtime_import_module``; will raise
        :class:`unittest.SkipTest` if pyarrow isn't available.
        """
        install = _auto_install(self.auto_install)
        try:
            pa = runtime_import_module("pyarrow", install=install)
        except ImportError:
            raise unittest.SkipTest(
                "'pyarrow' is required for Arrow interop but is not installed."
            )
        return pa.Table.from_pandas(df, preserve_index=False)

    # --- assertions ----------------------------------------------------
    def assertFrameEqual(
        self,
        actual: "pd.DataFrame",
        expected: "pd.DataFrame | dict[str, Any] | list[dict[str, Any]]",
        *,
        ordered: bool = True,
        check_dtype: bool = True,
        check_index: bool = False,
        **kwargs: Any,
    ) -> None:
        """Assert two DataFrames are equal.

        Parameters
        ----------
        actual : pd.DataFrame
        expected : pd.DataFrame | dict | list[dict]
            If a dict or list of dicts is given, it's coerced via
            ``pd.DataFrame``.
        ordered : bool, default True
            If False, both sides are sorted by all columns before compare
            and the index is reset.
        check_dtype : bool, default True
        check_index : bool, default False
            If False, reset both indexes before comparing.
        kwargs :
            Additional args forwarded to ``pandas.testing.assert_frame_equal``.
        """
        pd = self.pd
        if not isinstance(expected, pd.DataFrame):
            expected = pd.DataFrame(expected)

        left, right = actual, expected

        if not ordered:
            cols = list(right.columns)
            left = left.sort_values(cols).reset_index(drop=True)
            right = right.sort_values(cols).reset_index(drop=True)
        elif not check_index:
            left = left.reset_index(drop=True)
            right = right.reset_index(drop=True)

        from pandas.testing import assert_frame_equal
        try:
            assert_frame_equal(left, right, check_dtype=check_dtype, **kwargs)
        except AssertionError as exc:
            self.fail(f"DataFrames differ:\n{exc}")

    def assertSeriesEqual(
        self,
        actual: "pd.Series",
        expected: "pd.Series | list[Any]",
        *,
        check_dtype: bool = True,
        check_index: bool = False,
        check_names: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Assert two Series are equal.

        ``check_names`` defaults to True when ``expected`` is a Series and
        False when it's a raw list (a list has no name).
        """
        pd = self.pd
        coerced = not isinstance(expected, pd.Series)
        if coerced:
            expected = pd.Series(expected, name=actual.name)

        if check_names is None:
            check_names = not coerced

        left, right = actual, expected
        if not check_index:
            left = left.reset_index(drop=True)
            right = right.reset_index(drop=True)

        from pandas.testing import assert_series_equal
        try:
            assert_series_equal(
                left, right,
                check_dtype=check_dtype,
                check_names=check_names,
                **kwargs,
            )
        except AssertionError as exc:
            self.fail(f"Series differ:\n{exc}")