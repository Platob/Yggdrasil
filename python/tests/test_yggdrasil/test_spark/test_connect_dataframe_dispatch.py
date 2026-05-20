"""Regression coverage for Spark Connect DataFrame dispatch.

On PySpark 3.5 and earlier (the DBR 14/15 wire), ``pyspark.sql.connect.dataframe.DataFrame`` is **not** a subclass of
``pyspark.sql.DataFrame`` — they are parallel implementations.  The
cast dispatchers in :mod:`yggdrasil.arrow.cast`, :mod:`yggdrasil.spark.cast`,
and :mod:`yggdrasil.polars.cast` previously gated on
``isinstance(obj, pyspark.sql.DataFrame)`` and rejected Connect
DataFrames returned by ``spark.sql(...)`` on Databricks Connect with::

    TypeError: Unsupported Spark object: DataFrame[...]

These tests confirm :func:`yggdrasil.lazy_imports.spark_dataframe_classes`
exposes both classes when available, and that
:func:`yggdrasil.spark.cast.any_to_spark_dataframe` no-ops on a Spark
DataFrame that is **only** a subclass of the Connect class — the same
shape the Databricks runtime hands back.

No live Spark session is required; the test fabricates a stand-in
class that mimics the duck-type contract and asserts the dispatch
path treats it as a Spark DataFrame.
"""

from __future__ import annotations

import sys
import types
import unittest


class TestSparkDataFrameClassesHelper(unittest.TestCase):
    def test_classes_includes_classic(self) -> None:
        try:
            import pyspark.sql  # noqa: F401
        except ImportError:
            self.skipTest("PySpark not installed")

        from yggdrasil.lazy_imports import spark_dataframe_classes
        from pyspark.sql import DataFrame as ClassicDataFrame

        classes = spark_dataframe_classes()
        self.assertIn(ClassicDataFrame, classes)

    def test_classes_includes_connect_when_importable(self) -> None:
        try:
            from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
        except ImportError:
            self.skipTest("Spark Connect deps (grpcio / databricks-connect) not installed")

        from yggdrasil.lazy_imports import spark_dataframe_classes

        classes = spark_dataframe_classes()
        self.assertIn(ConnectDataFrame, classes)

    def test_column_classes_includes_classic(self) -> None:
        try:
            from pyspark.sql import Column as ClassicColumn
        except ImportError:
            self.skipTest("PySpark not installed")

        from yggdrasil.lazy_imports import spark_column_classes

        classes = spark_column_classes()
        self.assertIn(ClassicColumn, classes)


class TestSparkConnectDispatchSurrogate(unittest.TestCase):
    """Inject a stand-in Connect DataFrame class and confirm dispatch.

    Builds a fake ``pyspark.sql.connect.dataframe`` module exposing a
    DataFrame class that is *not* a subclass of ``pyspark.sql.DataFrame``.
    Then clears the ``lazy_imports`` cache so :func:`spark_dataframe_classes`
    picks it up alongside the real classic class, and verifies the cast
    dispatchers accept an instance of the surrogate as a Spark DataFrame.

    This is the exact pre-fix failure mode on Databricks Connect with
    PySpark 3.5.
    """

    def setUp(self) -> None:
        try:
            import pyspark.sql  # noqa: F401
        except ImportError:
            self.skipTest("PySpark not installed")

        # Build a surrogate Connect module + DataFrame class. We can't
        # import the real ``pyspark.sql.connect.dataframe`` here because
        # it requires grpcio + databricks-connect; the surrogate covers
        # the dispatch contract (isinstance check) without the deps.
        self._saved_modules: dict[str, object] = {}
        for mod_name in (
            "pyspark.sql.connect",
            "pyspark.sql.connect.dataframe",
            "pyspark.sql.connect.column",
        ):
            if mod_name in sys.modules:
                self._saved_modules[mod_name] = sys.modules[mod_name]

        connect_pkg = types.ModuleType("pyspark.sql.connect")
        connect_df_mod = types.ModuleType("pyspark.sql.connect.dataframe")
        connect_col_mod = types.ModuleType("pyspark.sql.connect.column")

        class _ConnectDataFrame:
            """Surrogate for ``pyspark.sql.connect.dataframe.DataFrame``."""

            def __init__(self, table) -> None:
                self._table = table

            @property
            def schema(self):
                from pyspark.sql.types import StructType
                # Use a real Spark schema so downstream code works.
                from yggdrasil.data.schema import Schema
                return Schema.from_arrow(self._table.schema).to_spark_schema()

            @property
            def columns(self) -> list[str]:
                return self._table.column_names

            def select(self, *cols: str) -> "_ConnectDataFrame":
                return _ConnectDataFrame(self._table.select(list(cols)))

            def toArrow(self):
                return self._table

        class _ConnectColumn:
            pass

        # Override the dunder so ``ObjectSerde.full_namespace`` reports
        # the surrogate as living under ``pyspark.sql.connect.dataframe``
        # — the namespace gate in ``any_to_arrow_table`` /
        # ``any_to_arrow_batch_iterator`` only routes objects whose
        # full namespace starts with ``pyspark.``.
        _ConnectDataFrame.__module__ = "pyspark.sql.connect.dataframe"
        _ConnectColumn.__module__ = "pyspark.sql.connect.column"

        connect_df_mod.DataFrame = _ConnectDataFrame
        connect_col_mod.Column = _ConnectColumn
        sys.modules["pyspark.sql.connect"] = connect_pkg
        sys.modules["pyspark.sql.connect.dataframe"] = connect_df_mod
        sys.modules["pyspark.sql.connect.column"] = connect_col_mod

        from yggdrasil import lazy_imports

        lazy_imports.spark_dataframe_classes.cache_clear()
        lazy_imports.spark_column_classes.cache_clear()
        self._surrogate_df_cls = _ConnectDataFrame

    def tearDown(self) -> None:
        # Restore module state and the cached class tuple.
        for mod_name in (
            "pyspark.sql.connect.column",
            "pyspark.sql.connect.dataframe",
            "pyspark.sql.connect",
        ):
            if mod_name in self._saved_modules:
                sys.modules[mod_name] = self._saved_modules[mod_name]  # type: ignore[assignment]
            else:
                sys.modules.pop(mod_name, None)

        from yggdrasil import lazy_imports

        lazy_imports.spark_dataframe_classes.cache_clear()
        lazy_imports.spark_column_classes.cache_clear()

    def test_dataframe_classes_picks_up_surrogate(self) -> None:
        from yggdrasil.lazy_imports import spark_dataframe_classes

        classes = spark_dataframe_classes()
        self.assertIn(self._surrogate_df_cls, classes)
        from pyspark.sql import DataFrame as ClassicDataFrame
        self.assertIn(ClassicDataFrame, classes)

    def test_any_to_spark_dataframe_accepts_surrogate(self) -> None:
        # Without the fix this raises ``TypeError: Unsupported Spark
        # object`` from ``_spark_to_arrow`` further down the call chain.
        import pyarrow as pa

        from yggdrasil.spark.cast import any_to_spark_dataframe

        # ``any_to_spark_dataframe`` short-circuits when the input is
        # already a "Spark DataFrame" (per the helper) — it does NOT
        # try to fan out to a Spark session.
        table = pa.table({"a": [1, 2, 3]})
        surrogate = self._surrogate_df_cls(table)

        # Returns the input unchanged through ``cast_spark_tabular``
        # (no target schema → identity path). The key assertion is
        # that the call does not raise.
        from yggdrasil.data.options import CastOptions

        out = any_to_spark_dataframe(surrogate, options=CastOptions())
        self.assertIs(out, surrogate)

    def test_arrow_spark_to_arrow_accepts_surrogate(self) -> None:
        # ``_spark_to_arrow`` (the function in the user's failing
        # traceback) gets the surrogate via the namespace-dispatched
        # path inside ``any_to_arrow_table``. Pre-fix this hit
        # ``TypeError: Unsupported Spark object: DataFrame[...]``.
        import pyarrow as pa

        from yggdrasil.arrow.cast import any_to_arrow_table

        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        surrogate = self._surrogate_df_cls(table)

        # The surrogate's ``toArrow`` returns the underlying table —
        # round-trip should preserve column names and row count.
        out = any_to_arrow_table(surrogate)
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column_names), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
