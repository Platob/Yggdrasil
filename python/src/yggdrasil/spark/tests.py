"""Unittest base class for PySpark tests.

Provides a single global SparkSession shared across *all* SparkTestCase
subclasses in the process. Spark startup is expensive (~5-10s) — creating a
session per test class would make the suite painfully slow.

Quick start
-----------
::

    from yggdrasil.spark.tests import SparkTestCase

    class TestMyStuff(SparkTestCase):
        def test_basic(self):
            df = self.spark.createDataFrame([(1, "a")], ["id", "val"])
            self.assertEqual(df.count(), 1)

        def test_arrow_roundtrip(self):
            import pyarrow as pa
            tbl = pa.table({"id": [1, 2], "val": ["a", "b"]})
            df = self.arrow_to_spark(tbl)
            self.assertSparkEqual(df, tbl)

        def test_with_scratch_path(self):
            # self.tmp_path is a fresh per-test directory, cleaned up after
            out = self.tmp_path / "data.parquet"
            self.spark.range(10).write.parquet(str(out))

pytest users
------------
The module also exposes a ``spark`` fixture (session-scoped) so you can
skip the class hierarchy if you prefer::

    def test_something(spark):
        assert spark.range(5).count() == 5

Design notes
------------
- The SparkSession is created lazily on first use and reused for the
  lifetime of the Python process. ``tearDownClass`` never stops it.
- Because the session is shared, ``spark_extra_config`` only takes effect
  for the *first* class that triggers creation. If you need a bespoke
  session, call ``reset_global_session()`` explicitly (expensive).
- Arrow interop uses ``spark.sql.execution.arrow.pyspark.enabled=true``
  by default — zero-copy-ish transfer for supported types.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

if TYPE_CHECKING:
    import pyarrow as pa
    from pyspark.sql import DataFrame, SparkSession

__all__ = [
    "SparkTestCase",
    "get_spark",
    "reset_global_session",
]

LOGGER = logging.getLogger(__name__)

# Module-level singleton — created once, reused everywhere.
_global_spark: "SparkSession | None" = None
_global_tmpdir: Path | None = None


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------
def _default_config(warehouse_dir: Path) -> dict[str, str]:
    """Sensible defaults for local-mode integration tests."""
    return {
        "spark.driver.host": "localhost",
        "spark.driver.bindAddress": "127.0.0.1",
        # Tiny shuffle partitions — we're not doing real work here.
        "spark.sql.shuffle.partitions": "1",
        "spark.default.parallelism": "2",
        # Arrow accelerates toPandas / createDataFrame(pandas).
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.execution.arrow.pyspark.fallback.enabled": "true",
        # Surface Python worker crashes instead of swallowing them.
        "spark.python.worker.faulthandler.enabled": "true",
        # Keep all test state out of the user's home directory.
        "spark.sql.warehouse.dir": str(warehouse_dir / "warehouse"),
        "spark.local.dir": str(warehouse_dir / "local"),
        # Fail fast rather than retrying on local-mode hiccups.
        "spark.task.maxFailures": "1",
        # Don't bind a UI — noisy and flaky in CI.
        "spark.ui.enabled": "false",
        "spark.ui.showConsoleProgress": "false",
    }


def _quiet_loggers() -> None:
    """Mute the chattiest Spark/py4j loggers at the Python level."""
    for name in (
        "py4j",
        "py4j.clientserver",
        "py4j.java_gateway",
        "pyspark",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_spark(
    app_name: str = "yggdrasil-test",
    extra_config: dict[str, str] | None = None,
) -> "SparkSession":
    """Return the global test SparkSession, creating it on first call.

    Subsequent calls ignore ``app_name`` and ``extra_config`` — the first
    caller wins. Use :func:`reset_global_session` if you need a fresh one.
    """
    global _global_spark, _global_tmpdir

    if _global_spark is not None:
        return _global_spark

    from yggdrasil.spark.setup import (
        configure_java_compat,
        ensure_hadoop_home,
        ensure_java,
    )

    # Make sure we have a compatible JDK (downloads Zulu 21 if needed).
    ensure_java(auto_download=True)
    # Windows needs Hadoop native binaries for local-mode Spark.
    ensure_hadoop_home()
    # JVM compat flags for Java 17+.
    configure_java_compat()

    # PySpark 4.x workers need to know which Python to use, otherwise they
    # can pick up a different interpreter and crash on startup.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    # Process-wide scratch dir for warehouse + spark.local.dir.
    _global_tmpdir = Path(tempfile.mkdtemp(prefix="ygg-spark-test-"))

    from pyspark.sql import SparkSession

    merged_config = _default_config(_global_tmpdir)
    merged_config["spark.app.name"] = app_name
    if extra_config:
        merged_config.update(extra_config)

    builder = SparkSession.builder.master("local[*]")
    for k, v in merged_config.items():
        builder = builder.config(k, v)

    _global_spark = builder.getOrCreate()
    _global_spark.sparkContext.setLogLevel("WARN")
    _quiet_loggers()
    LOGGER.info(
        "Global test SparkSession ready — version %s, scratch=%s",
        _global_spark.version,
        _global_tmpdir,
    )
    return _global_spark


def reset_global_session() -> None:
    """Stop and discard the global session. Call sparingly — it's expensive."""
    global _global_spark, _global_tmpdir
    if _global_spark is not None:
        try:
            _global_spark.stop()
        except Exception:
            LOGGER.exception("Error stopping global SparkSession")
        _global_spark = None
    if _global_tmpdir is not None and _global_tmpdir.exists():
        shutil.rmtree(_global_tmpdir, ignore_errors=True)
        _global_tmpdir = None


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------
class SparkTestCase(unittest.TestCase):
    """Base class for Spark integration tests.

    A single global SparkSession is created on first use and shared across
    every subclass in the process. Each test method also gets a fresh
    ``self.tmp_path`` (``pathlib.Path``) that is cleaned up in ``tearDown``.

    Attributes
    ----------
    spark : SparkSession
        The shared global session. Populated by ``setUpClass``.
    tmp_path : pathlib.Path
        Per-test scratch directory. Populated by ``setUp``.

    Class attributes
    ----------------
    spark_app_name : str
        Spark application name. Only effective for the first class that
        triggers session creation.
    spark_extra_config : dict[str, str]
        Extra Spark config entries. Same caveat — only the first class wins.
    """

    spark_app_name: ClassVar[str] = "yggdrasil-test"
    spark_extra_config: ClassVar[dict[str, str]] = {}

    spark: ClassVar["SparkSession"]
    tmp_path: Path

    # --- lifecycle ------------------------------------------------------
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Guard: skip the whole class if PySpark is not installed.
        try:
            import pyspark  # noqa: F401
        except ImportError:
            raise unittest.SkipTest(
                "PySpark is not installed. "
                "Install it with: pip install pyspark  "
                "or: pip install 'ygg[spark]'"
            )

        try:
            cls.spark = get_spark(
                app_name=cls.spark_app_name,
                extra_config=cls.spark_extra_config,
            )
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc))

    @classmethod
    def tearDownClass(cls) -> None:
        # Do NOT stop the session — it's shared globally.
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self.tmp_path = Path(tempfile.mkdtemp(prefix="ygg-sparktest-"))

    def tearDown(self) -> None:
        # Drop any temp views this test created so they don't leak across tests.
        try:
            self.spark.catalog.clearCache()
        except Exception:
            pass
        shutil.rmtree(self.tmp_path, ignore_errors=True)
        super().tearDown()

    # --- convenience constructors --------------------------------------
    def df(
        self,
        data: Iterable[Any],
        schema: Any = None,
    ) -> "DataFrame":
        """Shorthand for ``self.spark.createDataFrame(data, schema)``."""
        return self.spark.createDataFrame(list(data), schema=schema)

    def arrow_to_spark(self, table: "pa.Table") -> "DataFrame":
        """Convert a ``pyarrow.Table`` to a Spark DataFrame via pandas.

        Uses Arrow-backed pandas for zero-copy-ish transfer. Good enough
        for tests; don't use this on gigabyte-scale data.
        """
        return self.spark.createDataFrame(table.to_pandas())

    def spark_to_arrow(self, df: "DataFrame") -> "pa.Table":
        """Materialise a Spark DataFrame as a ``pyarrow.Table``."""
        import pyarrow as pa

        return pa.Table.from_pandas(df.toPandas(), preserve_index=False)

    # --- assertions ----------------------------------------------------
    def assertDataFrameEqual(
        self,
        actual: "DataFrame",
        expected: "DataFrame | pa.Table | list[dict[str, Any]]",
        *,
        ordered: bool = False,
        check_schema: bool = True,
    ) -> None:
        """Assert two DataFrames are equal.

        Parameters
        ----------
        actual : DataFrame
            The DataFrame produced by the code under test.
        expected : DataFrame | pa.Table | list[dict]
            The reference value. Accepts a Spark DataFrame, a pyarrow
            Table, or a list of row dicts (useful for inline literals).
        ordered : bool, default False
            If False, both sides are sorted by all columns before compare.
        check_schema : bool, default True
            If True, schemas (names + types) must match exactly.
        """
        from pyspark.sql import DataFrame as SparkDF

        # Normalise expected → Spark DataFrame.
        if isinstance(expected, SparkDF):
            expected_df = expected
        elif isinstance(expected, list):
            expected_df = self.spark.createDataFrame(expected)
        else:
            # Duck-type as pyarrow.Table.
            try:
                expected_df = self.arrow_to_spark(expected)
            except AttributeError as exc:
                raise TypeError(
                    f"expected must be DataFrame, pa.Table, or list[dict]; "
                    f"got {type(expected).__name__}"
                ) from exc

        if check_schema:
            self.assertEqual(
                sorted(actual.schema.fields, key=lambda f: f.name),
                sorted(expected_df.schema.fields, key=lambda f: f.name),
                "DataFrame schemas differ",
            )

        actual_rows = actual.collect()
        expected_rows = expected_df.collect()

        if not ordered:
            # Row is hashable when all field values are — fall back to
            # string-repr sort otherwise (cheap and deterministic enough).
            def _key(r: Any) -> str:
                return repr(sorted(r.asDict().items()))

            actual_rows = sorted(actual_rows, key=_key)
            expected_rows = sorted(expected_rows, key=_key)

        if actual_rows != expected_rows:
            # Build a readable diff.
            lines = ["DataFrames differ.", "--- expected ---"]
            lines.extend(repr(r) for r in expected_rows)
            lines.append("--- actual ---")
            lines.extend(repr(r) for r in actual_rows)
            self.fail("\n".join(lines))

    # Alias with naming that matches assertSparkEqual shorthand.
    assertSparkEqual = assertDataFrameEqual

    def assertSchemaEqual(
        self,
        actual: "DataFrame",
        expected_fields: list[tuple[str, Any]],
    ) -> None:
        """Assert a DataFrame has exactly the given ``(name, dtype)`` fields.

        ``dtype`` may be a ``pyspark.sql.types.DataType`` instance or its
        simpleString form (``"int"``, ``"string"``, ``"array<long>"``, ...).
        """
        from pyspark.sql.types import DataType

        got = [(f.name, f.dataType.simpleString()) for f in actual.schema.fields]
        want = [
            (n, t.simpleString() if isinstance(t, DataType) else str(t))
            for n, t in expected_fields
        ]
        self.assertEqual(got, want, "DataFrame schemas differ")


# ---------------------------------------------------------------------------
# pytest fixture (optional, auto-detected)
# ---------------------------------------------------------------------------
try:
    import pytest

    @pytest.fixture(scope="session")
    def spark() -> "SparkSession":
        """Session-scoped pytest fixture exposing the shared SparkSession."""
        return get_spark()

    @pytest.fixture()
    def spark_tmp_path(tmp_path: Path) -> Path:
        """Per-test scratch path — just an alias for clarity in Spark tests."""
        return tmp_path

except ImportError:  # pragma: no cover — pytest is optional
    pass