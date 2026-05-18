"""``Dataset.apply`` / ``map`` / ``parallelize`` accept any pyfunc shape.

The driver-side coverage for :func:`yggdrasil.dataclasses.build_row_invoker`
lives in ``test_safe_function``; this file proves the same shapes survive
the Spark executor round trip:

* single-positional function — ``def f(x): ...`` (the historical shape).
* multi-arg function — ``def f(id: int, name: str): ...`` whose dict
  rows spread as ``**kwargs`` with type coercion via the cast registry.
* ``**kwargs`` catch-all — ``def f(**row): ...`` receives the full dict.
* ``*args`` catch-all — ``def f(*xs): ...`` receives tuple/list rows
  spread positionally; scalar rows arrive as a single arg.
* keyword-only — ``def f(*, id, name): ...`` survives the same spread.

The fixtures forge a pure-local ``SparkSession`` directly (no
``PyEnv.spark_session`` round trip) so the test runs in environments
where ``databricks-connect`` is not installed but the local pyspark
JVM-mode is reachable.
"""
from __future__ import annotations

import os
import unittest

from yggdrasil.data import field, schema
from yggdrasil.data.types.primitive import (
    Float64Type,
    Int64Type,
    StringType,
)
from yggdrasil.io.tabular.spark import Dataset


def _local_spark():
    """Build a local-only ``SparkSession`` for the apply-signature tests.

    Skips cleanly when pyspark is missing or only supports
    Databricks Connect (pyspark 4.x without databricks-connect).
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise unittest.SkipTest("pyspark not installed")

    # Avoid PyEnv's connect-first resolution.
    os.environ.pop("DATABRICKS_HOST", None)
    # Spark 3.5 + Java 21 need the legacy ``sun.misc.Unsafe`` / direct
    # ``ByteBuffer.<init>(long, int)`` reflection paths reopened for Arrow
    # to allocate off-heap buffers; without these JVM args, ``mapInArrow``
    # crashes with ``UnsupportedOperationException`` from MemoryUtil.
    jvm_opens = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.misc=ALL-UNNAMED"
    )
    try:
        return (
            SparkSession.builder
            .master("local[2]")
            .appName("ygg-test-apply-signatures")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
            .config("spark.driver.extraJavaOptions", jvm_opens)
            .config("spark.executor.extraJavaOptions", jvm_opens)
            .getOrCreate()
        )
    except RuntimeError as exc:
        raise unittest.SkipTest(f"local SparkSession unavailable: {exc}")


_OUT_SCHEMA = schema([
    field("id", Int64Type, nullable=False),
    field("label", StringType),
    field("score", Float64Type),
])


# ---------------------------------------------------------------------------
# Module-scope user functions — pickled to the executor by yggdrasil.pickle.
# Lambdas / closures would also work, but referencing module-level names
# keeps the integration assertions readable.
# ---------------------------------------------------------------------------


def _single_arg(x: int) -> dict:
    return {"id": int(x), "label": f"row-{int(x)}", "score": float(x) * 1.5}


def _multi_arg(id: int, name: str) -> dict:
    return {"id": id, "label": name, "score": float(id)}


def _var_kw(**row) -> dict:
    return {
        "id": int(row["id"]),
        "label": str(row.get("name", "n/a")),
        "score": float(row.get("score", 0.0)),
    }


def _var_pos(*xs) -> dict:
    a, b = xs[0], xs[1]
    return {"id": int(a), "label": str(b), "score": float(a) + 0.5}


def _kw_only(*, id: int, name: str) -> dict:
    return {"id": id, "label": name, "score": float(id) * 2.0}


def _coerce_strings(id: int, name: str) -> dict:
    # Annotated ``int`` — string ``id`` must be coerced via the cast registry.
    return {"id": id, "label": name, "score": float(id)}


# ---------------------------------------------------------------------------
# TestCase — self-managed local SparkSession + Dataset module-install stub.
# ---------------------------------------------------------------------------


class _AppliedSignaturesBase(unittest.TestCase):
    spark = None  # type: ignore[assignment]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.spark = _local_spark()
        # Skip the ygg / pyarrow archive ship — the local cluster already
        # has the same interpreter and ``sys.path`` as the driver, so the
        # archive step would only burn CPU. Patch the module-level
        # function the dataset calls into (a class-level stub would be
        # dead code; ``_ensure_installed_on_session`` imports from the
        # module, not the class).
        from yggdrasil.spark import frame as _frame_module
        cls._frame_module = _frame_module
        cls._orig_install = _frame_module._install_modules_on_executors
        _frame_module._install_modules_on_executors = lambda _session, modules: set(modules)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._frame_module._install_modules_on_executors = cls._orig_install
        # Leave the session alive across the test process — Spark startup
        # is the dominant cost; tearing down would slow neighbours that
        # share the same JVM.
        super().tearDownClass()


class TestDatasetApplySignatures(_AppliedSignaturesBase):

    def test_single_arg_dynamic_to_typed(self) -> None:
        dyn = Dataset.from_iterable(range(5), spark_session=self.spark)
        out = dyn.apply(_single_arg, _OUT_SCHEMA)
        rows = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["id"] for r in rows], list(range(5)))
        self.assertEqual([r["label"] for r in rows], [f"row-{i}" for i in range(5)])

    def test_multi_arg_spreads_dict_as_kwargs(self) -> None:
        rows_in = [{"id": i, "name": f"r{i}"} for i in range(4)]
        dyn = Dataset.from_iterable(rows_in, spark_session=self.spark)
        out = dyn.apply(_multi_arg, _OUT_SCHEMA)
        rows = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["label"] for r in rows], [f"r{i}" for i in range(4)])

    def test_var_kw_function_receives_full_dict(self) -> None:
        rows_in = [{"id": i, "name": f"v{i}", "score": float(i)} for i in range(3)]
        dyn = Dataset.from_iterable(rows_in, spark_session=self.spark)
        out = dyn.apply(_var_kw, _OUT_SCHEMA)
        rows = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["label"] for r in rows], [f"v{i}" for i in range(3)])
        self.assertEqual([r["score"] for r in rows], [0.0, 1.0, 2.0])

    def test_var_positional_spreads_tuple_rows(self) -> None:
        # Each row is a 2-tuple — *xs catches both.
        rows_in = [(i, f"t{i}") for i in range(4)]
        dyn = Dataset.from_iterable(rows_in, spark_session=self.spark)
        out = dyn.apply(_var_pos, _OUT_SCHEMA)
        rows = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["label"] for r in rows], [f"t{i}" for i in range(4)])

    def test_keyword_only_function(self) -> None:
        rows_in = [{"id": i, "name": f"k{i}"} for i in range(3)]
        dyn = Dataset.from_iterable(rows_in, spark_session=self.spark)
        out = dyn.apply(_kw_only, _OUT_SCHEMA)
        rows = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["score"] for r in rows], [0.0, 2.0, 4.0])

    def test_annotation_coerces_string_inputs(self) -> None:
        # String-shaped ``id`` survives the int annotation via the cast registry.
        rows_in = [{"id": str(i), "name": f"c{i}"} for i in range(3)]
        dyn = Dataset.from_iterable(rows_in, spark_session=self.spark)
        out = dyn.apply(_coerce_strings, _OUT_SCHEMA)
        rows = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["id"] for r in rows], [0, 1, 2])

    def test_map_uses_same_invoker(self) -> None:
        # ``.map(func)`` (no schema) routes through the same dispatcher —
        # the dynamic output should still carry the right shapes.
        rows_in = [{"id": i, "name": f"m{i}"} for i in range(4)]
        dyn = Dataset.from_iterable(rows_in, spark_session=self.spark)
        out = dyn.map(_multi_arg)
        collected = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["label"] for r in collected], [f"m{i}" for i in range(4)])

    def test_parallelize_with_multi_arg_function(self) -> None:
        rows_in = [{"id": i, "name": f"p{i}"} for i in range(5)]
        out = Dataset.parallelize(
            _multi_arg, rows_in, schema=_OUT_SCHEMA, spark_session=self.spark,
        )
        rows = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["label"] for r in rows], [f"p{i}" for i in range(5)])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
