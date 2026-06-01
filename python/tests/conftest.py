import logging
import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "python" / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logger = logging.getLogger("yggdrasil")

def setup_logging(level: int = logging.INFO) -> None:
    # Avoid duplicate handlers if this gets called twice (common in notebooks/tests)
    if logger.handlers:
        return

    logger.setLevel(level)
    logger.propagate = False  # keep it from double-logging via root

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    )
    handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(handler)

# --- usage ---
setup_logging(logging.DEBUG)


# ---------------------------------------------------------------------------
# Spark test categorisation (see pyproject `[tool.pytest] markers`)
# ---------------------------------------------------------------------------
# Booting a local PySpark JVM is the suite's single heaviest cost, so Spark
# tests are made explicitly selectable:
#
#   * Every ``SparkTestCase`` subclass and every test using the ``spark``
#     fixture is auto-tagged ``spark`` — run a fast, JVM-free suite with
#     ``pytest -m "not spark"``.
#   * The heaviest Spark tests (tagged ``spark_integration``) are skipped by
#     default and only run when ``YGGDRASIL_SPARK_INTEGRATION`` is set — the
#     same env-gate pattern as the Databricks ``integration`` marker.
def _spark_test_case():
    try:
        from yggdrasil.spark.tests import SparkTestCase

        return SparkTestCase
    except Exception:
        return None


def pytest_collection_modifyitems(config, items):
    spark_case = _spark_test_case()
    run_heavy = bool(os.environ.get("YGGDRASIL_SPARK_INTEGRATION"))
    skip_heavy = pytest.mark.skip(
        reason="heavy Spark integration test — set YGGDRASIL_SPARK_INTEGRATION=1 to run"
    )
    for item in items:
        cls = getattr(item, "cls", None)
        is_spark_case = (
            spark_case is not None
            and cls is not None
            and isinstance(cls, type)
            and issubclass(cls, spark_case)
        )
        uses_spark_fixture = "spark" in getattr(item, "fixturenames", ())
        heavy = item.get_closest_marker("spark_integration") is not None

        if is_spark_case or uses_spark_fixture or heavy:
            item.add_marker(pytest.mark.spark)
        if heavy and not run_heavy:
            item.add_marker(skip_heavy)
