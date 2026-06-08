"""Live-integration tests for :class:`SQLEngine` and :class:`Table`.

Skipped unless ``DATABRICKS_HOST`` (and the matching credentials) are
exported via the standard SDK env vars — see
:class:`DatabricksIntegrationCase`.

Scope
-----
The fixture is pinned to ``trading.unittest`` (the catalog/schema can
be overridden via :envvar:`DATABRICKS_INTEGRATION_CATALOG` /
:envvar:`DATABRICKS_INTEGRATION_SCHEMA`). Each test touches a unique
managed table name so concurrent runs don't collide and a partial
failure leaves at most one orphan table behind.

Auto-create policy
------------------
The engine and table methods exercised here are *opportunistic*:
operations are attempted directly, and the catalog / schema / table
are only created on demand when the operation surfaces a missing
resource. ``setUpClass`` falls back to ``ensure_created`` when the
upfront probe fails so the run can still proceed against a pristine
workspace.
"""

from __future__ import annotations

import pytest

from .. import DatabricksIntegrationCase

__all__ = [
    "TestSparkConnect",
]


class TestSparkConnect(DatabricksIntegrationCase):

    @classmethod
    def setUpClass(cls) -> None:
        pytest.importorskip("databricks.connect")
        super().setUpClass()
        cls.spark = cls.client.spark()

    def test_spark_connected(self):
        assert self.spark is not None
