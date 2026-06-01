"""Shared base for Databricks live-integration tests.

:class:`DatabricksIntegrationCase` skips the whole class when
``DATABRICKS_HOST`` is missing, then exposes a single
:class:`DatabricksClient` (``cls.client``) and the bound
``WorkspaceClient`` (``cls.workspace``) for the test methods to
drive against a real workspace.

Subclasses are tagged with the ``integration`` pytest marker so the
default local run (``pytest``) skips them along with anything else
that touches a live Databricks endpoint.
"""

from __future__ import annotations

import os
import unittest
from typing import Any, ClassVar

import pytest

from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.client import invalidate_env_defaults


__all__ = ["DatabricksIntegrationCase"]


@pytest.mark.integration
class DatabricksIntegrationCase(unittest.TestCase):
    """Base class for tests that need a live Databricks workspace.

    The class is skipped at ``setUpClass`` time when ``DATABRICKS_HOST``
    is unset (or empty) so a plain ``pytest`` run on a developer
    laptop never touches the network. When the env var *is* set,
    ``cls.client`` is built from the standard ``DATABRICKS_*`` env
    vars (host, token / OAuth, profile, ...) — exactly what a freshly
    constructed :class:`DatabricksClient` already reads.
    """

    client: ClassVar[DatabricksClient]
    workspace: ClassVar[Any]
    spark: ClassVar[Any] = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        host = os.environ.get("DATABRICKS_HOST", "").strip()
        if not host:
            raise unittest.SkipTest(
                "DATABRICKS_HOST is not set — skipping Databricks "
                "integration tests. Set DATABRICKS_HOST (and the matching "
                "credentials, e.g. DATABRICKS_TOKEN or a config profile) "
                "to run them."
            )
        # Reload from the current environment. The client snapshots the
        # ``DATABRICKS_*`` env defaults once per process, so a client built
        # earlier in the same pytest run (e.g. by a unit test) would pin a
        # stale snapshot taken before these vars were set. Drop it so the
        # build below resolves host / token / profile from the live env.
        invalidate_env_defaults()
        cls.client = DatabricksClient()
        cls.workspace = cls.client.workspace_client()

        # Best-effort Spark Connect bootstrap — leaves ``cls.spark = None``
        # when ``databricks-connect`` isn't installed or the workspace
        # can't hand back a session. Subclasses that genuinely need Spark
        # should guard on ``cls.spark`` (or call ``client.spark()`` directly
        # and let it raise) instead of relying on this slot.
        try:
            cls.spark = cls.client.spark()
        except Exception:
            cls.spark = None
