"""
Shared base ``TestCase`` for all Databricks integration tests.

Every integration test class inherits from :class:`DatabricksCase` instead of
``unittest.TestCase`` directly.  The base class handles:

- Instant skip when ``DATABRICKS_HOST`` is not set in the environment.
- Creating and connecting a :class:`~yggdrasil.databricks.workspaces.Workspace`
  (which is also a ``DatabricksClient``) shared across the whole test class.
- Graceful skip when auth or network is unavailable.
- Closing the workspace connection in ``tearDownClass``.

Subclass pattern::

    class TestMyFeature(DatabricksCase):

        @classmethod
        def setUpClass(cls) -> None:
            super().setUpClass()          # env-var guard + workspace connection
            cls.engine = cls.workspace.sql(...)  # subclass-specific setup

        @classmethod
        def tearDownClass(cls) -> None:
            cls.engine.close()            # subclass-specific teardown
            super().tearDownClass()       # closes workspace
"""

from __future__ import annotations

import os
import unittest

__all__ = ["DatabricksCase"]


_SKIP_MSG = (
    "Integration tests require DATABRICKS_HOST to be set. "
    "Example: DATABRICKS_HOST=https://dbc-xxx.cloud.databricks.com/"
)


class DatabricksCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if not os.environ.get("DATABRICKS_HOST"):
            raise unittest.SkipTest(_SKIP_MSG)

        from yggdrasil.databricks import DatabricksClient

        try:
            cls.client = DatabricksClient().connect()
            # Lightweight auth probe — fails fast when the token / profile is wrong
            cls.client.workspace_client().current_user.me()
        except unittest.SkipTest:
            raise
        except Exception as exc:
            raise unittest.SkipTest(f"Databricks workspace not reachable: {exc}")

    @property
    def workspace(self):
        return self.client.workspace

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.workspace.close()
        except Exception:
            pass

