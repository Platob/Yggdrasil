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

import logging
import os
import secrets
import unittest
from typing import Any, ClassVar

import pytest

from yggdrasil.databricks import DatabricksClient


__all__ = ["DatabricksIntegrationCase"]

logger = logging.getLogger(__name__)


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

    #: The shared, persistent home for all live integration fixtures.
    #: Catalog ``trading_tgp_dev`` / schema ``ygg_integration`` by default
    #: (override via the matching env vars). The schema is created if
    #: missing and **never dropped** — only throw-away schemas minted by
    #: :meth:`scratch_schema` (always ``ygg_integration``-prefixed) are.
    INTEGRATION_CATALOG: ClassVar[str] = os.environ.get(
        "DATABRICKS_INTEGRATION_CATALOG", "trading_tgp_dev",
    ).strip() or "trading_tgp_dev"
    INTEGRATION_SCHEMA: ClassVar[str] = os.environ.get(
        "DATABRICKS_INTEGRATION_SCHEMA", "ygg_integration",
    ).strip() or "ygg_integration"
    #: Deletion guard: teardown refuses to drop any schema whose name
    #: doesn't start with this, and never drops a catalog.
    SCHEMA_PREFIX: ClassVar[str] = "ygg_integration"

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

    # ------------------------------------------------------------------
    # Shared fixture provisioning — one persistent home + safe teardown
    # ------------------------------------------------------------------
    @classmethod
    def integration_schema(cls):
        """The shared, persistent ``ygg_integration`` schema (created if
        missing, never dropped). Skips the class on a permission error."""
        from databricks.sdk.errors import DatabricksError
        from databricks.sdk.errors.platform import PermissionDenied

        sch = cls.client.schemas(
            catalog_name=cls.INTEGRATION_CATALOG,
        ).schema(schema_name=cls.INTEGRATION_SCHEMA)
        try:
            sch.ensure_created(comment="yggdrasil shared integration fixtures")
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"cannot create {cls.INTEGRATION_CATALOG}.{cls.INTEGRATION_SCHEMA}: "
                f"{exc}. Override DATABRICKS_INTEGRATION_CATALOG / _SCHEMA with a "
                f"catalog the test identity can CREATE SCHEMA on."
            ) from exc
        return sch

    @classmethod
    def integration_volume(cls, name: str = "ygg_integration"):
        """A shared managed volume under the integration schema (created if
        missing, never dropped) — the home for filesystem scratch dirs."""
        from databricks.sdk.errors import DatabricksError
        from databricks.sdk.errors.platform import PermissionDenied

        cls.integration_schema()
        vol = cls.client.volumes(
            catalog_name=cls.INTEGRATION_CATALOG,
            schema_name=cls.INTEGRATION_SCHEMA,
        ).volume(volume_name=name)
        try:
            vol.ensure_created()
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"cannot create volume {cls.INTEGRATION_CATALOG}."
                f"{cls.INTEGRATION_SCHEMA}.{name}: {exc}."
            ) from exc
        return vol

    @classmethod
    def scratch_schema(cls):
        """Mint a throw-away ``ygg_integration_<hex>`` schema for tests that
        need isolation (volume / external-volume lifecycle). Always
        prefixed so :meth:`safe_drop_schema` can clean it up. Skips on a
        permission error."""
        from databricks.sdk.errors import DatabricksError
        from databricks.sdk.errors.platform import PermissionDenied

        name = f"{cls.SCHEMA_PREFIX}_{secrets.token_hex(4)}"
        sch = cls.client.schemas(
            catalog_name=cls.INTEGRATION_CATALOG,
        ).schema(schema_name=name)
        try:
            sch.ensure_created(comment="yggdrasil throw-away integration schema")
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"cannot create scratch schema {cls.INTEGRATION_CATALOG}.{name}: "
                f"{exc}."
            ) from exc
        return sch

    @classmethod
    def safe_drop_schema(cls, schema) -> None:
        """Drop *schema* only when it's a throw-away ``ygg_integration``-
        prefixed one — and never the shared persistent schema, never a
        catalog. The guard against a test bug nuking real data."""
        if schema is None:
            return
        name = getattr(schema, "schema_name", "") or ""
        if not name.startswith(cls.SCHEMA_PREFIX) or name == cls.INTEGRATION_SCHEMA:
            logger.warning(
                "Refusing to drop schema %r — not an ygg_integration_<hex> "
                "throw-away schema.", name,
            )
            return
        try:
            schema.delete(force=True, raise_error=False)
        except Exception as exc:  # noqa: BLE001 — teardown is best-effort
            logger.warning("Failed to drop scratch schema %r: %r", name, exc)
