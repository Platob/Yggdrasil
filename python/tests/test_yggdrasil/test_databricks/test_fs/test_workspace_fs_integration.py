"""Live :class:`WorkspacePath` integration — the shared backend contract.

Provisions a per-run scratch directory under
:envvar:`DATABRICKS_INTEGRATION_WORKSPACE_DIR` (default:
``/Workspace/Users/<current-user>/yggdrasil-integration``) and inherits
the CRUD / remove / open contract from :class:`FsRoundTripMixin`. The
Workspace API doesn't report a byte size for every object, so
``checks_size`` is off.
"""
from __future__ import annotations

import os
import secrets
import unittest

from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied

from yggdrasil.databricks.fs import WorkspacePath

from ._base import FsIntegrationCase, FsRoundTripMixin


__all__ = ["TestWorkspaceRoundTrip"]


class TestWorkspaceRoundTrip(FsIntegrationCase, FsRoundTripMixin):
    ext = "txt"
    checks_size = False

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        base = os.environ.get("DATABRICKS_INTEGRATION_WORKSPACE_DIR", "").strip()
        if not base:
            user = cls.workspace.current_user.me().user_name
            base = f"/Workspace/Users/{user}/yggdrasil-integration"
        cls.root = WorkspacePath(
            f"{base.rstrip('/')}/run-{secrets.token_hex(4)}", client=cls.client,
        )
        try:
            cls.root.mkdir(parents=True, exist_ok=True)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"cannot write to {base}: {exc}") from exc
