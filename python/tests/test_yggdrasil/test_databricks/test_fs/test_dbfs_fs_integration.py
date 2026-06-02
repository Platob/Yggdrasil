"""Live :class:`DBFSPath` integration — the shared backend contract.

Provisions a per-run scratch directory under
:envvar:`DATABRICKS_INTEGRATION_DBFS_DIR` (default
``/dbfs/tmp/yggdrasil-integration``) and inherits the CRUD / remove /
open contract from :class:`FsRoundTripMixin`.
"""
from __future__ import annotations

import os
import secrets
import unittest

from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied

from yggdrasil.databricks.fs import DBFSPath

from ._base import FsIntegrationCase, FsRoundTripMixin


__all__ = ["TestDBFSRoundTrip"]


class TestDBFSRoundTrip(FsIntegrationCase, FsRoundTripMixin):
    ext = "bin"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        base = os.environ.get(
            "DATABRICKS_INTEGRATION_DBFS_DIR", "/dbfs/tmp/yggdrasil-integration",
        ).rstrip("/")
        cls.root = DBFSPath(f"{base}/run-{secrets.token_hex(4)}", client=cls.client)
        try:
            cls.root.mkdir(parents=True, exist_ok=True)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"cannot write to {base}: {exc}") from exc
