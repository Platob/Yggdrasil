"""DBFS service — the :class:`DatabricksService` backing :class:`DBFSPath`.

Mirrors the :class:`Workspaces` / :class:`Volumes` shape: a thin
collection-level handle that carries a :class:`DatabricksClient` and
acts as the default ``service`` for every :class:`DBFSPath` instance.
There is no DBFS catalog hierarchy (DBFS is a flat namespace), so
this service is intentionally minimal — its job is to be the typed
slot that :meth:`DatabricksPath.client` / :meth:`DatabricksPath.sql`
read through, not to expose extra navigation.
"""

from __future__ import annotations

from ..service import DatabricksService


__all__ = ["DBFSService"]


class DBFSService(DatabricksService):
    """Collection-level service handle for the DBFS namespace."""
