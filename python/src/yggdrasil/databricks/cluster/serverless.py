"""Serverless Databricks cluster wrapper.

Subclasses :class:`Cluster` to encode the bits of cluster operation
that differ for serverless compute:

- distinct URL scheme (``dbks+serverless-cluster://``) so a
  :class:`URLBased` round-trip preserves the serverless flavor.
- :meth:`start` / :meth:`restart` are no-ops — serverless compute is
  always-on from the caller's perspective; the SDK lifecycle calls
  don't apply.
- :meth:`install_libraries` is rejected — serverless workloads pin
  dependencies through environment specs and the workspace-level
  serverless config, not through the per-cluster libraries API.

Everything else (details fetch, execution-context creation, command
execution, deletion) is inherited unchanged. The class is light by
design: it captures the runtime divergence in one place so callers
that just need "a Cluster handle" still get the same surface, and
the executor layer (:class:`ClusterStatementExecutor`) doesn't have
to special-case serverless.
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional, Union

from databricks.sdk.service.compute import (
    ClusterAccessControlRequest,
    Library,
)

from yggdrasil.data.enums import Scheme
from yggdrasil.dataclasses.waiting import WaitingConfigArg

from .cluster import Cluster

__all__ = ["ServerlessCluster"]

LOGGER = logging.getLogger(__name__)


class ServerlessCluster(Cluster):
    """Cluster handle for Databricks serverless compute."""

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_SERVERLESS_CLUSTER

    # ------------------------------------------------------------------ #
    # Lifecycle — serverless has no caller-visible start / restart hook.
    # ------------------------------------------------------------------ #
    def start(self, wait: WaitingConfigArg = True) -> "ServerlessCluster":
        """No-op for serverless clusters.

        Serverless compute is provisioned on demand by the workspace —
        there's no caller-driven start phase. Returns ``self`` so the
        method composes cleanly with code paths that call ``.start()``
        unconditionally.
        """
        LOGGER.debug(
            "Skipping start on serverless cluster %r (serverless compute "
            "is always-on from the caller's perspective)", self,
        )
        return self

    def restart(self, wait: WaitingConfigArg = True) -> "ServerlessCluster":
        """No-op for serverless clusters — see :meth:`start`."""
        LOGGER.debug(
            "Skipping restart on serverless cluster %r (serverless "
            "compute does not expose a restart cycle)", self,
        )
        return self

    # ------------------------------------------------------------------ #
    # Library install — not supported on serverless.
    # ------------------------------------------------------------------ #
    def install_libraries(
        self,
        libraries: Optional[list[Union[str, Library]]] = None,
        wait: WaitingConfigArg = True,
        pip_settings: Any = None,
        remove_failed: bool = True,
        raise_error: bool = True,
    ) -> "ServerlessCluster":
        """Reject library installs on serverless clusters.

        Serverless workloads pin dependencies via environment specs,
        not via the per-cluster libraries API. Silently no-op'ing
        would hide a real configuration mismatch; raise instead so
        the caller knows to wire the dependency through the right
        channel.
        """
        if not libraries:
            return self
        raise NotImplementedError(
            f"{type(self).__name__}.install_libraries is not supported — "
            "pin serverless dependencies via the workspace's serverless "
            "environment spec instead of the per-cluster libraries API."
        )

    # ------------------------------------------------------------------ #
    # Permissions — inherits Cluster.update_permissions unchanged; the
    # Cluster ACL endpoint accepts serverless cluster ids the same way.
    # ------------------------------------------------------------------ #
    def update_permissions(
        self,
        permissions: Optional[list[str | ClusterAccessControlRequest]] = None,
    ) -> "ServerlessCluster":
        return super().update_permissions(permissions=permissions)  # type: ignore[return-value]
