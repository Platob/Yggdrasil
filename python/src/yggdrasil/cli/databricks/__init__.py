"""Databricks CLIs.

Sub-services each ship a small CLI subclass under
:mod:`yggdrasil.cli.databricks` — all of them share the
:class:`DatabricksCLI` base which owns the workspace-client flag group
(``--host`` / ``--token`` / ``--profile`` / etc.) and the
:class:`DatabricksClient` construction handshake.

Currently registered:

- :class:`yggdrasil.cli.databricks.genie.GenieCLI` — conversational
  Genie agent CLI (``ygg-genie``).

Add a new sub-service CLI by subclassing :class:`DatabricksCLI`,
overriding :meth:`DatabricksCLI.add_service_arguments` with any
service-specific flags, and implementing :meth:`DatabricksCLI.run`.
"""

from .base import DatabricksCLI

__all__ = ["DatabricksCLI"]
