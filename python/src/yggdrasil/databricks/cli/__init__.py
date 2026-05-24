"""Databricks CLI — ``ygg-databricks`` console script.

Mirrors the official ``databricks`` CLI surface but routes every
operation through the yggdrasil service layer
(:class:`~yggdrasil.databricks.client.DatabricksClient`,
:class:`~yggdrasil.databricks.jobs.service.Jobs`,
:class:`~yggdrasil.databricks.fs.workspace_path.WorkspacePath`, …).

Currently supported:

- ``ygg-databricks bundle deploy [-t <target>]``
- ``ygg-databricks jobs list/get/create/delete/run``
- ``ygg-databricks clusters list/get/create/delete``
- ``ygg-databricks warehouses list/get/create/delete``
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Optional, Sequence

from .bundle import BundleCommand
from .services import (
    ClustersCommand,
    JobsCommand,
    WarehousesCommand,
)

__all__ = ["main"]

LOGGER = logging.getLogger(__name__)

_CLIENT_FLAGS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("--host", "host", {"help": "Workspace URL (env: DATABRICKS_HOST)"}),
    ("--token", "token", {"help": "Personal access token (env: DATABRICKS_TOKEN)"}),
    ("--profile", "profile", {"help": "Profile in ~/.databrickscfg"}),
)


def _build_client(args: argparse.Namespace) -> "DatabricksClient":
    from yggdrasil.databricks.client import DatabricksClient

    kwargs: dict[str, Any] = {}
    for _flag, dest, _meta in _CLIENT_FLAGS:
        val = getattr(args, dest, None)
        if val is not None:
            kwargs[dest] = val
    return DatabricksClient(**kwargs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``ygg-databricks`` console script."""
    parser = argparse.ArgumentParser(
        prog="ygg-databricks",
        description="Databricks CLI powered by yggdrasil services.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Set yggdrasil logger to DEBUG.",
    )

    client_grp = parser.add_argument_group("Databricks client")
    for flag, dest, kwargs in _CLIENT_FLAGS:
        client_grp.add_argument(flag, dest=dest, default=None, **kwargs)

    subparsers = parser.add_subparsers(dest="command")
    BundleCommand.register(subparsers)
    JobsCommand.register(subparsers)
    ClustersCommand.register(subparsers)
    WarehousesCommand.register(subparsers)

    args = parser.parse_args(argv)

    if getattr(args, "debug", False):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("yggdrasil").setLevel(logging.DEBUG)

    if args.command is None:
        parser.print_help()
        return 1

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args, _build_client)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    except Exception as exc:
        sys.stderr.write(f"ygg-databricks: {exc}\n")
        LOGGER.debug("Full traceback:", exc_info=True)
        return 1
