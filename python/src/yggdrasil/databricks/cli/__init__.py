"""YGGDBKS — Databricks CLI powered by yggdrasil services.

Subcommands::

    ygg databricks clusters list/get/create/delete
    ygg databricks warehouses list/get/create/delete
    ygg databricks table async_insert --table-name … --data …
    ygg databricks table execute_insert --logs … | --log-file …
    ygg databricks genie spaces
    ygg databricks genie ask "top customers by revenue" --space <id>
    ygg databricks genie agent "why did Q3 revenue dip?" --space <id>
    ygg databricks genie repl --space <id>
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Optional, Sequence

from .services import (
    ClustersCommand,
    GenieCommand,
    TablesCommand,
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
    parser = argparse.ArgumentParser(
        prog="ygg databricks",
        description="YGGDBKS — Databricks CLI powered by yggdrasil services.",
    )
    parser.add_argument("--debug", action="store_true", help="Set logger to DEBUG.")

    client_grp = parser.add_argument_group("Databricks client")
    for flag, dest, kwargs in _CLIENT_FLAGS:
        client_grp.add_argument(flag, dest=dest, default=None, **kwargs)

    subparsers = parser.add_subparsers(dest="command")
    ClustersCommand.register(subparsers)
    GenieCommand.register(subparsers)
    TablesCommand.register(subparsers)
    WarehousesCommand.register(subparsers)

    args = parser.parse_args(argv)

    if getattr(args, "debug", False):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("yggdrasil").setLevel(logging.DEBUG)

    try:
        from yggdrasil.cli.style import force_color, print_logo
        # This CLI's output lands in ANSI-rendering surfaces — a terminal or a
        # Databricks job / notebook panel — so paint color even off a TTY
        # (NO_COLOR still opts out).
        force_color()
        print_logo("YGGDBKS")
    except Exception:
        pass

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
        sys.stderr.write(f"ygg databricks: {exc}\n")
        LOGGER.debug("Full traceback:", exc_info=True)
        return 1
