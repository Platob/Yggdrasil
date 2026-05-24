"""Databricks CLI — ``ygg-databricks`` console script.

Mirrors the official ``databricks`` CLI surface but routes every
operation through the yggdrasil service layer
(:class:`~yggdrasil.databricks.client.DatabricksClient`,
:class:`~yggdrasil.databricks.jobs.service.Jobs`,
:class:`~yggdrasil.databricks.fs.workspace_path.WorkspacePath`, …).

Currently supported:

- ``ygg-databricks bundle deploy [-t <target>]`` — parse a
  ``databricks.yml`` bundle, sync workspace files, and upsert jobs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from .bundle import BundleCommand

__all__ = ["main"]

LOGGER = logging.getLogger(__name__)


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

    subparsers = parser.add_subparsers(dest="command")
    BundleCommand.register(subparsers)

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
        return handler(args)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    except Exception as exc:
        sys.stderr.write(f"ygg-databricks: {exc}\n")
        LOGGER.debug("Full traceback:", exc_info=True)
        return 1
