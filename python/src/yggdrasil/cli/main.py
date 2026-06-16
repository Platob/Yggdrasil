"""``ygg`` — unified CLI entry point for yggdrasil.

Subcommands::

    ygg databricks      YGGDBKS Databricks management CLI
    ygg loki            Loki — the global yggdrasil agent (status/run/token)

``ygg`` is the single console-script entry point: deployed Databricks
python-wheel tasks (Auto Loader via ``ygg databricks table autoload``, the Loki
agent) all invoke ``ygg`` with a leading subcommand rather than a per-feature
script.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ygg",
        description="Yggdrasil CLI.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    sub = parser.add_subparsers(dest="command")

    # -- databricks --------------------------------------------------------
    dbks = sub.add_parser("databricks", help="YGGDBKS Databricks management.", add_help=False)
    dbks.set_defaults(handler=_databricks)

    # -- loki --------------------------------------------------------------
    loki = sub.add_parser("loki", help="Loki — the global yggdrasil agent.", add_help=False)
    loki.set_defaults(handler=_loki)

    # -- node --------------------------------------------------------------
    node = sub.add_parser("node", help="YGG node server (FastAPI backend + trading).", add_help=False)
    node.set_defaults(handler=_node)

    return parser


def _databricks(args: argparse.Namespace) -> int:
    from yggdrasil.databricks.cli import main as dbks_main
    remaining = sys.argv[2:] if len(sys.argv) > 2 else []
    return dbks_main(remaining)


def _loki(args: argparse.Namespace) -> int:
    from yggdrasil.loki.cli import main as loki_main
    remaining = sys.argv[2:] if len(sys.argv) > 2 else []
    return loki_main(remaining)


def _node(args: argparse.Namespace) -> int:
    from yggdrasil.node.cli import main as node_main
    remaining = sys.argv[2:] if len(sys.argv) > 2 else []
    return node_main(remaining)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    if argv is None:
        argv = sys.argv[1:]

    args, remaining = parser.parse_known_args(argv)

    if args.debug:
        import logging
        from yggdrasil.cli.style import install_logging
        install_logging(logging.DEBUG, force=True)

    if args.command is None:
        from yggdrasil.cli.style import print_logo
        print_logo("YGG")
        parser.print_help()
        return 0

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
