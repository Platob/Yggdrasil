"""``ygg`` — unified CLI entry point for yggdrasil.

Subcommands:

    ygg bot serve       Start the bot HTTP server
    ygg bot run         Call a @remote function on a bot server
    ygg genie           Launch the Genie conversational CLI
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ygg",
        description="Yggdrasil CLI — data tools, bot execution, and Databricks utilities.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command")

    # -- bot ---------------------------------------------------------------
    bot_parser = subparsers.add_parser("bot", help="Bot HTTP server and remote execution.")
    bot_sub = bot_parser.add_subparsers(dest="bot_action")

    serve = bot_sub.add_parser("serve", help="Start the bot HTTP server.")
    serve.add_argument("--host", default=None, help="Bind host (default: 127.0.0.1).")
    serve.add_argument("--port", type=int, default=None, help="Bind port (default: 8100).")
    serve.add_argument("--allow-remote", action="store_true", default=False, help="Allow non-local clients.")
    serve.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload.")
    serve.set_defaults(handler=_bot_serve)

    run = bot_sub.add_parser("run", help="Call a @remote function on a bot server.")
    run.add_argument("func", help="Function key (e.g. 'mymodule:my_func').")
    run.add_argument("args", nargs="*", default=[], help="Positional arguments.")
    run.add_argument("--url", default="http://127.0.0.1:8100", help="Bot server URL.")
    run.add_argument("--kwarg", action="append", default=[], metavar="KEY=VALUE", help="Keyword argument (repeatable).")
    run.add_argument("--timeout", type=float, default=600.0, help="Request timeout in seconds.")
    run.add_argument("--stream", action="store_true", default=False, help="Stream Arrow IPC batches.")
    run.set_defaults(handler=_bot_run)

    # -- genie -------------------------------------------------------------
    genie_parser = subparsers.add_parser(
        "genie",
        help="Launch the Genie conversational CLI.",
        add_help=False,
    )
    genie_parser.set_defaults(handler=_genie)

    return parser


def _bot_serve(args: argparse.Namespace) -> int:
    import os

    if args.host:
        os.environ["YGG_BOT_HOST"] = args.host
    if args.port:
        os.environ["YGG_BOT_PORT"] = str(args.port)
    if args.allow_remote:
        os.environ["YGG_BOT_ALLOW_REMOTE"] = "1"

    from yggdrasil.bot.config import get_settings
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "yggdrasil.bot.app:app",
        host=settings.host,
        port=settings.port,
        reload=args.reload,
    )
    return 0


def _bot_run(args: argparse.Namespace) -> int:
    from yggdrasil.bot.client import BotClient
    from yggdrasil.dataclasses.safe_function import check_function_args

    kwargs = {}
    for kv in args.kwarg:
        if "=" not in kv:
            print(f"Error: --kwarg must be KEY=VALUE, got {kv!r}", file=sys.stderr)
            return 1
        k, v = kv.split("=", 1)
        kwargs[k] = v

    client = BotClient(args.url, timeout=args.timeout)

    if args.stream:
        try:
            for batch in client.call_stream(args.func, *args.args, **kwargs):
                print(batch.to_pandas().to_string(index=False))
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        result = client.call(args.func, *args.args, **kwargs)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    import pyarrow as pa
    if isinstance(result, pa.Table):
        print(result.to_pandas().to_string(index=False))
    else:
        print(result)
    return 0


def _genie(args: argparse.Namespace) -> int:
    from yggdrasil.cli.databricks.genie import main as genie_main

    remaining = sys.argv[2:] if len(sys.argv) > 2 else []
    return genie_main(remaining)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()

    if argv is None:
        argv = sys.argv[1:]

    args, remaining = parser.parse_known_args(argv)

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    if args.command is None:
        parser.print_help()
        return 0

    handler = getattr(args, "handler", None)
    if handler is None:
        if args.command == "bot":
            parser.parse_args(["bot", "--help"])
        parser.print_help()
        return 0

    if args.command == "genie":
        return handler(args)

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
