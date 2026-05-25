"""``ygg`` — unified CLI entry point for yggdrasil.

Subcommands::

    ygg bot serve       Start YGGBOT server (foreground)
    ygg bot run         Call a @remote function
    ygg bot chat        Open YGGCHAT terminal
    ygg bot status      Show running bot status
    ygg bot stop        Stop the background bot
    ygg genie           Launch YGGGENIE conversational CLI
    ygg databricks      YGGDBKS Databricks management CLI
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _ensure_bot_running() -> str:
    try:
        from yggdrasil.bot.daemon import get_bot_url, spawn_bot
        spawn_bot()
        return get_bot_url()
    except Exception:
        return "http://127.0.0.1:8100"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ygg",
        description="Yggdrasil CLI — data tools, bot execution, and Databricks utilities.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--no-bot", action="store_true", help="Skip auto-spawning the background bot.")

    sub = parser.add_subparsers(dest="command")

    # -- bot ---------------------------------------------------------------
    bot = sub.add_parser("bot", help="YGGBOT server and remote execution.")
    bot_sub = bot.add_subparsers(dest="bot_action")

    serve = bot_sub.add_parser("serve", help="Start YGGBOT server (foreground).")
    serve.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0).")
    serve.add_argument("--port", type=int, default=None, help="Bind port (auto-scans if busy).")
    serve.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload.")
    serve.set_defaults(handler=_bot_serve)

    run = bot_sub.add_parser("run", help="Call a @remote function.")
    run.add_argument("func", help="Function key (e.g. 'mymodule:my_func').")
    run.add_argument("args", nargs="*", default=[], help="Positional arguments.")
    run.add_argument("--url", default=None, help="Bot server URL (default: auto).")
    run.add_argument("--kwarg", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--stream", action="store_true", default=False)
    run.set_defaults(handler=_bot_run)

    chat = bot_sub.add_parser("chat", help="Open YGGCHAT terminal.")
    chat.add_argument("--url", default=None, help="Bot server URL (default: auto).")
    chat.add_argument("--user", default=None, help="Display name.")
    chat.add_argument("--channel", default="general", help="Initial channel.")
    chat.set_defaults(handler=_bot_chat)

    status = bot_sub.add_parser("status", help="Show YGGBOT status.")
    status.set_defaults(handler=_bot_status)

    stop = bot_sub.add_parser("stop", help="Stop YGGBOT.")
    stop.set_defaults(handler=_bot_stop)

    # -- genie -------------------------------------------------------------
    genie = sub.add_parser("genie", help="Launch YGGGENIE.", add_help=False)
    genie.set_defaults(handler=_genie)

    # -- databricks --------------------------------------------------------
    dbks = sub.add_parser("databricks", help="YGGDBKS Databricks management.", add_help=False)
    dbks.set_defaults(handler=_databricks)

    return parser


def _bot_serve(args: argparse.Namespace) -> int:
    import os
    from yggdrasil.cli.style import Spinner, print_logo

    print_logo("YGGBOT")

    if args.host:
        os.environ["YGG_BOT_HOST"] = args.host
    if args.port:
        os.environ["YGG_BOT_PORT"] = str(args.port)

    from yggdrasil.bot.config import _find_open_port, get_settings
    from yggdrasil.bot.daemon import cleanup_old_logs, ensure_directories

    settings = get_settings()
    ensure_directories(settings)
    cleanup_old_logs(settings)

    port = args.port or _find_open_port(settings.port, settings.port + 100)
    host = args.host or settings.host

    from yggdrasil.cli.style import bold, cyan, dim, out
    out(f"  {cyan('node')}  {bold(settings.node_id)}\n")
    out(f"  {cyan('home')}  {dim(str(settings.bot_home))}\n")
    out(f"  {cyan('bind')}  {bold(f'{host}:{port}')}\n\n")

    import uvicorn
    uvicorn.run("yggdrasil.bot.app:app", host=host, port=port, reload=args.reload)
    return 0


def _bot_run(args: argparse.Namespace) -> int:
    from yggdrasil.bot.client import BotClient
    from yggdrasil.cli.style import Spinner

    url = args.url or _ensure_bot_running()
    kwargs = {}
    for kv in args.kwarg:
        if "=" not in kv:
            print(f"Error: --kwarg must be KEY=VALUE, got {kv!r}", file=sys.stderr)
            return 1
        k, v = kv.split("=", 1)
        kwargs[k] = v

    client = BotClient(url, timeout=args.timeout)

    if args.stream:
        try:
            for batch in client.call_stream(args.func, *args.args, **kwargs):
                print(batch.to_pandas().to_string(index=False))
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    with Spinner(f"calling {args.func}...", color="33"):
        try:
            result = client.call(args.func, *args.args, **kwargs)
        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1

    import pyarrow as pa
    if isinstance(result, pa.Table):
        print(result.to_pandas().to_string(index=False))
    else:
        print(result)
    return 0


def _bot_chat(args: argparse.Namespace) -> int:
    from yggdrasil.bot.chat import run_chat
    url = args.url or _ensure_bot_running()
    return run_chat(url=url, username=args.user, channel=args.channel)


def _bot_status(args: argparse.Namespace) -> int:
    from yggdrasil.bot.config import get_settings
    from yggdrasil.bot.daemon import _is_bot_running, ensure_directories
    from yggdrasil.cli.style import bold, cyan, dim, green, out, print_logo, red

    print_logo("YGGBOT")
    settings = get_settings()
    ensure_directories(settings)
    running, pid, port = _is_bot_running(settings)

    out(f"  {cyan('home')}    {dim(str(settings.bot_home))}\n")
    out(f"  {cyan('logs')}    {dim(str(settings.logs_root))}\n")
    out(f"  {cyan('cache')}   {dim(str(settings.cache_root))}\n")
    if running:
        out(f"  {cyan('status')}  {green(f'running')} {dim(f'(pid={pid}, port={port})')}\n")
        out(f"  {cyan('url')}     {bold(f'http://127.0.0.1:{port}')}\n")
    else:
        out(f"  {cyan('status')}  {red('stopped')}\n")
    return 0


def _bot_stop(args: argparse.Namespace) -> int:
    from yggdrasil.bot.daemon import stop_bot
    from yggdrasil.cli.style import green, out, print_logo, red

    print_logo("YGGBOT")
    if stop_bot():
        out(f"  {green('bot stopped.')}\n")
    else:
        out(f"  {red('no running bot found.')}\n")
    return 0


def _genie(args: argparse.Namespace) -> int:
    from yggdrasil.cli.databricks.genie import main as genie_main
    remaining = sys.argv[2:] if len(sys.argv) > 2 else []
    return genie_main(remaining)


def _databricks(args: argparse.Namespace) -> int:
    from yggdrasil.databricks.cli import main as dbks_main
    remaining = sys.argv[2:] if len(sys.argv) > 2 else []
    return dbks_main(remaining)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    if argv is None:
        argv = sys.argv[1:]

    args, remaining = parser.parse_known_args(argv)

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    if args.command is None:
        from yggdrasil.cli.style import print_logo
        print_logo("YGG")
        parser.print_help()
        return 0

    if not getattr(args, "no_bot", False) and args.command not in ("bot",):
        _ensure_bot_running()

    handler = getattr(args, "handler", None)
    if handler is None:
        if args.command == "bot":
            parser.parse_args(["bot", "--help"])
        parser.print_help()
        return 0

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
