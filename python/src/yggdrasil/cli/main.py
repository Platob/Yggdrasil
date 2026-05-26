"""``ygg`` — unified CLI entry point for yggdrasil.

Subcommands::

    ygg node serve      Start node server + frontend (foreground)
    ygg node front      Start frontend only (Next.js dev server)
    ygg node install    Install node+frontend as boot service (systemd/launchd)
    ygg node uninstall  Remove boot service (--purge to delete data)
    ygg node run        Call a @remote function
    ygg node chat       Open YGGCHAT terminal
    ygg node status     Show running node status
    ygg node stop       Stop the background node
    ygg databricks      YGGDBKS Databricks management CLI
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _ensure_node_running() -> str:
    try:
        from yggdrasil.node.daemon import get_node_url, spawn_node
        spawn_node()
        return get_node_url()
    except Exception:
        return "http://127.0.0.1:8100"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ygg",
        description="Yggdrasil CLI — data tools, node execution, and Databricks utilities.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--no-node", action="store_true", help="Skip auto-spawning the background node.")

    sub = parser.add_subparsers(dest="command")

    # -- node --------------------------------------------------------------
    node = sub.add_parser("node", help="Yggdrasil node server and remote execution.")
    node_sub = node.add_subparsers(dest="node_action")

    serve = node_sub.add_parser("serve", help="Start node server + frontend (foreground).")
    serve.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0).")
    serve.add_argument("--port", type=int, default=None, help="Bind port (auto-scans if busy).")
    serve.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload.")
    serve.add_argument("--no-front", action="store_true", default=False, help="Skip launching the frontend.")
    serve.set_defaults(handler=_node_serve)

    front = node_sub.add_parser("front", help="Start frontend dev server only.")
    front.add_argument("--port", type=int, default=None, help="Frontend port (default: 3000).")
    front.add_argument("--node-port", type=int, default=None, help="Node API port to proxy to.")
    front.set_defaults(handler=_node_front)

    run = node_sub.add_parser("run", help="Call a @remote function.")
    run.add_argument("func", help="Function key (e.g. 'mymodule:my_func').")
    run.add_argument("args", nargs="*", default=[], help="Positional arguments.")
    run.add_argument("--url", default=None, help="Node server URL (default: auto).")
    run.add_argument("--kwarg", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--stream", action="store_true", default=False)
    run.set_defaults(handler=_node_run)

    chat = node_sub.add_parser("chat", help="Open YGGCHAT terminal.")
    chat.add_argument("--url", default=None, help="Node server URL (default: auto).")
    chat.add_argument("--user", default=None, help="Display name.")
    chat.add_argument("--channel", default="general", help="Initial channel.")
    chat.set_defaults(handler=_node_chat)

    install = node_sub.add_parser("install", help="Install node as a boot service (systemd/launchd).")
    install.add_argument("--no-front", action="store_true", default=False, help="Skip frontend service.")
    install.add_argument("--port", type=int, default=None, help="Override node port.")
    install.add_argument("--front-port", type=int, default=None, help="Override frontend port.")
    install.set_defaults(handler=_node_install)

    uninstall = node_sub.add_parser("uninstall", help="Remove node boot service and stop.")
    uninstall.add_argument("--purge", action="store_true", default=False, help="Also remove all data in ~/.ygg.")
    uninstall.set_defaults(handler=_node_uninstall)

    status = node_sub.add_parser("status", help="Show node status.")
    status.set_defaults(handler=_node_status)

    stop = node_sub.add_parser("stop", help="Stop the node.")
    stop.set_defaults(handler=_node_stop)

    # -- databricks --------------------------------------------------------
    dbks = sub.add_parser("databricks", help="YGGDBKS Databricks management.", add_help=False)
    dbks.set_defaults(handler=_databricks)

    return parser


def _start_frontend(settings, *, node_port: int, front_port: int | None = None) -> "subprocess.Popen | None":
    import os
    import shutil
    import subprocess

    front_home = settings.front_home
    if not (front_home / "package.json").exists():
        from yggdrasil.cli.style import dim, out, yellow
        out(f"  {yellow('skip')}  frontend not found at {dim(str(front_home))}\n")
        return None

    npm = shutil.which("npm")
    if npm is None:
        from yggdrasil.cli.style import dim, out, yellow
        out(f"  {yellow('skip')}  npm not found — install Node.js to serve the frontend\n")
        return None

    if not (front_home / "node_modules").exists():
        from yggdrasil.cli.style import Spinner
        with Spinner("installing frontend dependencies...", color="33") as sp:
            subprocess.run(
                [npm, "install"],
                cwd=str(front_home),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            sp.stop()

    from yggdrasil.node.config import _find_open_port
    port = front_port or _find_open_port(settings.front_port, settings.front_port + 100)

    env = {**os.environ, "YGG_NODE_PORT": str(node_port), "PORT": str(port)}
    proc = subprocess.Popen(
        [npm, "run", "dev", "--", "--hostname", "0.0.0.0", "--port", str(port)],
        cwd=str(front_home),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    from yggdrasil.cli.style import bold, cyan, out
    out(f"  {cyan('front')} {bold(f'http://localhost:{port}')}\n")
    return proc


def _node_serve(args: argparse.Namespace) -> int:
    import os
    from yggdrasil.cli.style import print_logo

    print_logo("YGGNODE")

    if args.host:
        os.environ["YGG_NODE_HOST"] = args.host
    if args.port:
        os.environ["YGG_NODE_PORT"] = str(args.port)

    from yggdrasil.node.config import _find_open_port, get_settings
    from yggdrasil.node.daemon import cleanup_old_logs, ensure_directories

    settings = get_settings()
    ensure_directories(settings)
    cleanup_old_logs(settings)

    port = args.port or _find_open_port(settings.port, settings.port + 100)
    host = args.host or settings.host

    from yggdrasil.cli.style import bold, cyan, dim, out
    out(f"  {cyan('node')}  {bold(settings.node_id)}\n")
    out(f"  {cyan('home')}  {dim(str(settings.node_home))}\n")
    out(f"  {cyan('bind')}  {bold(f'{host}:{port}')}\n")

    front_proc = None
    if not args.no_front:
        front_proc = _start_frontend(settings, node_port=port)
    out("\n")

    import uvicorn
    try:
        uvicorn.run("yggdrasil.node.app:app", host=host, port=port, reload=args.reload)
    finally:
        if front_proc is not None:
            front_proc.terminate()
            front_proc.wait(timeout=5)
    return 0


def _node_front(args: argparse.Namespace) -> int:
    import signal

    from yggdrasil.node.config import get_settings
    from yggdrasil.cli.style import print_logo

    print_logo("YGGNODE")
    settings = get_settings()

    node_port = args.node_port or settings.port
    proc = _start_frontend(settings, node_port=node_port, front_port=args.port)
    if proc is None:
        return 1

    from yggdrasil.cli.style import dim, out
    out(f"  {dim('Press Ctrl+C to stop.')}\n\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    return 0


def _node_run(args: argparse.Namespace) -> int:
    from yggdrasil.node.client import NodeClient
    from yggdrasil.cli.style import Spinner

    url = args.url or _ensure_node_running()
    kwargs = {}
    for kv in args.kwarg:
        if "=" not in kv:
            print(f"Error: --kwarg must be KEY=VALUE, got {kv!r}", file=sys.stderr)
            return 1
        k, v = kv.split("=", 1)
        kwargs[k] = v

    client = NodeClient(url, timeout=args.timeout)

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


def _node_chat(args: argparse.Namespace) -> int:
    from yggdrasil.node.chat import run_chat
    url = args.url or _ensure_node_running()
    return run_chat(url=url, username=args.user, channel=args.channel)


def _node_install(args: argparse.Namespace) -> int:
    import os
    from yggdrasil.cli.style import bold, cyan, dim, green, out, print_logo, red, yellow
    from yggdrasil.node.config import get_settings
    from yggdrasil.node.service import install_service, service_status

    print_logo("YGGNODE")

    if args.port:
        os.environ["YGG_NODE_PORT"] = str(args.port)
    if args.front_port:
        os.environ["YGG_NODE_FRONT_PORT"] = str(args.front_port)

    settings = get_settings()

    out(f"  {cyan('install')} registering boot service...\n")
    out(f"  {cyan('node')}    port {bold(str(settings.port))}\n")
    if not args.no_front:
        out(f"  {cyan('front')}   port {bold(str(settings.front_port))}\n")
    out(f"  {cyan('home')}    {dim(str(settings.node_home))}\n\n")

    ok, msg = install_service(settings, no_front=args.no_front)

    if ok:
        out(f"  {green('✓')} {msg}\n\n")
        status = service_status(settings)
        for name, state in status.items():
            color = green if state == "active" or state == "running" else yellow
            out(f"  {cyan(name)}  {color(state)}\n")
        out(f"\n  Node will start automatically on boot.\n")
        out(f"  Uninstall with: {bold('ygg node uninstall')}\n")
    else:
        out(f"  {red('✗')} {msg}\n")
        return 1
    return 0


def _node_uninstall(args: argparse.Namespace) -> int:
    from yggdrasil.cli.style import bold, cyan, dim, green, out, print_logo, red, yellow
    from yggdrasil.node.config import get_settings
    from yggdrasil.node.service import uninstall_service

    print_logo("YGGNODE")

    settings = get_settings()

    if args.purge:
        out(f"  {yellow('purge')} will remove all data in {dim(str(settings.node_home))}\n\n")

    ok, msg = uninstall_service(settings, purge=args.purge)

    if ok:
        out(f"  {green('✓')} {msg}\n")
        out(f"\n  Node service removed. Will no longer start on boot.\n")
        if args.purge:
            out(f"  {dim('All node data has been removed.')}\n")
    else:
        out(f"  {red('✗')} {msg}\n")
        return 1
    return 0


def _node_status(args: argparse.Namespace) -> int:
    from yggdrasil.node.config import get_settings
    from yggdrasil.node.daemon import _is_node_running, ensure_directories
    from yggdrasil.node.service import service_status
    from yggdrasil.cli.style import bold, cyan, dim, green, out, print_logo, red, yellow

    print_logo("YGGNODE")
    settings = get_settings()
    ensure_directories(settings)
    running, pid, port = _is_node_running(settings)

    out(f"  {cyan('home')}    {dim(str(settings.node_home))}\n")
    out(f"  {cyan('logs')}    {dim(str(settings.logs_root))}\n")
    out(f"  {cyan('cache')}   {dim(str(settings.cache_root))}\n")
    if running:
        out(f"  {cyan('status')}  {green('running')} {dim(f'(pid={pid}, port={port})')}\n")
        out(f"  {cyan('url')}     {bold(f'http://127.0.0.1:{port}')}\n")
    else:
        out(f"  {cyan('status')}  {red('stopped')}\n")

    svc = service_status(settings)
    if svc:
        out(f"\n  {cyan('boot services:')}\n")
        for name, state in svc.items():
            color = green if state in ("active", "running") else (red if state == "not installed" else yellow)
            out(f"    {dim(name)}  {color(state)}\n")

    return 0


def _node_stop(args: argparse.Namespace) -> int:
    from yggdrasil.node.daemon import stop_node
    from yggdrasil.cli.style import green, out, print_logo, red

    print_logo("YGGNODE")
    if stop_node():
        out(f"  {green('node stopped.')}\n")
    else:
        out(f"  {red('no running node found.')}\n")
    return 0


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

    if not getattr(args, "no_node", False) and args.command not in ("node",):
        _ensure_node_running()

    handler = getattr(args, "handler", None)
    if handler is None:
        if args.command == "node":
            parser.parse_args(["node", "--help"])
        parser.print_help()
        return 0

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
