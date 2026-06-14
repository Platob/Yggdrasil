"""``ygg node`` — serve and inspect the FastAPI node backend.

::

    ygg node serve [--host H] [--port P] [--allow-remote] [--reload]
    ygg node status [--url URL]
    ygg node open [--url URL]
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed — run: pip install 'ygg[node]'", file=sys.stderr)
        return 1

    from pathlib import Path

    from yggdrasil.node.app import create_app
    from yggdrasil.node.config import Settings

    kwargs: dict = {}
    if getattr(args, "home", None):
        kwargs["node_home"] = Path(args.home)
        kwargs["front_home"] = Path(args.home)

    settings = Settings(allow_remote=args.allow_remote, **kwargs)
    print(f"  Ygg Node  →  http://{args.host}:{args.port}")
    print(f"  Home:        {settings.node_home}")
    print(f"  API docs:    http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        reload=getattr(args, "reload", False),
        log_level="info",
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    import json
    import urllib.request

    url = getattr(args, "url", "http://127.0.0.1:8100").rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/api/ping", timeout=3) as resp:
            data = json.loads(resp.read())
            print(f"  ● Node online at {url}")
            print(f"    status={data.get('status')}  ts={data.get('ts'):.2f}")
            return 0
    except Exception as exc:
        print(f"  ○ Node offline at {url}  ({exc})")

    from yggdrasil.node.config import Settings
    settings = Settings()
    print(f"  node_id:    {settings.node_id}")
    print(f"  node_home:  {settings.node_home}")
    return 1


def _open_dashboard(args: argparse.Namespace) -> int:
    import webbrowser
    url = getattr(args, "url", "http://127.0.0.1:3100")
    webbrowser.open(url)
    print(f"  Opening {url}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ygg node", description="Ygg Node backend.")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the FastAPI node server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)
    serve.add_argument("--allow-remote", action="store_true")
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--home", default=None, help="Override node home directory.")
    serve.set_defaults(handler=_serve)

    status = sub.add_parser("status", help="Ping node and print configuration.")
    status.add_argument("--url", default="http://127.0.0.1:8100")
    status.set_defaults(handler=_status)

    open_cmd = sub.add_parser("open", help="Open the trading dashboard in a browser.")
    open_cmd.add_argument("--url", default="http://127.0.0.1:3100")
    open_cmd.set_defaults(handler=_open_dashboard)

    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)
