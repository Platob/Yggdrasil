"""``ygg node`` — serve and probe the trading node.

``serve`` boots uvicorn on the configured host/port (``--dev`` enables
reload + permissive CORS); ``status`` pings a running node and reports its
identity/version. Kept deliberately thin — all behavior lives in the app
factory and services.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import Settings

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ygg node", description="Yggdrasil trading node.")
    sub = parser.add_subparsers(dest="action", required=True)

    serve = sub.add_parser("serve", help="Start the node server.")
    serve.add_argument("--host", default=None, help="Bind host (default 127.0.0.1).")
    serve.add_argument("--port", type=int, default=None, help="Bind port (default 8765).")
    serve.add_argument("--home", default=None, help="Node data home directory.")
    serve.add_argument("--dev", action="store_true", help="Dev mode: reload + open CORS.")
    serve.set_defaults(handler=_serve)

    status = sub.add_parser("status", help="Check whether a node is running.")
    status.add_argument("--host", default=None)
    status.add_argument("--port", type=int, default=None)
    status.set_defaults(handler=_status)

    return parser


def _settings_from(args: argparse.Namespace) -> Settings:
    overrides: dict = {}
    if getattr(args, "host", None):
        overrides["host"] = args.host
    if getattr(args, "port", None):
        overrides["port"] = args.port
    if getattr(args, "home", None):
        overrides["node_home"] = Path(args.home).expanduser()
    return Settings(**overrides)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = _settings_from(args)
    if args.dev:
        settings.cors_origins = ["*"]

    from yggdrasil.cli import style

    style.info(f"node {settings.node_id!r} serving on http://{settings.host}:{settings.port}")
    style.step(f"data home: {settings.node_home}")

    # reload needs an import string; otherwise hand uvicorn the live app.
    if args.dev:
        import os

        os.environ["YGG_NODE_HOME"] = str(settings.node_home)
        uvicorn.run(
            "yggdrasil.node.app:create_api",
            factory=True,
            host=settings.host,
            port=settings.port,
            reload=True,
        )
    else:
        from .app import create_api

        uvicorn.run(create_api(settings), host=settings.host, port=settings.port)
    return 0


def _status(args: argparse.Namespace) -> int:
    import httpx

    from yggdrasil.cli import style

    settings = _settings_from(args)
    url = f"http://{settings.host}:{settings.port}/api/ping"
    try:
        resp = httpx.get(url, timeout=2.0)
    except httpx.HTTPError as exc:
        style.fail(f"node not reachable at {url}: {exc}")
        return 1
    if resp.status_code != 200:
        style.fail(f"node returned {resp.status_code} from {url}")
        return 1
    body = resp.json()
    style.ok(f"node {body.get('node_id')!r} up (ts={body.get('ts')})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
