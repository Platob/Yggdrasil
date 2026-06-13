"""``ygg node`` — serve the local YGG FastAPI node."""
from __future__ import annotations

import argparse
from typing import Sequence

from .config import Settings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ygg node", description="YGG node server.")
    sub = parser.add_subparsers(dest="action")

    serve = sub.add_parser("serve", help="Serve the node FastAPI app.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)
    serve.add_argument("--node-home", default=None)
    serve.add_argument("--allow-remote", action="store_true")
    serve.add_argument("--no-front", action="store_true", help="Accepted for compatibility; the node has no bundled frontend.")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.action != "serve":
        parser.print_help()
        return 0

    import uvicorn

    from .app import create_app

    settings = Settings(host=args.host, port=args.port, allow_remote=args.allow_remote)
    if args.node_home:
        settings.node_home = settings.node_home.__class__(args.node_home)
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
