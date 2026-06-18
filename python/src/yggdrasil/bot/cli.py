"""``ygg bot`` — bot server CLI (serve / status / bench)."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ygg bot", description="YGG Bot CLI")
    sub = p.add_subparsers(dest="cmd")

    srv = sub.add_parser("serve", help="Start the FastAPI bot server.")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--reload", action="store_true")
    srv.add_argument("--log-level", default="info")

    sub.add_parser("status", help="Check /api/ping health endpoint.")

    return p


def _serve(args: argparse.Namespace) -> int:
    import uvicorn
    from .config import BotSettings
    settings = BotSettings(host=args.host, port=args.port, reload=args.reload,
                           log_level=args.log_level)
    uvicorn.run(
        "yggdrasil.bot.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    import json
    from urllib.request import urlopen
    url = "http://localhost:8000/api/ping"
    try:
        with urlopen(url, timeout=3) as resp:  # noqa: S310
            data = json.loads(resp.read())
        print(f"ok  ygg-bot {data.get('version', '?')}  (ts={data.get('ts', '?')})")
        return 0
    except Exception as exc:
        print(f"error: {exc}")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return _serve(args)
    if args.cmd == "status":
        return _status(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
