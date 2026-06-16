"""``ygg node`` — YGG node server CLI.

Subcommands::

    ygg node serve         Start the FastAPI node server (uvicorn)
    ygg node status        Check if the node is running
    ygg node bench         Run the built-in benchmark suite

Usage::

    ygg node serve --host 0.0.0.0 --port 8100
    ygg node serve --no-front            # API only, no frontend proxy
    ygg node bench
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install uvicorn", file=sys.stderr)
        return 1

    from .config import Settings
    from .api.app import create_api

    node_home = Path(args.node_home) if args.node_home else Path.cwd() / ".ygg_node"
    node_home.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        node_id=args.node_id,
        node_home=node_home,
        front_home=node_home / "front",
        allow_remote=not args.no_front,
    )

    app = create_api(settings)
    print(f"YGG node starting on http://{args.host}:{args.port}")
    print(f"  node_id:   {settings.node_id}")
    print(f"  node_home: {settings.node_home}")
    print(f"  API docs:  http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info" if not args.debug else "debug",
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    import urllib.request
    import json

    url = f"http://{args.host}:{args.port}/api/v2/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        print(f"Node status: {data.get('status', 'unknown')}")
        uptime = int(data.get("uptime", 0))
        print(f"Uptime: {uptime // 60}m {uptime % 60}s")
        return 0
    except Exception as exc:
        print(f"Node not reachable at {url}: {exc}", file=sys.stderr)
        return 1


def _bench(args: argparse.Namespace) -> int:
    """Run the built-in messenger + v2 endpoint benchmark."""
    import asyncio
    import statistics
    import time

    try:
        import httpx
    except ImportError:
        print("httpx is required: pip install httpx", file=sys.stderr)
        return 1

    from .api.app import create_api

    app = create_api()

    async def run() -> None:
        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            endpoints = [
                ("/api/ping",                 "ping       "),
                ("/api/v2/stats",             "stats      "),
                ("/api/v2/backend",           "backend    "),
                ("/api/v2/backend/summary",   "back/sum   "),
                ("/api/v2/health",            "health     "),
                ("/api/v2/audit?limit=20",    "audit      "),
                ("/api/v2/pyfunc",            "pyfunc/list"),
                ("/api/v2/pyenv",             "pyenv/list "),
            ]
            # warm-up
            for path, _ in endpoints:
                for _ in range(5):
                    await client.get(path)

            n = args.n
            print(f"\n  endpoint     n     p50us    p99us    avgus    req/s    status")
            print(f"  {'-' * 70}")
            for path, label in endpoints:
                samples: list[float] = []
                status = 0
                for _ in range(n):
                    t0 = time.perf_counter()
                    r = await client.get(path)
                    samples.append((time.perf_counter() - t0) * 1_000_000)
                    status = r.status_code
                samples.sort()
                p50 = samples[n // 2]
                p99 = samples[int(n * 0.99)]
                avg = statistics.mean(samples)
                rps = 1_000_000 / avg if avg else 0
                print(f"  {label}  {n:>4d}  {p50:>7.0f}  {p99:>7.0f}  {avg:>7.0f}  {rps:>7.0f}    {status}")

    asyncio.run(run())
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ygg node", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    serve_p = sub.add_parser("serve", help="Start the node server.")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8100)
    serve_p.add_argument("--node-id", default="default", dest="node_id")
    serve_p.add_argument("--node-home", default=None, dest="node_home")
    serve_p.add_argument("--no-front", action="store_true")
    serve_p.add_argument("--debug", action="store_true")
    serve_p.set_defaults(handler=_serve)

    status_p = sub.add_parser("status", help="Check node health.")
    status_p.add_argument("--host", default="127.0.0.1")
    status_p.add_argument("--port", type=int, default=8100)
    status_p.set_defaults(handler=_status)

    bench_p = sub.add_parser("bench", help="Run built-in endpoint benchmark.")
    bench_p.add_argument("-n", type=int, default=500, help="Iterations per endpoint")
    bench_p.set_defaults(handler=_bench)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args, _ = parser.parse_known_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)
