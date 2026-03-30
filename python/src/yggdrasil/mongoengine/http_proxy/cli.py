from __future__ import annotations

import argparse
import asyncio
import os
from typing import Optional

from .server import MongoHTTPProxyConfig, MongoHTTPProxyServer, parse_host_port

__all__ = ["build_parser", "main"]


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yggdrasil.mongoengine.http_proxy",
        description="HTTP CONNECT proxy optimized for MongoDB/PyMongo tunnel traffic.",
    )
    parser.add_argument("--listen", default=os.getenv("YGG_MONGO_HTTP_PROXY_LISTEN", "127.0.0.1:8080"))
    parser.add_argument("--upstream", default=os.getenv("YGG_MONGO_HTTP_PROXY_UPSTREAM", "127.0.0.1:27017"))
    parser.add_argument("--mongo-uri", default=os.getenv("YGG_MONGO_HTTP_PROXY_MONGO_URI"))
    parser.add_argument("--mongo-db", default=os.getenv("YGG_MONGO_HTTP_PROXY_MONGO_DB"))
    parser.add_argument("--mongo-alias", default=os.getenv("YGG_MONGO_HTTP_PROXY_MONGO_ALIAS", "http_proxy"))
    parser.add_argument("--connect-timeout", type=float, default=float(os.getenv("YGG_MONGO_HTTP_PROXY_CONNECT_TIMEOUT", "10")))
    parser.add_argument("--idle-timeout", type=float, default=float(os.getenv("YGG_MONGO_HTTP_PROXY_IDLE_TIMEOUT", "300")))
    parser.add_argument("--max-header-bytes", type=int, default=int(os.getenv("YGG_MONGO_HTTP_PROXY_MAX_HEADER_BYTES", str(64 * 1024))))
    parser.add_argument("--debug", action="store_true", default=_bool_from_env("YGG_MONGO_HTTP_PROXY_DEBUG", False))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    listen_host, listen_port = parse_host_port(args.listen, 8080)
    upstream_host, upstream_port = parse_host_port(args.upstream, 27017)

    config = MongoHTTPProxyConfig(
        listen_host=listen_host,
        listen_port=listen_port,
        upstream_host=upstream_host,
        upstream_port=upstream_port,
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
        mongo_alias=args.mongo_alias,
        connect_timeout_s=args.connect_timeout,
        idle_timeout_s=args.idle_timeout,
        max_header_bytes=args.max_header_bytes,
        log_level="DEBUG" if args.debug else "INFO",
    )

    server = MongoHTTPProxyServer(config)
    try:
        asyncio.run(server.serve_forever())
        return 0
    except KeyboardInterrupt:
        return 130
