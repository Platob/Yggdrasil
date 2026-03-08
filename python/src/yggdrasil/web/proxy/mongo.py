# yggdrasil/web/mongo_proxy.py
from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import os
import signal
import ssl
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ProxyConfig:
    listen_host: str
    listen_port: int
    upstream_host: str
    upstream_port: int = 27017

    # Upstream TLS (Atlas expects TLS)
    upstream_tls: bool = True
    upstream_tls_verify: bool = True
    upstream_sni: Optional[str] = None  # defaults to upstream_host

    # Optional mTLS to upstream (rare for Atlas)
    upstream_client_cert: Optional[str] = None
    upstream_client_key: Optional[str] = None
    upstream_ca_file: Optional[str] = None

    # Tuning
    connect_timeout_s: float = 10.0
    idle_timeout_s: float = 0.0  # 0 disables
    log_connections: bool = True


def _parse_hostport(s: str, default_port: int) -> Tuple[str, int]:
    # Supports: host:port, [ipv6]:port, host
    s = s.strip()
    if not s:
        raise ValueError("empty hostport")

    if s.startswith("["):
        # [ipv6]:port
        host, rest = s[1:].split("]", 1)
        port = default_port
        if rest.startswith(":"):
            port = int(rest[1:])
        return host, port

    if ":" in s and s.count(":") == 1:
        host, port_s = s.split(":")
        return host, int(port_s)

    # Could be IPv6 without brackets or host without port.
    # If it's a valid IPv6 literal, keep it and use default port.
    with contextlib.suppress(ValueError):
        ipaddress.IPv6Address(s)
        return s, default_port

    # host without port
    return s, default_port


def _make_upstream_ssl(cfg: ProxyConfig) -> Optional[ssl.SSLContext]:
    if not cfg.upstream_tls:
        return None

    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    if not cfg.upstream_tls_verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    if cfg.upstream_ca_file:
        ctx.load_verify_locations(cafile=cfg.upstream_ca_file)

    if cfg.upstream_client_cert:
        ctx.load_cert_chain(
            certfile=cfg.upstream_client_cert,
            keyfile=cfg.upstream_client_key,
        )

    # MongoDB wire protocol is just TCP; no ALPN needed.
    return ctx


async def _pipe(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    idle_timeout_s: float,
    direction: str,
) -> None:
    """
    Copy bytes from reader to writer until EOF.
    """
    last = time.monotonic()
    try:
        while True:
            if idle_timeout_s > 0:
                # Wait for data with timeout to detect idle connections.
                try:
                    data = await asyncio.wait_for(reader.read(64 * 1024), timeout=1.0)
                except asyncio.TimeoutError:
                    if time.monotonic() - last >= idle_timeout_s:
                        raise TimeoutError(f"idle timeout ({direction})")
                    continue
            else:
                data = await reader.read(64 * 1024)

            if not data:
                break

            last = time.monotonic()
            writer.write(data)
            await writer.drain()
    finally:
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def _handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    cfg: ProxyConfig,
    ssl_ctx: Optional[ssl.SSLContext],
) -> None:
    peer = client_writer.get_extra_info("peername")
    local = client_writer.get_extra_info("sockname")

    upstream_reader: asyncio.StreamReader
    upstream_writer: asyncio.StreamWriter

    sni = cfg.upstream_sni or cfg.upstream_host

    if cfg.log_connections:
        print(f"[mongo-proxy] + conn from={peer} to_local={local} -> {cfg.upstream_host}:{cfg.upstream_port} tls={cfg.upstream_tls}", flush=True)

    try:
        connect_coro = asyncio.open_connection(
            host=cfg.upstream_host,
            port=cfg.upstream_port,
            ssl=ssl_ctx,
            server_hostname=sni if (cfg.upstream_tls and cfg.upstream_tls_verify) else sni if cfg.upstream_tls else None,
        )

        upstream_reader, upstream_writer = await asyncio.wait_for(
            connect_coro,
            timeout=cfg.connect_timeout_s,
        )

        # Bi-directional piping
        t1 = asyncio.create_task(
            _pipe(client_reader, upstream_writer, idle_timeout_s=cfg.idle_timeout_s, direction="client->upstream")
        )
        t2 = asyncio.create_task(
            _pipe(upstream_reader, client_writer, idle_timeout_s=cfg.idle_timeout_s, direction="upstream->client")
        )

        done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_EXCEPTION)

        # Bubble exceptions if any
        for d in done:
            exc = d.exception()
            if exc:
                raise exc

        for p in pending:
            p.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await p

    except Exception as e:
        if cfg.log_connections:
            print(f"[mongo-proxy] ! conn error peer={peer} err={type(e).__name__}: {e}", flush=True)
    finally:
        with contextlib.suppress(Exception):
            client_writer.close()
            await client_writer.wait_closed()


async def serve(cfg: ProxyConfig) -> None:
    ssl_ctx = _make_upstream_ssl(cfg)

    server = await asyncio.start_server(
        lambda r, w: _handle_client(r, w, cfg, ssl_ctx),
        host=cfg.listen_host,
        port=cfg.listen_port,
        start_serving=True,
    )

    addrs = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    print(f"[mongo-proxy] listening on {addrs}", flush=True)

    stop = asyncio.Event()

    def _ask_stop(*_args: object) -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _ask_stop)

    await stop.wait()

    print("[mongo-proxy] shutting down...", flush=True)
    server.close()
    await server.wait_closed()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="yggdrasil.web.mongo_proxy", description="Local TCP proxy to MongoDB Atlas (TLS upstream).")
    p.add_argument("--listen", default=os.getenv("MONGO_PROXY_LISTEN", "127.0.0.1:27017"),
                   help="Listen address (host:port). Default: 127.0.0.1:27017")
    p.add_argument("--upstream", default=os.getenv("MONGO_PROXY_UPSTREAM", "intraday-pl-0.e05ii1.mongodb.net:27017"),
                   help="Upstream address (host:port). Default: intraday-pl-0.e05ii1.mongodb.net:27017")

    p.add_argument("--upstream-tls", action="store_true", default=_env_bool("MONGO_PROXY_UPSTREAM_TLS", True),
                   help="Enable TLS to upstream (Atlas wants this).")
    p.add_argument("--no-upstream-tls", dest="upstream_tls", action="store_false",
                   help="Disable TLS to upstream (not recommended).")

    p.add_argument("--upstream-verify", action="store_true", default=_env_bool("MONGO_PROXY_UPSTREAM_VERIFY", True),
                   help="Verify upstream cert + hostname (recommended).")
    p.add_argument("--no-upstream-verify", dest="upstream_verify", action="store_false",
                   help="Disable upstream cert verification (debug only).")

    p.add_argument("--upstream-sni", default=os.getenv("MONGO_PROXY_UPSTREAM_SNI"),
                   help="Override SNI/hostname used for TLS verification (defaults to upstream host).")

    p.add_argument("--upstream-ca", default=os.getenv("MONGO_PROXY_UPSTREAM_CA"),
                   help="Custom CA bundle path for upstream verification.")
    p.add_argument("--upstream-client-cert", default=os.getenv("MONGO_PROXY_UPSTREAM_CLIENT_CERT"),
                   help="Client cert path for upstream mTLS (optional).")
    p.add_argument("--upstream-client-key", default=os.getenv("MONGO_PROXY_UPSTREAM_CLIENT_KEY"),
                   help="Client key path for upstream mTLS (optional).")

    p.add_argument("--connect-timeout", type=float, default=float(os.getenv("MONGO_PROXY_CONNECT_TIMEOUT", "10")),
                   help="Upstream connect timeout seconds.")
    p.add_argument("--idle-timeout", type=float, default=float(os.getenv("MONGO_PROXY_IDLE_TIMEOUT", "0")),
                   help="Idle timeout seconds (0 disables).")
    p.add_argument("--quiet", action="store_true", help="Reduce logging.")

    args = p.parse_args(argv)

    listen_host, listen_port = _parse_hostport(args.listen, 27017)
    upstream_host, upstream_port = _parse_hostport(args.upstream, 27017)

    cfg = ProxyConfig(
        listen_host=listen_host,
        listen_port=listen_port,
        upstream_host=upstream_host,
        upstream_port=upstream_port,
        upstream_tls=bool(args.upstream_tls),
        upstream_tls_verify=bool(args.upstream_verify),
        upstream_sni=args.upstream_sni,
        upstream_ca_file=args.upstream_ca,
        upstream_client_cert=args.upstream_client_cert,
        upstream_client_key=args.upstream_client_key,
        connect_timeout_s=float(args.connect_timeout),
        idle_timeout_s=float(args.idle_timeout),
        log_connections=not args.quiet,
    )

    try:
        asyncio.run(serve(cfg))
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())