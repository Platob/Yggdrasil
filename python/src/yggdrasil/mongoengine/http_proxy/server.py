from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import socket
import time
from dataclasses import dataclass, field
from typing import Mapping

from yggdrasil.mongoengine.lib import connect, get_connection, get_db

__all__ = [
    "ProxyStats",
    "MongoHTTPProxyConfig",
    "parse_host_port",
    "MongoHTTPProxyServer",
]

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ProxyStats:
    started_at: float = field(default_factory=time.time)
    active_connections: int = 0
    total_connections: int = 0
    total_connect_success: int = 0
    total_connect_failure: int = 0
    bytes_client_to_upstream: int = 0
    bytes_upstream_to_client: int = 0

    def as_dict(self) -> dict[str, float | int]:
        uptime_s = max(0.0, time.time() - self.started_at)
        return {
            "started_at": self.started_at,
            "uptime_s": uptime_s,
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "total_connect_success": self.total_connect_success,
            "total_connect_failure": self.total_connect_failure,
            "bytes_client_to_upstream": self.bytes_client_to_upstream,
            "bytes_upstream_to_client": self.bytes_upstream_to_client,
        }


@dataclass(frozen=True, slots=True)
class MongoHTTPProxyConfig:
    listen_host: str = "127.0.0.1"
    listen_port: int = 8080
    upstream_host: str = "127.0.0.1"
    upstream_port: int = 27017
    mongo_uri: str | None = None
    mongo_db: str | None = None
    mongo_alias: str = "http_proxy"
    connect_timeout_s: float = 10.0
    idle_timeout_s: float = 300.0
    max_header_bytes: int = 64 * 1024
    log_level: str = "INFO"


def parse_host_port(value: str, default_port: int) -> tuple[str, int]:
    raw = value.strip()
    if not raw:
        raise ValueError("Host/port value cannot be empty")

    if raw.startswith("["):
        host, sep, tail = raw[1:].partition("]")
        if not sep:
            raise ValueError(f"Invalid bracketed host value: {value!r}")
        if tail.startswith(":"):
            return host, int(tail[1:])
        return host, default_port

    if raw.count(":") == 1:
        host, port = raw.split(":", 1)
        return host, int(port)

    return raw, default_port


class MongoHTTPProxyServer:
    def __init__(self, config: MongoHTTPProxyConfig):
        self.config = config
        self.stats = ProxyStats()
        self._server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        self._configure_logging()
        self._connect_mongoengine()
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.config.listen_host,
            port=self.config.listen_port,
            reuse_address=True,
            reuse_port=False,
            start_serving=True,
        )
        sockets = self._server.sockets or []
        for sock in sockets:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        addrs = ", ".join(str(sock.getsockname()) for sock in sockets)
        LOGGER.info("Mongo HTTP proxy listening on %s", addrs)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _stop(*_args: object) -> None:
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _stop)

        await stop_event.wait()
        await self.close()

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    def _configure_logging(self) -> None:
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def _connect_mongoengine(self) -> None:
        if self.config.mongo_uri:
            connect(
                alias=self.config.mongo_alias,
                host=self.config.mongo_uri,
                db=self.config.mongo_db,
                serverSelectionTimeoutMS=int(self.config.connect_timeout_s * 1000),
            )
            client = get_connection(alias=self.config.mongo_alias)
            with contextlib.suppress(Exception):
                client.admin.command("ping")
            LOGGER.info("MongoEngine connection established for alias=%s", self.config.mongo_alias)

    async def _handle_client(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        self.stats.active_connections += 1
        self.stats.total_connections += 1
        peer = client_writer.get_extra_info("peername")

        try:
            method, target, version, headers = await self._read_request_head(client_reader)
            m_upper = method.upper()
            if m_upper == "CONNECT":
                await self._handle_connect(client_reader, client_writer, target)
                return

            if target.startswith("/health"):
                await self._send_json(
                    client_writer,
                    status_code=200,
                    payload={
                        "ok": True,
                        "upstream": f"{self.config.upstream_host}:{self.config.upstream_port}",
                    },
                )
                return

            if target.startswith("/metrics"):
                await self._send_json(client_writer, status_code=200, payload=self.stats.as_dict())
                return

            if target.startswith("/mongo/current-db"):
                await self._send_json(
                    client_writer,
                    status_code=200,
                    payload=self._current_db_payload(),
                )
                return

            await self._send_json(
                client_writer,
                status_code=405,
                payload={
                    "error": "Use CONNECT host:port for MongoDB tunneling.",
                    "method": m_upper,
                    "target": target,
                    "version": version,
                    "headers": headers,
                },
            )
        except Exception as exc:
            LOGGER.warning("Client error from %s: %s", peer, exc)
            await self._send_json(
                client_writer,
                status_code=400,
                payload={"error": type(exc).__name__, "message": str(exc)},
            )
        finally:
            self.stats.active_connections = max(0, self.stats.active_connections - 1)
            with contextlib.suppress(Exception):
                client_writer.close()
                await client_writer.wait_closed()

    def _current_db_payload(self) -> dict[str, object]:
        db_name = self.config.mongo_db
        if not db_name:
            with contextlib.suppress(Exception):
                db_name = str(get_db(alias=self.config.mongo_alias).name)
        return {
            "alias": self.config.mongo_alias,
            "db": db_name,
            "upstream": f"{self.config.upstream_host}:{self.config.upstream_port}",
        }

    async def _read_request_head(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[str, str, str, dict[str, str]]:
        buffer = bytearray()
        while b"\r\n\r\n" not in buffer:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=self.config.idle_timeout_s)
            if not chunk:
                break
            buffer.extend(chunk)
            if len(buffer) > self.config.max_header_bytes:
                raise ValueError("Header size exceeds configured limit")

        head, _, _ = bytes(buffer).partition(b"\r\n\r\n")
        lines = head.decode("iso-8859-1").split("\r\n")
        if not lines or len(lines[0].split(" ")) != 3:
            raise ValueError("Malformed HTTP request line")

        method, target, version = lines[0].split(" ", 2)
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            key, sep, value = line.partition(":")
            if not sep:
                continue
            headers[key.strip().lower()] = value.strip()
        return method, target, version, headers

    async def _handle_connect(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        target: str,
    ) -> None:
        host, port = parse_host_port(target, self.config.upstream_port)
        host = host or self.config.upstream_host
        timeout = self.config.connect_timeout_s

        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(host=host, port=port),
                timeout=timeout,
            )
        except Exception as exc:
            self.stats.total_connect_failure += 1
            LOGGER.warning("CONNECT failed for %s:%s (%s)", host, port, exc)
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
            await client_writer.drain()
            return

        self.stats.total_connect_success += 1
        client_writer.write(b"HTTP/1.1 200 Connection Established\r\nProxy-Agent: yggdrasil-mongo-http-proxy\r\n\r\n")
        await client_writer.drain()

        async def _pump(
            source: asyncio.StreamReader,
            sink: asyncio.StreamWriter,
            stat_key: str,
        ) -> None:
            while True:
                chunk = await source.read(64 * 1024)
                if not chunk:
                    break
                sink.write(chunk)
                await sink.drain()
                if stat_key == "c2u":
                    self.stats.bytes_client_to_upstream += len(chunk)
                else:
                    self.stats.bytes_upstream_to_client += len(chunk)
            with contextlib.suppress(Exception):
                sink.close()
                await sink.wait_closed()

        t1 = asyncio.create_task(_pump(client_reader, upstream_writer, "c2u"))
        t2 = asyncio.create_task(_pump(upstream_reader, client_writer, "u2c"))

        done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for task in done:
            with contextlib.suppress(Exception):
                task.result()

    async def _send_json(
        self,
        writer: asyncio.StreamWriter,
        *,
        status_code: int,
        payload: Mapping[str, object],
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        reason = {
            200: "OK",
            400: "Bad Request",
            405: "Method Not Allowed",
        }.get(status_code, "OK")
        head = (
            f"HTTP/1.1 {status_code} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii")
        writer.write(head + body)
        await writer.drain()
