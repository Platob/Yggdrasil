from __future__ import annotations

import contextlib
import select
import socket
import threading
from dataclasses import dataclass
from typing import Any
from yggdrasil.environ import runtime_import_module

pymongo = runtime_import_module("pymongo", pip_name="pymongo", install=True)

__all__ = [
    "HTTPProxyConfig",
    "HTTPProxyTunnel",
    "ProxyMongoClient",
    "autoselect_mongo_client",
]


@dataclass(frozen=True, slots=True)
class HTTPProxyConfig:
    proxy_host: str
    proxy_port: int
    target_host: str
    target_port: int = 27017
    connect_timeout_s: float = 10.0


class HTTPProxyTunnel:
    """Local TCP tunnel that reaches MongoDB through an HTTP CONNECT proxy."""

    def __init__(self, config: HTTPProxyConfig):
        self.config = config
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._workers: set[threading.Thread] = set()
        self._lock = threading.Lock()
        self.local_host = "127.0.0.1"
        self.local_port = 0

    def start(self) -> None:
        if self._server_sock is not None:
            return
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.local_host, 0))
        server.listen(128)
        server.settimeout(1.0)
        self.local_port = int(server.getsockname()[1])
        self._server_sock = server
        self._thread = threading.Thread(target=self._accept_loop, name="mongo-http-proxy-tunnel", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._server_sock is not None:
            with contextlib.suppress(Exception):
                self._server_sock.close()
            self._server_sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join(timeout=1.0)

    @property
    def local_address(self) -> tuple[str, int]:
        return self.local_host, self.local_port

    def _accept_loop(self) -> None:
        assert self._server_sock is not None
        while not self._stop.is_set():
            try:
                client_sock, _ = self._server_sock.accept()
            except OSError:
                break
            except socket.timeout:
                continue

            worker = threading.Thread(target=self._handle_client, args=(client_sock,), daemon=True)
            with self._lock:
                self._workers.add(worker)
            worker.start()

    def _handle_client(self, client_sock: socket.socket) -> None:
        try:
            proxy_sock = socket.create_connection(
                (self.config.proxy_host, self.config.proxy_port),
                timeout=self.config.connect_timeout_s,
            )
        except OSError:
            with contextlib.suppress(Exception):
                client_sock.close()
            self._remove_worker()
            return

        try:
            self._connect_proxy(proxy_sock)
            self._pipe_bidirectional(client_sock, proxy_sock)
        finally:
            with contextlib.suppress(Exception):
                client_sock.close()
            with contextlib.suppress(Exception):
                proxy_sock.close()
            self._remove_worker()

    def _remove_worker(self) -> None:
        current = threading.current_thread()
        with self._lock:
            self._workers.discard(current)

    def _connect_proxy(self, proxy_sock: socket.socket) -> None:
        request = (
            f"CONNECT {self.config.target_host}:{self.config.target_port} HTTP/1.1\r\n"
            f"Host: {self.config.target_host}:{self.config.target_port}\r\n"
            "Proxy-Connection: Keep-Alive\r\n"
            "\r\n"
        ).encode("ascii")
        proxy_sock.sendall(request)
        response = self._read_http_response_head(proxy_sock)
        status_line = response.split(b"\r\n", 1)[0].decode("iso-8859-1", errors="replace")
        if " 200 " not in status_line:
            raise OSError(f"HTTP CONNECT failed: {status_line}")

    @staticmethod
    def _read_http_response_head(sock_: socket.socket) -> bytes:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock_.recv(4096)
            if not chunk:
                break
            data += chunk
            if len(data) > 64 * 1024:
                break
        return data

    @staticmethod
    def _pipe_bidirectional(left: socket.socket, right: socket.socket) -> None:
        left.setblocking(False)
        right.setblocking(False)
        sockets = [left, right]

        while True:
            readable, _, errored = select.select(sockets, [], sockets, 1.0)
            if errored:
                break
            if not readable:
                continue
            for sock_ in readable:
                other = right if sock_ is left else left
                try:
                    data = sock_.recv(64 * 1024)
                except OSError:
                    return
                if not data:
                    return
                try:
                    other.sendall(data)
                except OSError:
                    return


class ProxyMongoClient(pymongo.MongoClient):
    """MongoClient that transparently tunnels through an HTTP CONNECT proxy."""

    def __init__(
        self,
        mongo_host: str,
        *,
        mongo_port: int = 27017,
        proxy_host: str,
        proxy_port: int,
        connect_timeout_s: float = 10.0,
        **kwargs,
    ):
        self._http_proxy_tunnel = HTTPProxyTunnel(
            HTTPProxyConfig(
                proxy_host=proxy_host,
                proxy_port=proxy_port,
                target_host=mongo_host,
                target_port=mongo_port,
                connect_timeout_s=connect_timeout_s,
            )
        )
        self._http_proxy_tunnel.start()
        local_host, local_port = self._http_proxy_tunnel.local_address

        kwargs.setdefault("directConnection", True)
        kwargs.setdefault("connectTimeoutMS", int(connect_timeout_s * 1000))

        try:
            super().__init__(host=local_host, port=local_port, **kwargs)
        except Exception:
            self._http_proxy_tunnel.close()
            raise

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._http_proxy_tunnel.close()


def autoselect_mongo_client(
    mongo_host: str,
    *,
    mongo_port: int = 27017,
    proxy_host: str | None = None,
    proxy_port: int | None = None,
    connect_timeout_s: float = 10.0,
    ping_on_init: bool = True,
    **kwargs: Any,
) -> pymongo.MongoClient:
    """
    Build a MongoClient, preferring HTTP-proxy mode when available.

    Behavior:
    1. If proxy settings are provided, try ProxyMongoClient first.
    2. If tunnel/ping fails, fall back to a direct basic MongoClient.
    """
    kwargs.setdefault("connectTimeoutMS", int(connect_timeout_s * 1000))
    kwargs.setdefault("serverSelectionTimeoutMS", int(connect_timeout_s * 1000))

    if proxy_host and proxy_port:
        proxied: ProxyMongoClient | None = None
        try:
            proxied = ProxyMongoClient(
                mongo_host=mongo_host,
                mongo_port=mongo_port,
                proxy_host=proxy_host,
                proxy_port=proxy_port,
                connect_timeout_s=connect_timeout_s,
                **kwargs,
            )
            if ping_on_init:
                proxied.admin.command("ping")
            return proxied
        except Exception:
            if proxied is not None:
                with contextlib.suppress(Exception):
                    proxied.close()

    basic = pymongo.MongoClient(host=mongo_host, port=mongo_port, **kwargs)
    if ping_on_init:
        basic.admin.command("ping")
    return basic
