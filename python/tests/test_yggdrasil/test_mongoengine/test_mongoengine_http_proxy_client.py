from __future__ import annotations

import contextlib
import select
import socket
import threading
import time

import yggdrasil.mongoengine.http_proxy.client as mod


def _recv_until(sock_: socket.socket, marker: bytes) -> bytes:
    data = b""
    while marker not in data:
        chunk = sock_.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def _start_echo_server() -> tuple[socket.socket, int]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(5)

    def _run() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                break
            threading.Thread(target=_echo_conn, args=(conn,), daemon=True).start()

    def _echo_conn(conn: socket.socket) -> None:
        with contextlib.suppress(Exception):
            while True:
                payload = conn.recv(4096)
                if not payload:
                    break
                conn.sendall(payload)
        with contextlib.suppress(Exception):
            conn.close()

    threading.Thread(target=_run, daemon=True).start()
    return server, int(server.getsockname()[1])


def _start_http_connect_proxy(target_port: int) -> tuple[socket.socket, int]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(5)

    def _run() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                break
            threading.Thread(target=_proxy_conn, args=(conn,), daemon=True).start()

    def _proxy_conn(client: socket.socket) -> None:
        head = _recv_until(client, b"\r\n\r\n")
        assert f"CONNECT 127.0.0.1:{target_port}".encode("ascii") in head

        upstream = socket.create_connection(("127.0.0.1", target_port), timeout=2)
        client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")

        sockets = [client, upstream]
        try:
            while True:
                readable, _, _ = select.select(sockets, [], [], 0.5)
                if not readable:
                    continue
                for sock_ in readable:
                    other = upstream if sock_ is client else client
                    data = sock_.recv(4096)
                    if not data:
                        return
                    other.sendall(data)
        finally:
            with contextlib.suppress(Exception):
                client.close()
            with contextlib.suppress(Exception):
                upstream.close()

    threading.Thread(target=_run, daemon=True).start()
    return server, int(server.getsockname()[1])


def test_http_proxy_tunnel_forwards_bytes_over_connect() -> None:
    target_server, target_port = _start_echo_server()
    proxy_server, proxy_port = _start_http_connect_proxy(target_port)

    tunnel = mod.HTTPProxyTunnel(
        mod.HTTPProxyConfig(
            proxy_host="127.0.0.1",
            proxy_port=proxy_port,
            target_host="127.0.0.1",
            target_port=target_port,
        )
    )
    tunnel.start()

    try:
        sock = socket.create_connection(tunnel.local_address, timeout=2)
        sock.sendall(b"hello-proxy")
        echoed = sock.recv(64)
        assert echoed == b"hello-proxy"
        sock.close()
    finally:
        tunnel.close()
        target_server.close()
        proxy_server.close()


def test_proxy_mongo_client_starts_local_tunnel(monkeypatch) -> None:
    target_server, target_port = _start_echo_server()
    proxy_server, proxy_port = _start_http_connect_proxy(target_port)

    captured = {}
    orig_init = mod.pymongo.MongoClient.__init__
    orig_close = mod.pymongo.MongoClient.close

    def fake_init(self, host=None, port=None, **kwargs):
        captured["host"] = host
        captured["port"] = port
        captured["kwargs"] = kwargs

    def fake_close(self):
        return None

    monkeypatch.setattr(mod.pymongo.MongoClient, "__init__", fake_init, raising=True)
    monkeypatch.setattr(mod.pymongo.MongoClient, "close", fake_close, raising=True)

    client = mod.ProxyMongoClient(
        mongo_host="127.0.0.1",
        mongo_port=target_port,
        proxy_host="127.0.0.1",
        proxy_port=proxy_port,
    )
    try:
        assert captured["host"] == "127.0.0.1"
        assert isinstance(captured["port"], int)
        assert captured["port"] > 0
        assert captured["kwargs"]["directConnection"] is True
    finally:
        client.close()
        time.sleep(0.05)
        target_server.close()
        proxy_server.close()
        monkeypatch.setattr(mod.pymongo.MongoClient, "__init__", orig_init, raising=True)
        monkeypatch.setattr(mod.pymongo.MongoClient, "close", orig_close, raising=True)


def test_autoselect_uses_proxy_client_when_ping_succeeds(monkeypatch) -> None:
    class GoodProxyClient:
        class _Admin:
            @staticmethod
            def command(value: str) -> dict[str, int]:
                assert value == "ping"
                return {"ok": 1}

        admin = _Admin()

    monkeypatch.setattr(mod, "ProxyMongoClient", lambda **_: GoodProxyClient(), raising=True)

    result = mod.autoselect_mongo_client(
        mongo_host="mongo.internal",
        proxy_host="proxy.internal",
        proxy_port=8080,
    )
    assert isinstance(result, GoodProxyClient)


def test_autoselect_falls_back_to_basic_client(monkeypatch) -> None:
    calls = {}

    class BasicClient:
        class _Admin:
            @staticmethod
            def command(value: str) -> dict[str, int]:
                assert value == "ping"
                return {"ok": 1}

        admin = _Admin()

    def fail_proxy(**_kwargs):
        raise RuntimeError("proxy failed")

    def make_basic(*, host, port, **kwargs):
        calls["host"] = host
        calls["port"] = port
        calls["kwargs"] = kwargs
        return BasicClient()

    monkeypatch.setattr(mod, "ProxyMongoClient", fail_proxy, raising=True)
    monkeypatch.setattr(mod.pymongo, "MongoClient", make_basic, raising=True)

    result = mod.autoselect_mongo_client(
        mongo_host="mongo.internal",
        mongo_port=27018,
        proxy_host="proxy.internal",
        proxy_port=8080,
    )

    assert isinstance(result, BasicClient)
    assert calls["host"] == "mongo.internal"
    assert calls["port"] == 27018
