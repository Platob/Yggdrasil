from __future__ import annotations

import pytest

import yggdrasil.mongoengine.http_proxy.server as mod


def test_parse_host_port_with_explicit_port() -> None:
    assert mod.parse_host_port("localhost:27018", 27017) == ("localhost", 27018)


def test_parse_host_port_with_ipv6_brackets() -> None:
    assert mod.parse_host_port("[::1]:27019", 27017) == ("::1", 27019)


def test_parse_host_port_with_default_port() -> None:
    assert mod.parse_host_port("mongo.internal", 27017) == ("mongo.internal", 27017)


def test_connect_mongoengine_uses_connect_and_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class DummyAdmin:
        def command(self, value: str) -> dict[str, int]:
            assert value == "ping"
            return {"ok": 1}

    class DummyClient:
        admin = DummyAdmin()

    def fake_connect(**kwargs):
        calls.append(kwargs)

    def fake_get_connection(alias: str):
        assert alias == "alias1"
        return DummyClient()

    monkeypatch.setattr(mod, "connect", fake_connect, raising=True)
    monkeypatch.setattr(mod, "get_connection", fake_get_connection, raising=True)

    config = mod.MongoHTTPProxyConfig(
        mongo_uri="mongodb://localhost:27017/test",
        mongo_db="test",
        mongo_alias="alias1",
    )
    server = mod.MongoHTTPProxyServer(config)

    server._connect_mongoengine()

    assert len(calls) == 1
    assert calls[0]["alias"] == "alias1"
    assert calls[0]["host"] == "mongodb://localhost:27017/test"


def test_current_db_payload_uses_alias_db(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyDB:
        name = "proxydb"

    def fake_get_db(alias: str):
        assert alias == "ctx_alias"
        return DummyDB()

    monkeypatch.setattr(mod, "get_db", fake_get_db, raising=True)

    server = mod.MongoHTTPProxyServer(
        mod.MongoHTTPProxyConfig(
            mongo_alias="ctx_alias",
            upstream_host="mongo.internal",
            upstream_port=27017,
        )
    )
    payload = server._current_db_payload()

    assert payload["alias"] == "ctx_alias"
    assert payload["db"] == "proxydb"
    assert payload["upstream"] == "mongo.internal:27017"
