from __future__ import annotations

import pytest

import yggdrasil.mongoengine.lib as mod


class DummyURL:
    @staticmethod
    def parse_str(value, default_scheme="mongodb"):
        return f"{default_scheme}://{value}" if "://" not in value else value


@pytest.fixture(autouse=True)
def reset_connection_state(monkeypatch):
    monkeypatch.setattr(
        mod.mongoengine.connection,
        "_connection_settings",
        {},
        raising=False,
    )
    monkeypatch.setattr(
        mod.mongoengine.connection,
        "_connections",
        {},
        raising=False,
    )
    monkeypatch.setattr(mod, "URL", DummyURL, raising=False)


def test_get_connection_settings_missing_alias():
    with pytest.raises(KeyError, match="No MongoEngine connection settings found"):
        mod.get_connection_settings("missing")


def test_get_connection_settings_jsonable_flattens_host():
    mod.mongoengine.connection._connection_settings["default"] = {
        "db": "test",
        "alias": "default",
        "host": ["mongodb://user:pass@localhost:27017/test"],
        "port": 27017,
        "username": "user",
        "password": "pass",
    }

    got = mod.get_connection_settings("default", jsonable=True)

    assert got == {
        "db": "test",
        "alias": "default",
        "host": "mongodb://user:pass@localhost:27017/test",
        "port": 27017,
        "username": "user",
        "password": "pass",
    }


def test_get_connection_settings_returns_copy():
    mod.mongoengine.connection._connection_settings["default"] = {
        "db": "test",
        "host": "mongodb://localhost:27017/test",
    }

    got = mod.get_connection_settings("default")
    got["db"] = "mutated"

    assert mod.mongoengine.connection._connection_settings["default"]["db"] == "test"


def test_connect_removes_closed_connection_and_reconnects(monkeypatch):
    class ClosedClient:
        _closed = True

    calls = []

    def fake_base_connect(**kwargs):
        calls.append(kwargs)
        return "new-client"

    monkeypatch.setattr(mod, "_base_connect", fake_base_connect, raising=True)

    mod.mongoengine.connection._connections["default"] = ClosedClient()
    mod.mongoengine.connection._connection_settings["default"] = {
        "db": "old",
        "host": "mongodb://localhost:27017/old",
    }

    built = mod.connect(db="newdb", alias="default", host="mongodb://localhost:27017/newdb")

    assert built == "new-client"
    assert "default" not in mod.mongoengine.connection._connections
    assert "default" not in mod.mongoengine.connection._connection_settings
    assert calls[0]["db"] == "newdb"


def test_connect_reregisters_when_host_changes(monkeypatch):
    class LiveClient:
        _closed = False

    registered = []
    base_calls = []

    def fake_register_connection(**kwargs):
        registered.append(kwargs)

    def fake_base_connect(**kwargs):
        base_calls.append(kwargs)
        return "client"

    monkeypatch.setattr(mod, "_base_register_connection", fake_register_connection, raising=True)
    monkeypatch.setattr(mod, "_base_connect", fake_base_connect, raising=True)

    mod.mongoengine.connection._connections["default"] = LiveClient()
    mod.mongoengine.connection._connection_settings["default"] = {
        "db": "test",
        "host": "mongodb://user1@localhost:27017/test",
    }

    mod.connect(
        db="test",
        alias="default",
        host="mongodb://user2@localhost:27017/test",
    )

    assert len(registered) == 1
    assert registered[0]["host"] == "mongodb://user2@localhost:27017/test"
    assert len(base_calls) == 1


def test_connect_accepts_registered_host_as_string_and_input_host_as_list(monkeypatch):
    class LiveClient:
        _closed = False

    registered = []

    def fake_register_connection(**kwargs):
        registered.append(kwargs)

    def fake_base_connect(**kwargs):
        return "client"

    monkeypatch.setattr(mod, "_base_register_connection", fake_register_connection, raising=True)
    monkeypatch.setattr(mod, "_base_connect", fake_base_connect, raising=True)

    mod.mongoengine.connection._connections["default"] = LiveClient()
    mod.mongoengine.connection._connection_settings["default"] = {
        "db": "test",
        "host": "mongodb://localhost:27017/test",
    }

    mod.connect(
        db="test",
        alias="default",
        host=["mongodb://localhost:27018/test"],
    )

    assert len(registered) == 1
    assert registered[0]["host"] == ["mongodb://localhost:27018/test"]