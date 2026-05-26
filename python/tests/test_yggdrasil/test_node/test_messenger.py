from __future__ import annotations

import asyncio
import time

import pytest

from yggdrasil.node.config import Settings
from yggdrasil.node.schemas.messenger import MessageSend
from yggdrasil.node.services.messenger import MessengerService, _CHANNEL_TTL


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


def test_send_message_default_channel(client):
    resp = client.post("/api/messenger", json={"text": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["channel"] == "general"
    assert data["text"] == "hello"
    assert data["node_id"] == "test-node"
    assert data["id"]
    assert data["timestamp"]


def test_send_message_with_sender(client):
    resp = client.post("/api/messenger", json={"text": "hi", "sender": "alice"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sender"] == "alice"


def test_send_to_custom_channel(client):
    client.post("/api/messenger/channels", params={"name": "dev"})
    resp = client.post("/api/messenger", json={"text": "dev msg", "channel": "dev"})
    assert resp.status_code == 200
    assert resp.json()["channel"] == "dev"


def test_send_auto_creates_channel(client):
    resp = client.post("/api/messenger", json={"text": "first", "channel": "auto-chan"})
    assert resp.status_code == 200
    assert resp.json()["channel"] == "auto-chan"

    ch_resp = client.get("/api/messenger/channels/auto-chan")
    assert ch_resp.status_code == 200
    assert ch_resp.json()["channel"]["name"] == "auto-chan"


def test_list_channels_includes_general(client):
    resp = client.get("/api/messenger/channels")
    assert resp.status_code == 200
    data = resp.json()
    names = [ch["name"] for ch in data["channels"]]
    assert "general" in names
    assert data["node_id"] == "test-node"


def test_create_channel(client):
    resp = client.post("/api/messenger/channels", params={"name": "ops"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["channel"]["name"] == "ops"
    assert data["node_id"] == "test-node"
    assert data["channel"]["message_count"] == 0
    assert data["channel"]["members"] == []


def test_create_duplicate_channel_409(client):
    client.post("/api/messenger/channels", params={"name": "dup"})
    resp = client.post("/api/messenger/channels", params={"name": "dup"})
    assert resp.status_code == 409


def test_get_channel(client):
    client.post("/api/messenger/channels", params={"name": "info"})
    resp = client.get("/api/messenger/channels/info")
    assert resp.status_code == 200
    assert resp.json()["channel"]["name"] == "info"


def test_get_nonexistent_channel_404(client):
    resp = client.get("/api/messenger/channels/no-such-channel")
    assert resp.status_code == 404


def test_delete_channel(client):
    client.post("/api/messenger/channels", params={"name": "temp"})
    del_resp = client.delete("/api/messenger/channels/temp")
    assert del_resp.status_code == 200
    assert del_resp.json()["channel"]["name"] == "temp"

    get_resp = client.get("/api/messenger/channels/temp")
    assert get_resp.status_code == 404


def test_cannot_delete_general(client):
    resp = client.delete("/api/messenger/channels/general")
    assert resp.status_code == 403


def test_get_messages(client):
    for i in range(5):
        client.post("/api/messenger", json={"text": f"msg-{i}", "channel": "general"})

    resp = client.get("/api/messenger/channels/general/messages", params={"limit": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) == 3
    assert data["channel"] == "general"
    assert data["node_id"] == "test-node"


def test_get_messages_after_timestamp(client):
    client.post("/api/messenger", json={"text": "old", "channel": "general"})
    first_resp = client.get("/api/messenger/channels/general/messages")
    ts = first_resp.json()["messages"][-1]["timestamp"]

    client.post("/api/messenger", json={"text": "new-1", "channel": "general"})
    client.post("/api/messenger", json={"text": "new-2", "channel": "general"})

    resp = client.get(
        "/api/messenger/channels/general/messages",
        params={"after": ts},
    )
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["text"] == "new-1"
    assert messages[1]["text"] == "new-2"


def test_members_tracked(client):
    client.post("/api/messenger", json={"text": "a", "sender": "alice", "channel": "general"})
    client.post("/api/messenger", json={"text": "b", "sender": "bob", "channel": "general"})
    client.post("/api/messenger", json={"text": "c", "sender": "alice", "channel": "general"})

    resp = client.get("/api/messenger/channels/general")
    assert resp.status_code == 200
    members = resp.json()["channel"]["members"]
    assert "alice" in members
    assert "bob" in members


def test_poll_returns_new_messages(client):
    seed = client.post("/api/messenger", json={"text": "seed", "channel": "general"})
    seed_id = seed.json()["id"]

    client.post("/api/messenger", json={"text": "after-seed", "channel": "general"})

    resp = client.get(
        "/api/messenger/channels/general/poll",
        params={"after_id": seed_id, "timeout": 1},
    )
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert len(messages) == 1
    assert messages[0]["text"] == "after-seed"


def test_poll_timeout_returns_empty(client):
    seed = client.post("/api/messenger", json={"text": "only", "channel": "general"})
    seed_id = seed.json()["id"]

    resp = client.get(
        "/api/messenger/channels/general/poll",
        params={"after_id": seed_id, "timeout": 0.1},
    )
    assert resp.status_code == 200
    assert resp.json()["messages"] == []


# ---------------------------------------------------------------------------
# Service unit tests
# ---------------------------------------------------------------------------


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_service() -> MessengerService:
    settings = Settings(
        node_home="/tmp/ygg_test_messenger",
        node_id="test-node",
        allow_remote=True,
    )
    return MessengerService(settings)


def test_service_send_and_retrieve():
    svc = _make_service()
    msg = _run(svc.send_message(MessageSend(text="hello", channel="general")))
    assert msg.text == "hello"
    assert msg.channel == "general"

    result = _run(svc.get_messages("general"))
    assert len(result.messages) == 1
    assert result.messages[0].id == msg.id


def test_service_stale_channel_purged():
    svc = _make_service()
    _run(svc.create_channel("ephemeral"))

    ch = svc._channels["ephemeral"]
    ch._last_active_mono = time.monotonic() - _CHANNEL_TTL - 1
    ch.last_active = "2000-01-01T00:00:00+00:00"

    channels = _run(svc.list_channels())
    names = [c.name for c in channels.channels]
    assert "ephemeral" not in names
    assert "general" in names


def test_service_general_never_purged():
    svc = _make_service()

    ch = svc._channels["general"]
    ch._last_active_mono = time.monotonic() - _CHANNEL_TTL - 9999
    ch.last_active = "2000-01-01T00:00:00+00:00"

    channels = _run(svc.list_channels())
    names = [c.name for c in channels.channels]
    assert "general" in names


def test_service_message_deque_bounded():
    svc = _make_service()
    for i in range(1050):
        _run(svc.send_message(MessageSend(text=f"m{i}", channel="general")))

    ch = svc._channels["general"]
    assert len(ch.messages) <= 1000
