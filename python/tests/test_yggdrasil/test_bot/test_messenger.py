from __future__ import annotations

import asyncio
import time
import unittest

from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.config import Settings
from yggdrasil.node.services.messenger import MessengerService, _CHANNEL_TTL


class TestMessengerEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = Settings(allow_remote=True)
        cls.app = create_app(settings)
        cls.client = TestClient(cls.app)

    # -- send + get messages -----------------------------------------------

    def test_send_message(self):
        resp = self.client.post(
            "/api/messenger",
            json={"text": "hello world", "sender": "alice"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["text"], "hello world")
        self.assertEqual(data["sender"], "alice")
        self.assertEqual(data["channel"], "general")
        self.assertIn("id", data)
        self.assertIn("timestamp", data)

    def test_send_message_default_channel(self):
        resp = self.client.post(
            "/api/messenger",
            json={"text": "test msg"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["channel"], "general")

    def test_send_to_custom_channel(self):
        self.client.post("/api/messenger/channels?name=dev")
        resp = self.client.post(
            "/api/messenger",
            json={"text": "dev msg", "sender": "bob", "channel": "dev"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["channel"], "dev")

    def test_send_auto_creates_channel(self):
        resp = self.client.post(
            "/api/messenger",
            json={"text": "auto create", "sender": "carl", "channel": "auto-chan"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["channel"], "auto-chan")

        resp = self.client.get("/api/messenger/channels/auto-chan")
        self.assertEqual(resp.status_code, 200)

    def test_get_messages(self):
        for i in range(5):
            self.client.post(
                "/api/messenger",
                json={"text": f"msg-{i}", "sender": "alice", "channel": "general"},
            )
        resp = self.client.get("/api/messenger/channels/general/messages?limit=3")
        self.assertEqual(resp.status_code, 200)
        msgs = resp.json()["messages"]
        self.assertLessEqual(len(msgs), 3)

    def test_get_messages_after_timestamp(self):
        r0 = self.client.post(
            "/api/messenger",
            json={"text": "ts-msg-0", "sender": "alice", "channel": "general"},
        )
        ts = r0.json()["timestamp"]

        time.sleep(0.01)

        for i in range(1, 3):
            self.client.post(
                "/api/messenger",
                json={"text": f"ts-msg-{i}", "sender": "alice", "channel": "general"},
            )

        resp = self.client.get(f"/api/messenger/channels/general/messages?after={ts}")
        self.assertEqual(resp.status_code, 200)
        msgs = resp.json()["messages"]
        self.assertGreater(len(msgs), 0)
        for m in msgs:
            self.assertGreaterEqual(m["timestamp"], ts)

    # -- channel CRUD ------------------------------------------------------

    def test_list_channels(self):
        resp = self.client.get("/api/messenger/channels")
        self.assertEqual(resp.status_code, 200)
        names = [ch["name"] for ch in resp.json()["channels"]]
        self.assertIn("general", names)

    def test_create_channel(self):
        resp = self.client.post("/api/messenger/channels?name=test-create")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["channel"]["name"], "test-create")

    def test_create_duplicate_channel(self):
        self.client.post("/api/messenger/channels?name=dup-test")
        resp = self.client.post("/api/messenger/channels?name=dup-test")
        self.assertEqual(resp.status_code, 409)

    def test_get_channel(self):
        resp = self.client.get("/api/messenger/channels/general")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["channel"]["name"], "general")

    def test_get_channel_not_found(self):
        resp = self.client.get("/api/messenger/channels/nonexistent999")
        self.assertEqual(resp.status_code, 404)

    def test_delete_channel(self):
        self.client.post("/api/messenger/channels?name=to-delete")
        resp = self.client.delete("/api/messenger/channels/to-delete")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/api/messenger/channels/to-delete")
        self.assertEqual(resp.status_code, 404)

    def test_cannot_delete_general(self):
        resp = self.client.delete("/api/messenger/channels/general")
        self.assertEqual(resp.status_code, 403)

    # -- members -----------------------------------------------------------

    def test_members_tracked(self):
        self.client.post("/api/messenger/channels?name=member-test")
        self.client.post(
            "/api/messenger",
            json={"text": "hi", "sender": "alice", "channel": "member-test"},
        )
        self.client.post(
            "/api/messenger",
            json={"text": "hey", "sender": "bob", "channel": "member-test"},
        )
        resp = self.client.get("/api/messenger/channels/member-test")
        members = resp.json()["channel"]["members"]
        self.assertIn("alice", members)
        self.assertIn("bob", members)


class TestMessengerServiceDirect(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(allow_remote=True)
        self.service = MessengerService(self.settings)
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_send_and_retrieve(self):
        from yggdrasil.node.schemas.messenger import MessageSend
        msg = self._run(self.service.send_message(
            MessageSend(text="direct test", sender="tester")
        ))
        self.assertEqual(msg.text, "direct test")

        resp = self._run(self.service.get_messages("general"))
        texts = [m.text for m in resp.messages]
        self.assertIn("direct test", texts)

    def test_auto_purge_stale_channels(self):
        from yggdrasil.node.schemas.messenger import MessageSend
        import datetime as dt

        self._run(self.service.create_channel("stale-chan"))
        ch = self.service._channels["stale-chan"]
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=_CHANNEL_TTL + 100)
        ch.last_active = old.isoformat()
        ch._last_active_mono = time.monotonic() - _CHANNEL_TTL - 100

        resp = self._run(self.service.list_channels())
        names = [c.name for c in resp.channels]
        self.assertNotIn("stale-chan", names)
        self.assertIn("general", names)

    def test_general_never_purged(self):
        import datetime as dt

        ch = self.service._channels["general"]
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=_CHANNEL_TTL + 100)
        ch.last_active = old.isoformat()

        resp = self._run(self.service.list_channels())
        names = [c.name for c in resp.channels]
        self.assertIn("general", names)

    def test_poll_returns_new_messages(self):
        from yggdrasil.node.schemas.messenger import MessageSend

        self._run(self.service.send_message(
            MessageSend(text="seed", sender="a")
        ))
        msgs = self._run(self.service.get_messages("general"))
        seed_id = msgs.messages[-1].id

        self._run(self.service.send_message(
            MessageSend(text="new msg", sender="b")
        ))

        poll_resp = self._run(self.service.poll_messages(
            "general", after_id=seed_id, timeout=1.0
        ))
        self.assertEqual(len(poll_resp.messages), 1)
        self.assertEqual(poll_resp.messages[0].text, "new msg")

    def test_poll_timeout_returns_empty(self):
        from yggdrasil.node.schemas.messenger import MessageSend

        self._run(self.service.send_message(
            MessageSend(text="x", sender="a")
        ))
        msgs = self._run(self.service.get_messages("general"))
        last_id = msgs.messages[-1].id

        t0 = time.monotonic()
        poll_resp = self._run(self.service.poll_messages(
            "general", after_id=last_id, timeout=0.5
        ))
        elapsed = time.monotonic() - t0
        self.assertEqual(len(poll_resp.messages), 0)
        self.assertGreaterEqual(elapsed, 0.4)

    def test_message_deque_bounded(self):
        from yggdrasil.node.schemas.messenger import MessageSend

        for i in range(1100):
            self._run(self.service.send_message(
                MessageSend(text=f"m{i}", sender="bulk")
            ))

        ch = self.service._channels["general"]
        self.assertLessEqual(len(ch.messages), 1000)


class TestMessengerPoll(unittest.TestCase):
    """Test long-poll with concurrent send."""

    def test_poll_wakes_on_send(self):
        settings = Settings(allow_remote=True)
        service = MessengerService(settings)
        loop = asyncio.new_event_loop()

        from yggdrasil.node.schemas.messenger import MessageSend

        loop.run_until_complete(service.send_message(
            MessageSend(text="seed", sender="a")
        ))
        msgs = loop.run_until_complete(service.get_messages("general"))
        seed_id = msgs.messages[-1].id

        async def _test():
            async def _delayed_send():
                await asyncio.sleep(0.2)
                await service.send_message(MessageSend(text="wakeup", sender="b"))

            send_task = asyncio.create_task(_delayed_send())
            t0 = time.monotonic()
            result = await service.poll_messages("general", after_id=seed_id, timeout=5.0)
            elapsed = time.monotonic() - t0
            await send_task
            return result, elapsed

        result, elapsed = loop.run_until_complete(_test())
        loop.close()

        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].text, "wakeup")
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
