from __future__ import annotations

import asyncio
import time
import unittest

from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.config import Settings
from yggdrasil.node.services.discovery import DiscoveryService, _PEER_TTL


class TestDiscoveryEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = Settings(allow_remote=True)
        cls.app = create_app(settings)
        cls.client = TestClient(cls.app)

    # -- GET /api/hello (self-info) ----------------------------------------

    def test_get_hello_returns_node_info(self):
        resp = self.client.get("/api/hello")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("node_id", data)
        self.assertIn("version", data)
        self.assertIn("uptime", data)
        self.assertIn("host", data)
        self.assertIn("port", data)
        self.assertGreaterEqual(data["uptime"], 0.0)

    def test_get_hello_includes_functions(self):
        resp = self.client.get("/api/hello")
        data = resp.json()
        self.assertIn("functions", data)
        self.assertIsInstance(data["functions"], list)

    def test_get_hello_includes_channels(self):
        resp = self.client.get("/api/hello")
        data = resp.json()
        self.assertIn("channels", data)
        self.assertIsInstance(data["channels"], list)

    # -- POST /api/hello (peer registration) -------------------------------

    def test_post_hello_registers_peer(self):
        resp = self.client.post(
            "/api/hello",
            json={
                "node_id": "peer-abc",
                "host": "10.0.0.5",
                "port": 9000,
                "version": "0.2.0",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("node_id", data)
        self.assertIn("peers", data)
        self.assertIsInstance(data["peers"], list)
        # The caller should appear in the peer list
        peer_ids = [p["node_id"] for p in data["peers"]]
        self.assertIn("peer-abc", peer_ids)

    def test_post_hello_returns_own_info(self):
        resp = self.client.post(
            "/api/hello",
            json={
                "node_id": "peer-xyz",
                "host": "10.0.0.6",
                "port": 9001,
                "version": "0.1.0",
            },
        )
        data = resp.json()
        self.assertIn(data["host"], ("127.0.0.1", "0.0.0.0"))
        self.assertEqual(data["port"], 8100)
        self.assertEqual(data["version"], "0.1.0")

    def test_post_hello_updates_existing_peer(self):
        self.client.post(
            "/api/hello",
            json={
                "node_id": "peer-update",
                "host": "10.0.0.10",
                "port": 8000,
                "version": "0.1.0",
            },
        )
        # Update with new port/version
        resp = self.client.post(
            "/api/hello",
            json={
                "node_id": "peer-update",
                "host": "10.0.0.10",
                "port": 9999,
                "version": "0.3.0",
            },
        )
        data = resp.json()
        peers = {p["node_id"]: p for p in data["peers"]}
        self.assertEqual(peers["peer-update"]["port"], 9999)
        self.assertEqual(peers["peer-update"]["version"], "0.3.0")

    # -- GET /api/hello/peers ----------------------------------------------

    def test_get_peers_returns_known_peers(self):
        # Register a peer first
        self.client.post(
            "/api/hello",
            json={
                "node_id": "peer-list-test",
                "host": "10.0.0.20",
                "port": 7777,
                "version": "0.1.0",
            },
        )
        resp = self.client.get("/api/hello/peers")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("node_id", data)
        self.assertIn("peers", data)
        peer_ids = [p["node_id"] for p in data["peers"]]
        self.assertIn("peer-list-test", peer_ids)

    # -- POST /api/hello/discover ------------------------------------------

    def test_discover_empty_targets(self):
        resp = self.client.post("/api/hello/discover", json=[])
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["peers"], [])

    # -- self-info channels from messenger ---------------------------------

    def test_self_info_includes_messenger_channels(self):
        # Create a channel through the messenger service
        self.client.post("/api/messenger/channels?name=discovery-test-chan")
        resp = self.client.get("/api/hello")
        data = resp.json()
        self.assertIn("discovery-test-chan", data["channels"])


class TestDiscoveryServiceDirect(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(allow_remote=True)
        self.service = DiscoveryService(self.settings)
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_stale_peers_are_purged(self):
        from yggdrasil.node.schemas.discovery import HelloRequest

        # Register a peer
        self._run(self.service.hello(HelloRequest(
            node_id="stale-peer",
            host="10.0.0.99",
            port=5555,
            version="0.1.0",
        )))

        # Verify it is present
        resp = self._run(self.service.get_peers())
        peer_ids = [p.node_id for p in resp.peers]
        self.assertIn("stale-peer", peer_ids)

        # Simulate staleness by backdating last_seen
        with self.service._lock:
            self.service._peers["stale-peer"].last_seen = (
                time.monotonic() - _PEER_TTL - 100
            )

        # Now it should be purged on next read
        resp = self._run(self.service.get_peers())
        peer_ids = [p.node_id for p in resp.peers]
        self.assertNotIn("stale-peer", peer_ids)

    def test_hello_logs_new_peer(self):
        from yggdrasil.node.schemas.discovery import HelloRequest

        resp = self._run(self.service.hello(HelloRequest(
            node_id="fresh-peer",
            host="10.0.0.1",
            port=4000,
            version="0.1.0",
        )))
        self.assertEqual(resp.node_id, self.settings.node_id)
        peer_ids = [p.node_id for p in resp.peers]
        self.assertIn("fresh-peer", peer_ids)

    def test_get_self_info_uptime(self):
        time.sleep(0.05)
        info = self._run(self.service.get_self_info())
        self.assertGreater(info.uptime, 0.0)
        self.assertEqual(info.node_id, self.settings.node_id)
        self.assertEqual(info.version, self.settings.app_version)

    def test_discover_friends_unreachable(self):
        # Targets that don't exist should be handled gracefully
        resp = self._run(self.service.discover_friends(
            ["http://192.0.2.1:9999"]  # RFC 5737 TEST-NET, unreachable
        ))
        self.assertEqual(resp.peers, [])


if __name__ == "__main__":
    unittest.main()
