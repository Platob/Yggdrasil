"""Tests for the inter-agent mesh — shared memory the fleet coordinates through."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from yggdrasil.loki.mesh import MeshStore, from_env


class TestMeshStore(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "mesh-kv.json")

    def test_put_get_and_all(self):
        ms = MeshStore(self.path)
        ms.put("ingest", "backend/src/app/ingest.py")
        ms.put("api", {"ws": "/ws"})
        self.assertEqual(ms.get("ingest"), "backend/src/app/ingest.py")
        self.assertEqual(ms.get("api"), {"ws": "/ws"})
        self.assertEqual(set(ms.all()), {"ingest", "api"})
        self.assertIsNone(ms.get("missing"))

    def test_append_builds_a_shared_log(self):
        ms = MeshStore(self.path)
        ms.append("log", "agent1 done")
        ms.append("log", "agent2 done")
        self.assertEqual(ms.get("log"), ["agent1 done", "agent2 done"])

    def test_a_second_store_sees_the_first_writes(self):
        # Two MeshStore handles on the same path = two agent processes sharing.
        MeshStore(self.path).put("shared", 42)
        self.assertEqual(MeshStore(self.path).get("shared"), 42)

    def test_missing_file_reads_empty(self):
        self.assertEqual(MeshStore(self.path).all(), {})

    def test_from_env(self):
        self.assertIsNone(from_env())
        with patch.dict(os.environ, {"LOKI_MESH": self.path}):
            store = from_env()
        self.assertIsInstance(store, MeshStore)
        self.assertEqual(str(store.path), self.path)


if __name__ == "__main__":
    unittest.main()
