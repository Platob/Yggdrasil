"""Cross-node filesystem: peer URL resolution, the global-tree node roots,
and folder-zip download. Proxying itself needs a live peer (covered by
integration), so here we test the gateway logic that drives it.
"""
from __future__ import annotations

import asyncio
import tempfile
import zipfile
from pathlib import Path

import unittest

from yggdrasil.node.api.routers.fs import list_nodes
from yggdrasil.node.api.schemas.network import PeerRegisterRequest
from yggdrasil.node.api.services.backend import BackendService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.api.services.network import NetworkService
from yggdrasil.node.api.services.saga import SagaService
from yggdrasil.node.config import Settings


def _ctx(home: Path):
    settings = Settings(
        node_id="local-node", node_home=home, front_home=home,
        saga_home=home / ".saga", host="127.0.0.1", port=8100,
    )
    fs = FsService(settings)
    network = NetworkService(settings, BackendService(settings))
    return settings, fs, network


class TestPeerUrl(unittest.TestCase):
    def test_self_and_unknown_resolve_to_none(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, network = _ctx(Path(d))
            self.assertIsNone(network.peer_url("local-node"))   # self → local
            self.assertIsNone(network.peer_url("ghost-node"))   # unknown peer

    def test_registered_peer_resolves(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, network = _ctx(Path(d))
            asyncio.run(network.register_peer(PeerRegisterRequest(
                node_id="peer-2", host="10.0.0.5", port=8100,
            )))
            self.assertEqual(network.peer_url("peer-2"), "http://10.0.0.5:8100")


class TestNodeRoots(unittest.TestCase):
    def test_self_is_always_a_root(self):
        with tempfile.TemporaryDirectory() as d:
            settings, fs, network = _ctx(Path(d))
            saga = SagaService(settings)
            res = asyncio.run(list_nodes(service=fs, network=network, saga=saga))
            roots = res["nodes"]
            self.assertEqual(roots[0]["node_id"], "local-node")
            self.assertTrue(roots[0]["self"])

    def test_linked_peer_appears_as_root(self):
        with tempfile.TemporaryDirectory() as d:
            settings, fs, network = _ctx(Path(d))
            saga = SagaService(settings)
            asyncio.run(network.register_peer(PeerRegisterRequest(
                node_id="peer-2", host="10.0.0.5", port=8100,
            )))
            res = asyncio.run(list_nodes(service=fs, network=network, saga=saga))
            ids = [n["node_id"] for n in res["nodes"]]
            self.assertIn("peer-2", ids)
            peer = next(n for n in res["nodes"] if n["node_id"] == "peer-2")
            self.assertFalse(peer["self"])


class TestFolderZip(unittest.TestCase):
    def test_build_dir_zip_contains_files(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "sub").mkdir()
            (home / "a.txt").write_text("alpha", encoding="utf-8")
            (home / "sub" / "b.txt").write_text("beta", encoding="utf-8")
            _, fs, _ = _ctx(home)
            zip_path, name = fs.build_dir_zip("")
            try:
                self.assertTrue(name.endswith(".zip"))
                with zipfile.ZipFile(zip_path) as zf:
                    names = set(zf.namelist())
                self.assertIn("a.txt", names)
                self.assertIn("sub/b.txt", names)
            finally:
                zip_path.unlink(missing_ok=True)

    def test_build_dir_zip_rejects_file(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "a.txt").write_text("alpha", encoding="utf-8")
            _, fs, _ = _ctx(home)
            from yggdrasil.node.exceptions import ForbiddenError
            with self.assertRaises(ForbiddenError):
                fs.build_dir_zip("a.txt")


if __name__ == "__main__":
    unittest.main()
