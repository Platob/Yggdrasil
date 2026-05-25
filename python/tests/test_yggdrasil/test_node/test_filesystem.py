"""Tests for filesystem endpoints and service."""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.config import Settings


@pytest.fixture
def tmp_home(tmp_path):
    return tmp_path / "ygg_home"


@pytest.fixture
def settings(tmp_home):
    return Settings(
        node_home=tmp_home,
        node_id="test-node-001",
    )


@pytest.fixture
def client(settings):
    app = create_app(settings)
    return TestClient(app)


class TestFilesystemListDir:
    def test_list_root_has_defaults(self, client):
        resp = client.get("/api/fs/ls")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "test-node-001"
        names = [e["name"] for e in data["entries"]]
        for d in ("data", "cache", "logs"):
            assert d in names, f"{d} missing from root listing"

    def test_list_with_files(self, client, settings):
        # Create some files in a subdirectory of node home
        fs_root = settings.node_home
        test_dir = fs_root / "workspace"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "hello.txt").write_text("hi")
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "nested.txt").write_text("nested")

        resp = client.get("/api/fs/ls", params={"path": "workspace"})
        assert resp.status_code == 200
        data = resp.json()
        entries = data["entries"]
        assert len(entries) == 2
        # Directories should come first
        assert entries[0]["name"] == "subdir"
        assert entries[0]["is_dir"] is True
        assert entries[1]["name"] == "hello.txt"
        assert entries[1]["is_dir"] is False

    def test_list_subdirectory(self, client, settings):
        fs_root = settings.node_home
        fs_root.mkdir(parents=True, exist_ok=True)
        (fs_root / "mydir").mkdir()
        (fs_root / "mydir" / "a.txt").write_text("aaa")

        resp = client.get("/api/fs/ls", params={"path": "mydir"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == "mydir"
        assert len(data["entries"]) == 1
        assert data["entries"][0]["name"] == "a.txt"

    def test_list_nonexistent(self, client):
        resp = client.get("/api/fs/ls", params={"path": "noexist"})
        assert resp.status_code == 404

    def test_list_traversal_blocked(self, client):
        resp = client.get("/api/fs/ls", params={"path": "../../etc"})
        assert resp.status_code == 403


class TestFilesystemReadWrite:
    def test_write_and_read_text(self, client):
        # Write
        resp = client.post("/api/fs/write", json={
            "path": "docs/readme.txt",
            "content": "Hello Yggdrasil!",
        })
        assert resp.status_code == 200
        info = resp.json()
        assert info["name"] == "readme.txt"
        assert info["is_dir"] is False
        assert info["size"] > 0

        # Read
        resp = client.get("/api/fs/read", params={"path": "docs/readme.txt"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Hello Yggdrasil!"
        assert data["encoding"] == "utf-8"
        assert data["size"] == 16

    def test_write_and_read_binary(self, client):
        binary_data = bytes(range(256))
        encoded = base64.b64encode(binary_data).decode("ascii")

        resp = client.post("/api/fs/write", json={
            "path": "data/binary.bin",
            "content": encoded,
            "encoding": "base64",
        })
        assert resp.status_code == 200

        resp = client.get("/api/fs/read", params={"path": "data/binary.bin"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["encoding"] == "base64"
        assert base64.b64decode(data["content"]) == binary_data

    def test_write_no_mkdir(self, client):
        resp = client.post("/api/fs/write", json={
            "path": "nonexist/deep/file.txt",
            "content": "x",
            "mkdir": False,
        })
        assert resp.status_code == 404

    def test_read_nonexistent(self, client):
        resp = client.get("/api/fs/read", params={"path": "nope.txt"})
        assert resp.status_code == 404

    def test_read_traversal_blocked(self, client):
        resp = client.get("/api/fs/read", params={"path": "../../../etc/passwd"})
        assert resp.status_code == 403


class TestFilesystemDelete:
    def test_delete_file(self, client, settings):
        fs_root = settings.node_home
        fs_root.mkdir(parents=True, exist_ok=True)
        (fs_root / "doomed.txt").write_text("bye")

        resp = client.delete("/api/fs/delete", params={"path": "doomed.txt"})
        assert resp.status_code == 204
        assert not (fs_root / "doomed.txt").exists()

    def test_delete_directory(self, client, settings):
        fs_root = settings.node_home
        d = fs_root / "dir_to_rm"
        d.mkdir(parents=True, exist_ok=True)
        (d / "inner.txt").write_text("x")

        resp = client.delete("/api/fs/delete", params={"path": "dir_to_rm"})
        assert resp.status_code == 204
        assert not d.exists()

    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/fs/delete", params={"path": "ghost.txt"})
        assert resp.status_code == 404


class TestFilesystemMove:
    def test_move_file(self, client, settings):
        fs_root = settings.node_home
        fs_root.mkdir(parents=True, exist_ok=True)
        (fs_root / "src.txt").write_text("moving")

        resp = client.post("/api/fs/move", json={
            "source": "src.txt",
            "destination": "dest.txt",
        })
        assert resp.status_code == 200
        info = resp.json()
        assert info["name"] == "dest.txt"
        assert not (fs_root / "src.txt").exists()
        assert (fs_root / "dest.txt").read_text() == "moving"

    def test_move_nonexistent(self, client):
        resp = client.post("/api/fs/move", json={
            "source": "nope.txt",
            "destination": "also_nope.txt",
        })
        assert resp.status_code == 404


class TestFilesystemMkdir:
    def test_mkdir(self, client, settings):
        resp = client.post("/api/fs/mkdir", params={"path": "new/nested/dir"})
        assert resp.status_code == 200
        info = resp.json()
        assert info["is_dir"] is True
        assert info["name"] == "dir"

        fs_root = settings.node_home
        assert (fs_root / "new" / "nested" / "dir").is_dir()


class TestFilesystemStat:
    def test_stat_file(self, client, settings):
        fs_root = settings.node_home
        fs_root.mkdir(parents=True, exist_ok=True)
        (fs_root / "info.txt").write_text("data")

        resp = client.get("/api/fs/stat", params={"path": "info.txt"})
        assert resp.status_code == 200
        info = resp.json()
        assert info["name"] == "info.txt"
        assert info["is_dir"] is False
        assert info["size"] == 4
        assert info["modified_at"] != ""

    def test_stat_nonexistent(self, client):
        resp = client.get("/api/fs/stat", params={"path": "ghost"})
        assert resp.status_code == 404


class TestFilesystemStream:
    def test_stream_download(self, client, settings):
        fs_root = settings.node_home
        fs_root.mkdir(parents=True, exist_ok=True)
        content = b"streaming content " * 100
        (fs_root / "big.bin").write_bytes(content)

        resp = client.get("/api/fs/stream", params={"path": "big.bin"})
        assert resp.status_code == 200
        assert resp.content == content
        assert "attachment" in resp.headers.get("content-disposition", "")

    def test_stream_upload(self, client, settings):
        payload = b"uploaded chunk data"
        resp = client.post(
            "/api/fs/upload",
            params={"path": "uploaded.bin"},
            content=payload,
        )
        assert resp.status_code == 200
        info = resp.json()
        assert info["name"] == "uploaded.bin"

        fs_root = settings.node_home
        assert (fs_root / "uploaded.bin").read_bytes() == payload


class TestFilesystemSecurity:
    """Ensure path traversal is blocked in all endpoints."""

    def test_write_traversal(self, client):
        resp = client.post("/api/fs/write", json={
            "path": "../../evil.txt",
            "content": "pwned",
        })
        assert resp.status_code == 403

    def test_move_source_traversal(self, client, settings):
        # Create a file within root to have a valid dest
        fs_root = settings.node_home
        fs_root.mkdir(parents=True, exist_ok=True)
        (fs_root / "legit.txt").write_text("ok")

        resp = client.post("/api/fs/move", json={
            "source": "../../etc/passwd",
            "destination": "stolen.txt",
        })
        assert resp.status_code == 403

    def test_delete_traversal(self, client):
        resp = client.delete("/api/fs/delete", params={"path": "../../../etc"})
        assert resp.status_code == 403

    def test_mkdir_traversal(self, client):
        resp = client.post("/api/fs/mkdir", params={"path": "../../outside"})
        assert resp.status_code == 403

    def test_stat_traversal(self, client):
        resp = client.get("/api/fs/stat", params={"path": "../../etc/passwd"})
        assert resp.status_code == 403
