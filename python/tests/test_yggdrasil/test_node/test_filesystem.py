from __future__ import annotations

import base64

import pytest


PREFIX = "/api/fs"


def test_ls_root(client):
    resp = client.get(f"{PREFIX}/ls")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert data["path"] == ""
    names = {e["name"] for e in data["entries"]}
    assert "data" in names
    assert "logs" in names
    assert "cache" in names
    assert "mirrors" in names


def test_ls_with_files(client, settings):
    (settings.node_home / "file_a.txt").write_text("aaa")
    (settings.node_home / "file_b.txt").write_text("bbb")
    resp = client.get(f"{PREFIX}/ls")
    assert resp.status_code == 200
    names = [e["name"] for e in resp.json()["entries"]]
    assert "file_a.txt" in names
    assert "file_b.txt" in names


def test_ls_subdirectory(client, settings):
    sub = settings.node_home / "data" / "files" / "tmp"
    (sub / "inner.txt").write_text("hello")
    resp = client.get(f"{PREFIX}/ls", params={"path": "data/files/tmp"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["path"] == "data/files/tmp"
    names = [e["name"] for e in data["entries"]]
    assert "inner.txt" in names


def test_ls_nonexistent_404(client):
    resp = client.get(f"{PREFIX}/ls", params={"path": "does/not/exist"})
    assert resp.status_code == 404


def test_ls_traversal_blocked_403(client):
    resp = client.get(f"{PREFIX}/ls", params={"path": "../../etc"})
    assert resp.status_code == 403


def test_write_and_read_text(client):
    resp = client.post(f"{PREFIX}/write", json={
        "path": "hello.txt",
        "content": "Hello, Yggdrasil!",
    })
    assert resp.status_code == 200
    info = resp.json()
    assert info["name"] == "hello.txt"
    assert info["is_dir"] is False
    assert info["size"] > 0

    resp = client.get(f"{PREFIX}/read", params={"path": "hello.txt"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "Hello, Yggdrasil!"
    assert data["encoding"] == "utf-8"
    assert data["size"] > 0


def test_write_and_read_binary(client):
    raw = bytes(range(256))
    encoded = base64.b64encode(raw).decode("ascii")
    resp = client.post(f"{PREFIX}/write", json={
        "path": "binary.bin",
        "content": encoded,
        "encoding": "base64",
    })
    assert resp.status_code == 200
    assert resp.json()["name"] == "binary.bin"

    resp = client.get(f"{PREFIX}/read", params={"path": "binary.bin"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["encoding"] == "base64"
    decoded = base64.b64decode(data["content"])
    assert decoded == raw


def test_write_creates_parent_dirs(client, settings):
    resp = client.post(f"{PREFIX}/write", json={
        "path": "deep/nested/dir/file.txt",
        "content": "created",
    })
    assert resp.status_code == 200
    assert (settings.node_home / "deep" / "nested" / "dir" / "file.txt").exists()


def test_write_no_mkdir_404(client):
    resp = client.post(f"{PREFIX}/write", json={
        "path": "nonexistent_parent/child.txt",
        "content": "data",
        "mkdir": False,
    })
    assert resp.status_code == 404


def test_read_nonexistent_404(client):
    resp = client.get(f"{PREFIX}/read", params={"path": "no_such_file.txt"})
    assert resp.status_code == 404


def test_read_traversal_403(client):
    resp = client.get(f"{PREFIX}/read", params={"path": "../../../etc/passwd"})
    assert resp.status_code == 403


def test_delete_file(client, settings):
    target = settings.node_home / "to_delete.txt"
    target.write_text("bye")
    assert target.exists()

    resp = client.delete(f"{PREFIX}/delete", params={"path": "to_delete.txt"})
    assert resp.status_code == 204
    assert not target.exists()


def test_delete_directory(client, settings):
    d = settings.node_home / "dir_to_delete" / "sub"
    d.mkdir(parents=True)
    (d / "file.txt").write_text("content")

    resp = client.delete(f"{PREFIX}/delete", params={"path": "dir_to_delete"})
    assert resp.status_code == 204
    assert not (settings.node_home / "dir_to_delete").exists()


def test_delete_nonexistent_404(client):
    resp = client.delete(f"{PREFIX}/delete", params={"path": "ghost.txt"})
    assert resp.status_code == 404


def test_move_file(client, settings):
    src = settings.node_home / "move_src.txt"
    src.write_text("movable")

    resp = client.post(f"{PREFIX}/move", json={
        "source": "move_src.txt",
        "destination": "moved/move_dst.txt",
    })
    assert resp.status_code == 200
    info = resp.json()
    assert info["name"] == "move_dst.txt"
    assert not src.exists()
    assert (settings.node_home / "moved" / "move_dst.txt").exists()


def test_move_nonexistent_404(client):
    resp = client.post(f"{PREFIX}/move", json={
        "source": "nowhere.txt",
        "destination": "somewhere.txt",
    })
    assert resp.status_code == 404


def test_mkdir(client, settings):
    resp = client.post(f"{PREFIX}/mkdir", params={"path": "new/nested/dir"})
    assert resp.status_code == 200
    info = resp.json()
    assert info["is_dir"] is True
    assert info["name"] == "dir"
    assert (settings.node_home / "new" / "nested" / "dir").is_dir()


def test_stat_file(client, settings):
    target = settings.node_home / "stat_target.txt"
    target.write_text("stat me")

    resp = client.get(f"{PREFIX}/stat", params={"path": "stat_target.txt"})
    assert resp.status_code == 200
    info = resp.json()
    assert info["name"] == "stat_target.txt"
    assert info["is_dir"] is False
    assert info["size"] == len("stat me")
    assert info["modified_at"] != ""
    assert info["created_at"] != ""


def test_stat_nonexistent_404(client):
    resp = client.get(f"{PREFIX}/stat", params={"path": "nonexistent.dat"})
    assert resp.status_code == 404


def test_stream_download(client, settings):
    target = settings.node_home / "download_me.txt"
    target.write_text("stream content here")

    resp = client.get(f"{PREFIX}/stream", params={"path": "download_me.txt"})
    assert resp.status_code == 200
    assert resp.content == b"stream content here"
    assert "attachment" in resp.headers.get("content-disposition", "")
    assert "download_me.txt" in resp.headers.get("content-disposition", "")


def test_stream_upload(client, settings):
    payload = b"uploaded binary data \x00\xff"
    resp = client.post(
        f"{PREFIX}/upload",
        params={"path": "uploaded.bin"},
        content=payload,
    )
    assert resp.status_code == 200
    info = resp.json()
    assert info["name"] == "uploaded.bin"
    assert (settings.node_home / "uploaded.bin").read_bytes() == payload


def test_write_traversal_403(client):
    resp = client.post(f"{PREFIX}/write", json={
        "path": "../../etc/evil.txt",
        "content": "hacked",
    })
    assert resp.status_code == 403


def test_delete_traversal_403(client):
    resp = client.delete(f"{PREFIX}/delete", params={"path": "../../etc/passwd"})
    assert resp.status_code == 403


def test_move_source_traversal_403(client):
    resp = client.post(f"{PREFIX}/move", json={
        "source": "../../../etc/passwd",
        "destination": "stolen.txt",
    })
    assert resp.status_code == 403


def test_mkdir_traversal_403(client):
    resp = client.post(f"{PREFIX}/mkdir", params={"path": "../../evil_dir"})
    assert resp.status_code == 403


def test_stat_traversal_403(client):
    resp = client.get(f"{PREFIX}/stat", params={"path": "../../etc/passwd"})
    assert resp.status_code == 403
