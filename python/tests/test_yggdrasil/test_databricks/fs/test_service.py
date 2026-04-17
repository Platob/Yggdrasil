from __future__ import annotations

from unittest.mock import Mock

import pytest

import yggdrasil.databricks.fs.service as module_under_test
from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.fs.service import FileSystem


@pytest.fixture
def service() -> FileSystem:
    return FileSystem(client=DatabricksClient(host="https://adb-123.databricks.com"))


def test_client_exposes_cached_filesystem_service():
    client = DatabricksClient(host="https://adb-123.databricks.com")

    fs1 = client.filesystem
    fs2 = client.filesystem

    assert isinstance(fs1, FileSystem)
    assert fs1 is fs2
    assert client.fs is fs1


def test_path_uses_databricks_path_parse(service, monkeypatch):
    expected = object()
    parse = Mock(return_value=expected)
    monkeypatch.setattr(module_under_test.DatabricksPath, "parse", parse)

    got = service.path("/dbfs/tmp/demo.txt", temporary=True)

    assert got is expected
    parse.assert_called_once_with(
        "/dbfs/tmp/demo.txt",
        client=service.client,
        temporary=True,
    )


def test_open_delegates_with_builtin_open_shape(service, monkeypatch):
    path = Mock()
    path.open.return_value = "HANDLE"
    calls = []

    def fake_path(self, arg, **kwargs):
        calls.append((arg, kwargs))
        return path

    monkeypatch.setattr(FileSystem, "path", fake_path)

    got = service.open("/dbfs/tmp/demo.txt", mode="rt", encoding="utf-8", newline="\n")

    assert got == "HANDLE"
    assert calls == [("/dbfs/tmp/demo.txt", {})]
    path.open.assert_called_once_with(
        mode="rt",
        buffering=-1,
        encoding="utf-8",
        errors=None,
        newline="\n",
    )


def test_listdir_returns_child_names(service, monkeypatch):
    child_a = Mock()
    child_a.name = "a.txt"
    child_b = Mock()
    child_b.name = "nested"
    path = Mock()
    path.ls.return_value = [child_a, child_b]
    calls = []

    def fake_path(self, arg, **kwargs):
        calls.append((arg, kwargs))
        return path

    monkeypatch.setattr(FileSystem, "path", fake_path)

    got = service.listdir("/dbfs/tmp")

    assert got == ["a.txt", "nested"]
    assert calls == [("/dbfs/tmp", {})]
    path.ls.assert_called_once_with(recursive=False, allow_not_found=True)


def test_makedirs_uses_parent_creation_semantics(service, monkeypatch):
    path = Mock()
    expected = Mock()
    path.mkdir.return_value = expected
    calls = []

    def fake_path(self, arg, **kwargs):
        calls.append((arg, kwargs))
        return path

    monkeypatch.setattr(FileSystem, "path", fake_path)

    got = service.makedirs("/dbfs/tmp/a/b", exist_ok=True)

    assert got is expected
    assert calls == [("/dbfs/tmp/a/b", {})]
    path.mkdir.assert_called_once_with(mode=0o777, parents=True, exist_ok=True)


def test_remove_targets_files_only(service, monkeypatch):
    path = Mock()
    calls = []

    def fake_path(self, arg, **kwargs):
        calls.append((arg, kwargs))
        return path

    monkeypatch.setattr(FileSystem, "path", fake_path)

    service.remove("/dbfs/tmp/file.txt", missing_ok=True)

    assert calls == [("/dbfs/tmp/file.txt", {})]
    path.rmfile.assert_called_once_with(allow_not_found=True)


def test_copy_and_rename_delegate(service, monkeypatch):
    src_for_copy = Mock()
    dst_for_copy = Mock()
    src_for_rename = Mock()
    dst_for_rename = Mock()
    calls = []
    returns = iter([src_for_copy, dst_for_copy, src_for_rename, dst_for_rename])

    def fake_path(self, arg, **kwargs):
        calls.append((arg, kwargs))
        return next(returns)

    monkeypatch.setattr(FileSystem, "path", fake_path)

    copied = service.copy("/dbfs/src.txt", "/dbfs/dst.txt")
    renamed = service.rename("/dbfs/src.txt", "/dbfs/dst.txt")

    assert copied is dst_for_copy
    assert renamed == src_for_rename.rename.return_value
    assert calls == [
        ("/dbfs/src.txt", {}),
        ("/dbfs/dst.txt", {}),
        ("/dbfs/src.txt", {}),
        ("/dbfs/dst.txt", {}),
    ]
    src_for_copy.copy_to.assert_called_once_with(dst_for_copy)
    src_for_rename.rename.assert_called_once_with(dst_for_rename)


def test_read_write_helpers_delegate(service, monkeypatch):
    path = Mock()
    path.read_bytes.return_value = b"abc"
    path.read_text.return_value = "abc"
    calls = []

    def fake_path(self, arg, **kwargs):
        calls.append((arg, kwargs))
        return path

    monkeypatch.setattr(FileSystem, "path", fake_path)

    assert service.read_bytes("/dbfs/a") == b"abc"
    assert service.read_text("/dbfs/a") == "abc"
    assert service.write_bytes("/dbfs/a", b"xyz") is path
    assert service.write_text("/dbfs/a", "xyz") is path

    assert calls == [
        ("/dbfs/a", {}),
        ("/dbfs/a", {}),
        ("/dbfs/a", {}),
        ("/dbfs/a", {}),
    ]
    path.read_bytes.assert_called_once_with(use_cache=False)
    path.read_text.assert_called_once_with(encoding="utf-8", errors=None)
    path.write_bytes.assert_called_once_with(b"xyz")
    path.write_text.assert_called_once_with("xyz", encoding="utf-8", errors=None, newline=None)


def test_walk_groups_children_by_parent(service, monkeypatch):
    root = Mock()
    root.full_path.return_value = "/dbfs/tmp/root"

    dir_path = Mock()
    dir_path.full_path.return_value = "/dbfs/tmp/root/sub"
    dir_path.parent = root
    dir_path.is_dir.return_value = True

    file_path = Mock()
    file_path.full_path.return_value = "/dbfs/tmp/root/file.txt"
    file_path.parent = root
    file_path.is_dir.return_value = False

    nested_file = Mock()
    nested_file.full_path.return_value = "/dbfs/tmp/root/sub/nested.txt"
    nested_file.parent = dir_path
    nested_file.is_dir.return_value = False

    root.ls.return_value = [dir_path, file_path, nested_file]
    calls = []

    def fake_path(self, arg, **kwargs):
        calls.append((arg, kwargs))
        return root

    monkeypatch.setattr(FileSystem, "path", fake_path)

    walked = list(service.walk("/dbfs/tmp/root"))

    assert calls == [("/dbfs/tmp/root", {})]
    assert walked[0] == (root, [dir_path], [file_path])
    assert walked[1] == (dir_path, [], [nested_file])
