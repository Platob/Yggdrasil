from __future__ import annotations

import os
from pathlib import Path

import pytest

from yggdrasil.pickle.ser.paths import (
    MAX_INLINE_DIR_BYTES,
    PathSerialized,
    _current_os_name,
    _dir_total_bytes,
    _extract_zip_bytes_to_temp_dir,
    _metadata_merge,
    _rebuild_path,
    _should_exclude_path,
    _write_file_bytes_to_temp_path,
    _zip_directory_to_bytes,
)


def test_metadata_merge_none():
    assert _metadata_merge(None, None) is None


def test_metadata_merge_base_only():
    out = _metadata_merge({b"a": b"1"}, None)
    assert out == {b"a": b"1"}


def test_metadata_merge_extra_only():
    out = _metadata_merge(None, {b"b": b"2"})
    assert out == {b"b": b"2"}


def test_metadata_merge_both_extra_wins():
    out = _metadata_merge({b"a": b"1", b"b": b"2"}, {b"b": b"x", b"c": b"3"})
    assert out == {b"a": b"1", b"b": b"x", b"c": b"3"}


def test_current_os_name_is_supported():
    assert _current_os_name() in {b"windows", b"posix"}


def test_dir_total_bytes_counts_only_files(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_bytes(b"abc")
    (root / "sub").mkdir()
    (root / "sub" / "b.bin").write_bytes(b"12345")

    assert _dir_total_bytes(root) == 8


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (("__pycache__", "x.pyc"), True),
        ((".git", "config"), True),
        (("node_modules", "left-pad", "index.js"), True),
        (("target", "debug", "app"), True),
        (("pkg.egg-info", "PKG-INFO"), True),
        (("pkg.dist-info", "METADATA"), True),
        (("foo.pyc",), True),
        (("foo.pyo",), True),
        (("foo.pyd",), True),
        (("foo.so",), True),
        (("foo.dll",), True),
        (("src", "main.py"), False),
        (("data", "part-000.parquet"), False),
    ],
)
def test_should_exclude_path(parts: tuple[str, ...], expected: bool):
    assert _should_exclude_path(parts) is expected


def test_zip_directory_to_bytes_excludes_expected_entries(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()

    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('ok')\n")

    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"junk")

    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n")

    (root / "node_modules").mkdir()
    (root / "node_modules" / "mod.js").write_text("module.exports = 1\n")

    (root / "pkg.egg-info").mkdir()
    (root / "pkg.egg-info" / "PKG-INFO").write_text("metadata\n")

    payload = _zip_directory_to_bytes(root)
    assert payload

    extracted = _extract_zip_bytes_to_temp_dir(payload, dirname="restored")

    assert (extracted / "src" / "main.py").exists()
    assert not (extracted / "__pycache__").exists()
    assert not (extracted / ".git").exists()
    assert not (extracted / "node_modules").exists()
    assert not (extracted / "pkg.egg-info").exists()


def test_extract_zip_bytes_to_temp_dir_roundtrip(tmp_path: Path):
    root = tmp_path / "dir"
    root.mkdir()
    (root / "a.txt").write_text("hello")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("world")

    payload = _zip_directory_to_bytes(root)
    out = _extract_zip_bytes_to_temp_dir(payload, dirname="roundtrip")

    assert (out / "a.txt").read_text() == "hello"
    assert (out / "sub" / "b.txt").read_text() == "world"


def test_write_file_bytes_to_temp_path(tmp_path: Path):
    out = _write_file_bytes_to_temp_path(b"abc123", filename="x.parquet")

    assert out.exists()
    assert out.suffix == ".parquet"
    assert out.read_bytes() == b"abc123"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("relative/path/file.txt", Path("relative/path/file.txt")),
        ("./local/file.txt", Path("./local/file.txt")),
    ],
)
def test_rebuild_path_same_os_roundtrip(raw: str, expected: Path):
    source_os = _current_os_name()
    assert _rebuild_path(raw, source_os=source_os) == expected


def test_rebuild_path_windows_to_posix_relative(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(os, "name", "posix")
    out = _rebuild_path(r"foo\bar\baz.txt", source_os=b"windows")
    assert str(out).replace("\\", "/") == "foo/bar/baz.txt"



def test_rebuild_path_windows_to_posix_absolute_drive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(os, "name", "posix")
    out = _rebuild_path(r"C:\tmp\folder\file.txt", source_os=b"windows")
    assert str(out).replace("\\", "/") == "/c/tmp/folder/file.txt"


def test_rebuild_path_posix_to_windows_relative(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(os, "name", "nt")
    out = _rebuild_path("foo/bar/baz.txt", source_os=b"posix")
    assert out == Path("foo/bar/baz.txt")


def test_rebuild_path_posix_to_windows_absolute(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(os, "name", "nt")
    out = _rebuild_path("/var/tmp/file.txt", source_os=b"posix")
    assert str(out) in {r"\var\tmp\file.txt", "/var/tmp/file.txt"}


def test_path_serialized_plain_path_same_os(tmp_path: Path):
    path = tmp_path / "some" / "place" / "file.txt"

    ser = PathSerialized.from_value(path)
    restored = ser.as_python()

    assert restored == Path(str(path))


def test_path_serialized_file_roundtrip(tmp_path: Path):
    src = tmp_path / "payload.bin"
    src.write_bytes(b"payload-bytes")

    ser = PathSerialized.from_value(src)
    restored = ser.as_python()

    assert restored.exists()
    assert restored.read_bytes() == b"payload-bytes"
    assert restored.name != src.name or restored != src


def test_path_serialized_small_dir_roundtrip(tmp_path: Path):
    root = tmp_path / "small_dir"
    root.mkdir()
    (root / "a.txt").write_text("A")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("B")

    ser = PathSerialized.from_value(root)
    restored = ser.as_python()

    assert restored.exists()
    assert restored.is_dir()
    assert (restored / "a.txt").read_text() == "A"
    assert (restored / "sub" / "b.txt").read_text() == "B"


def test_path_serialized_small_dir_excludes_noise(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('ok')\n")

    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "junk.pyc").write_bytes(b"x")

    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "state").write_text("cache")

    ser = PathSerialized.from_value(root)
    restored = ser.as_python()

    assert (restored / "src" / "main.py").exists()
    assert not (restored / "__pycache__").exists()
    assert not (restored / ".pytest_cache").exists()


def test_path_serialized_large_dir_falls_back_to_path_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "large_dir"
    root.mkdir()
    (root / "big.bin").write_bytes(b"x" * 8)

    monkeypatch.setattr(
        "yggdrasil.pickle.ser.paths._dir_total_bytes",
        lambda _: MAX_INLINE_DIR_BYTES + 1,
    )

    ser = PathSerialized.from_value(root)
    restored = ser.as_python()

    assert restored == Path(str(root))


def test_path_serialized_path_mode_cross_os_windows_to_posix(tmp_path: Path):
    ser = PathSerialized.build(
        tag=PathSerialized.TAG,
        data=r"C:\tmp\folder\file.txt".encode("utf-8"),
        metadata={
            b"path_mode": b"path",
            b"path_os": b"windows",
        },
        codec=None,
    )

    restored = ser.as_python()

    if os.name == "posix":
        assert restored == Path("/c/tmp/folder/file.txt")
    else:
        # On Windows, same-family restore returns the original semantics.
        assert isinstance(restored, Path)


def test_path_serialized_path_mode_cross_os_posix_to_windows_shape():
    ser = PathSerialized.build(
        tag=PathSerialized.TAG,
        data=b"/var/tmp/file.txt",
        metadata={
            b"path_mode": b"path",
            b"path_os": b"posix",
        },
        codec=None,
    )

    restored = ser.as_python()
    assert isinstance(restored, Path)


def test_path_serialized_from_python_object_non_path_returns_none():
    assert PathSerialized.from_python_object("not-a-path") is None


def test_path_serialized_from_python_object_path_returns_serialized(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("hello")

    ser = PathSerialized.from_python_object(p)

    assert ser is not None
    assert isinstance(ser, PathSerialized)


def test_path_serialized_unknown_mode_raises():
    ser = PathSerialized.build(
        tag=PathSerialized.TAG,
        data=b"abc",
        metadata={b"path_mode": b"wat"},
        codec=None,
    )

    with pytest.raises(ValueError, match="Unsupported PATH payload mode"):
        _ = ser.as_python()
