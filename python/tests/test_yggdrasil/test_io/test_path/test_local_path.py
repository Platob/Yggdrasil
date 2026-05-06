"""Behavior tests for :class:`yggdrasil.io.path.local_path.LocalPath`.

`LocalPath` is the local-filesystem :class:`Path` — fd-backed,
URL-addressed, holder-flavored. The contract under test:

* construction shapes — bare path string, ``pathlib.Path`` input,
  URL input, and the bare ``LocalPath()`` staging-file mint;
* lifecycle — opening / closing the long-lived fd, reuse across
  ``with``-blocks, transient fd for casual ``read_bytes`` /
  ``write_bytes`` calls when the holder isn't acquired;
* filesystem surface — :meth:`exists` / :meth:`is_file` /
  :meth:`is_dir` / :meth:`mkdir` / :meth:`unlink` / :meth:`remove`;
* pure-path API delegated to :class:`URL` — :attr:`name`,
  :attr:`stem`, :attr:`suffix`, :meth:`with_name`, :meth:`with_suffix`,
  :meth:`joinpath` / ``/``;
* ``open(mode)`` returns a :class:`BytesIO` over the path, and the
  ``with`` block commits writes to disk.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.url import URL


class TestConstruction:

    def test_string_path(self, tmp_path) -> None:
        target = str(tmp_path / "out.bin")
        lp = LocalPath(target)
        assert lp.os_path == target
        assert lp.is_local_path

    def test_pathlib_path(self, tmp_path) -> None:
        p = pathlib.Path(tmp_path / "p.bin")
        lp = LocalPath(p)
        assert lp.os_path == os.fspath(p)

    def test_url_input(self, tmp_path) -> None:
        url = URL.from_(tmp_path / "u.bin")
        lp = LocalPath(url=url)
        assert os.fspath(url) == lp.os_path

    def test_bare_constructor_mints_staging(self) -> None:
        lp = LocalPath()
        assert "yggdrasil-staging" in lp.os_path
        assert lp.temporary

    def test_staging_path_classmethod(self) -> None:
        lp = LocalPath.staging_path()
        assert lp.temporary
        assert "yggdrasil-staging" in lp.os_path

    def test_holder_dispatch_rejects_unknown_scheme(self) -> None:
        from yggdrasil.io.holder import Holder
        with pytest.raises(ValueError, match="Unknown scheme"):
            Holder(url="madeup://nowhere")


class TestStat:

    def test_missing_file_reads_zero(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "absent"))
        assert lp.size == 0
        assert not lp.exists()
        assert not lp.is_file()
        assert not lp.is_dir()

    def test_existing_file(self, tmp_path) -> None:
        p = tmp_path / "x"
        p.write_bytes(b"abc")
        lp = LocalPath(str(p))
        assert lp.exists()
        assert lp.is_file()
        assert lp.size == 3

    def test_directory_kind(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path))
        assert lp.is_dir()
        assert not lp.is_file()


class TestReadWriteCycle:
    """`read_bytes` / `write_bytes` / ``size`` work on closed holders too."""

    def test_write_creates_file(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "new.txt"))
        n = lp.write_bytes(b"hello")
        assert n == 5
        assert lp.read_bytes() == b"hello"
        assert lp.size == 5

    def test_overwrite_keeps_size(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "x.bin"))
        lp.write_bytes(b"abcdef")
        lp.pwrite(b"ZZ", 2)
        assert lp.read_bytes() == b"abZZef"

    def test_truncate_shrinks(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "x.bin"))
        lp.write_bytes(b"abcdef")
        lp.truncate(3)
        assert lp.size == 3
        assert lp.read_bytes() == b"abc"

    def test_clear_unlinks(self, tmp_path) -> None:
        p = tmp_path / "x.bin"
        p.write_bytes(b"data")
        lp = LocalPath(str(p))
        lp.clear()
        assert not p.exists()


class TestOpenReturnsBytesIO:

    def test_open_default_returns_bytes_io(self, tmp_path) -> None:
        p = tmp_path / "x.bin"
        p.write_bytes(b"hello")
        lp = LocalPath(str(p))
        with lp.open("rb") as bio:
            assert isinstance(bio, BytesIO)
            assert bio.read() == b"hello"

    def test_open_writes_commit_to_disk(self, tmp_path) -> None:
        target = str(tmp_path / "out.bin")
        lp = LocalPath(target)
        with lp.open("wb") as bio:
            bio.write(b"new-bytes")
        assert pathlib.Path(target).read_bytes() == b"new-bytes"

    def test_open_with_mode_str_alias(self, tmp_path) -> None:
        target = str(tmp_path / "x.bin")
        lp = LocalPath(target)
        with lp.open("write") as bio:
            bio.write(b"alias-mode")
        assert lp.read_bytes() == b"alias-mode"

    def test_append_mode_lands_at_eof(self, tmp_path) -> None:
        p = tmp_path / "log"
        p.write_bytes(b"first\n")
        lp = LocalPath(str(p))
        with lp.open("ab") as bio:
            bio.write(b"second\n")
        assert lp.read_bytes() == b"first\nsecond\n"


class TestPathApiDelegation:
    """Pure-path manipulation flows through :class:`URL`."""

    def test_name_and_stem(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "trades.csv"))
        assert lp.name == "trades.csv"
        assert lp.stem == "trades"
        assert lp.suffix == ".csv"

    def test_with_name(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "a.bin"))
        renamed = lp.with_name("b.bin")
        assert renamed.name == "b.bin"

    def test_with_suffix(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "trades.csv"))
        assert lp.with_suffix(".parquet").suffix == ".parquet"

    def test_with_invalid_suffix_raises(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "trades.csv"))
        with pytest.raises(ValueError, match="must start with '.'"):
            lp.with_suffix("parquet")

    def test_joinpath(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path)) / "child" / "deeper.bin"
        assert lp.name == "deeper.bin"
        assert "child" in lp.parts


class TestMkdirRemove:

    def test_mkdir_creates_dir(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "deep" / "nested"))
        target.mkdir()
        assert target.is_dir()

    def test_unlink_file(self, tmp_path) -> None:
        p = tmp_path / "x.bin"
        p.write_bytes(b"x")
        lp = LocalPath(str(p))
        lp.unlink()
        assert not p.exists()

    def test_unlink_missing_ok_default(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "absent"))
        # missing_ok=True by default — doesn't raise.
        lp.unlink()

    def test_unlink_missing_strict_raises(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError):
            lp.unlink(missing_ok=False)

    def test_unlink_directory_is_typeerror(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path))
        with pytest.raises(IsADirectoryError):
            lp.unlink()

    def test_remove_directory(self, tmp_path) -> None:
        d = tmp_path / "nest"
        d.mkdir()
        (d / "a.txt").write_text("hi")
        lp = LocalPath(str(d))
        lp.remove(recursive=True)
        assert not d.exists()


class TestIterdir:

    def test_lists_children(self, tmp_path) -> None:
        for name in ("a.txt", "b.txt", "c.txt"):
            (tmp_path / name).write_text("x")
        lp = LocalPath(str(tmp_path))
        names = sorted(child.name for child in lp.iterdir())
        assert names == ["a.txt", "b.txt", "c.txt"]

    def test_iterdir_on_missing_is_empty(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "absent"))
        assert list(lp.iterdir()) == []


class TestStagingTemporary:

    def test_anonymous_holder_clears_on_close(self) -> None:
        lp = LocalPath()
        path = lp.os_path
        with lp.open("wb") as bio:
            bio.write(b"throwaway")
        # close() runs the temporary clear.
        assert not pathlib.Path(path).exists()

    def test_directory_url_stages_child(self, tmp_path) -> None:
        # Trailing-slash URL → directory; LocalPath stages a child.
        # The staging file is owned by the holder (temporary=True) and
        # gets cleared on close — assert the URL was rebound under
        # tmp_path, not whether the file persists.
        target = LocalPath(str(tmp_path) + "/")
        with target.open("wb") as bio:
            bio.write(b"staged-bytes")
        assert target.os_path.startswith(str(tmp_path))
        assert target.temporary


class TestTouch:

    def test_touch_creates_empty_file(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "fresh.bin"))
        assert not lp.exists()
        lp.touch()
        assert lp.exists() and lp.size == 0
