"""Tests for :class:`yggdrasil.io.nested.folder_io.FolderIO`."""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.nested.folder_io import FolderIO


class TestRegistration:

    def test_folder_inherits_holder(self) -> None:
        assert issubclass(FolderIO, Holder)

    def test_folder_in_registry(self) -> None:
        from yggdrasil.data.enums import MimeTypes

        assert Holder.class_for_media_type(MimeTypes.FOLDER) is FolderIO


class TestConstruction:

    def test_string_path(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        assert folder.path is not None

    def test_pathlib_path(self, tmp_path) -> None:
        folder = FolderIO(path=pathlib.Path(tmp_path))
        assert folder.path is not None

    def test_missing_path_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a path"):
            FolderIO()


class TestByteOpsRaise:
    """A folder is a directory — byte primitives raise."""

    def test_read_mv_raises(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        with pytest.raises(NotImplementedError, match="directory"):
            folder._read_mv(1, 0)

    def test_write_mv_raises(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        with pytest.raises(NotImplementedError, match="directory"):
            folder._write_mv(memoryview(b"x"), 0)

    def test_size_is_zero(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        assert folder.size == 0


class TestIterChildren:

    def test_skips_dotfiles(self, tmp_path) -> None:
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.parquet").write_bytes(b"")
        folder = FolderIO(path=str(tmp_path))
        # Child URLs/paths come through the leaf's bound holder.
        seen = []
        for c in folder.iter_children():
            parent = c.parent if hasattr(c, "parent") and c.parent is not None else c
            name = parent.url.name if hasattr(parent, "url") else None
            seen.append(name)
        assert ".hidden" not in seen

    def test_recurses_into_subdirectories(self, tmp_path) -> None:
        (tmp_path / "sub").mkdir()
        folder = FolderIO(path=str(tmp_path))
        children = list(folder.iter_children())
        assert any(isinstance(c, FolderIO) for c in children)

    def test_missing_folder_yields_empty(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path / "absent"))
        assert list(folder.iter_children()) == []


class TestRoundTrip:

    def test_write_then_read_arrow(self, tmp_path) -> None:
        folder = FolderIO(path=str(tmp_path))
        table = pa.table({"id": [1, 2, 3]})
        folder.write_arrow_table(table)
        # At least one part file should land on disk.
        parts = [p for p in tmp_path.iterdir() if p.is_file()]
        assert len(parts) >= 1
        # And reading it back returns the same data.
        got = folder.read_arrow_table()
        assert got.column("id").to_pylist() == [1, 2, 3]
