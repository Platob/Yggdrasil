"""Tests for :class:`yggdrasil.io.nested.folder_path.FolderPath`."""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.nested.folder_path import FolderPath


class TestRegistration:

    def test_folder_inherits_holder(self) -> None:
        assert issubclass(FolderPath, Holder)

    def test_folder_in_registry(self) -> None:
        from yggdrasil.data.enums import MimeTypes

        assert Holder.class_for_media_type(MimeTypes.FOLDER) is FolderPath


class TestConstruction:

    def test_string_path(self, tmp_path) -> None:
        folder = FolderPath(path=str(tmp_path))
        assert folder.path is not None

    def test_pathlib_path(self, tmp_path) -> None:
        folder = FolderPath(path=pathlib.Path(tmp_path))
        assert folder.path is not None

    def test_missing_path_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a path"):
            FolderPath()


class TestByteOpsRaise:
    """A folder is a directory — byte primitives raise."""

    def test_read_mv_raises(self, tmp_path) -> None:
        folder = FolderPath(path=str(tmp_path))
        with pytest.raises(NotImplementedError, match="directory"):
            folder._read_mv(1, 0)

    def test_write_mv_raises(self, tmp_path) -> None:
        folder = FolderPath(path=str(tmp_path))
        with pytest.raises(NotImplementedError, match="directory"):
            folder._write_mv(memoryview(b"x"), 0)

    def test_size_is_zero(self, tmp_path) -> None:
        folder = FolderPath(path=str(tmp_path))
        assert folder.size == 0


class TestIterChildren:

    def test_skips_dotfiles(self, tmp_path) -> None:
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.parquet").write_bytes(b"")
        folder = FolderPath(path=str(tmp_path))
        # Child URLs/paths come through the leaf's bound holder.
        seen = []
        for c in folder.iter_children():
            parent = c.parent if hasattr(c, "parent") and c.parent is not None else c
            name = parent.url.name if hasattr(parent, "url") else None
            seen.append(name)
        assert ".hidden" not in seen

    def test_recurses_into_subdirectories(self, tmp_path) -> None:
        (tmp_path / "sub").mkdir()
        folder = FolderPath(path=str(tmp_path))
        children = list(folder.iter_children())
        assert any(isinstance(c, FolderPath) for c in children)

    def test_missing_folder_yields_empty(self, tmp_path) -> None:
        folder = FolderPath(path=str(tmp_path / "absent"))
        assert list(folder.iter_children()) == []


class TestRoundTrip:

    def test_write_then_read_arrow(self, tmp_path) -> None:
        folder = FolderPath(path=str(tmp_path))
        table = pa.table({"id": [1, 2, 3]})
        folder.write_arrow_table(table)
        # At least one part file should land on disk.
        parts = [p for p in tmp_path.iterdir() if p.is_file()]
        assert len(parts) >= 1
        # And reading it back returns the same data.
        got = folder.read_arrow_table()
        assert got.column("id").to_pylist() == [1, 2, 3]


class TestMediaTypeMetadata:
    """``FolderPath._persist_schema`` stamps ``Field.media_type``."""

    def test_in_memory_schema_carries_media_type(self, tmp_path) -> None:
        from yggdrasil.data.enums.media_type import MediaTypes
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        batch = pa.record_batch([pa.array([1, 2])], names=["id"])
        folder.write_arrow_batches((batch,), options=FolderOptions())
        # Default child media type is Arrow IPC — schema should
        # report it after a write.
        assert folder.collect_schema().media_type == MediaTypes.ARROW_IPC

    def test_sidecar_round_trips_media_type(self, tmp_path) -> None:
        from yggdrasil.data.enums.media_type import MediaTypes
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        batch = pa.record_batch([pa.array([1, 2])], names=["id"])
        folder.write_arrow_batches(
            (batch,),
            options=FolderOptions(child_media_type=MediaTypes.PARQUET),
        )
        # Drop the in-memory singleton so the next read forces a
        # sidecar load, then confirm the media-type stamp survived.
        FolderPath._INSTANCES.clear()
        reopened = FolderPath(path=str(tmp_path))
        assert reopened.collect_schema().media_type == MediaTypes.PARQUET

    def test_no_media_type_when_never_persisted(self, tmp_path) -> None:
        folder = FolderPath(path=str(tmp_path))
        # No write — schema falls back to empty / inferred, no media
        # type was ever stamped.
        assert folder.collect_schema().media_type is None
