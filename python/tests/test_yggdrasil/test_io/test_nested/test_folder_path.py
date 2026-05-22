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


def _partitioned_batch(part_values: list[int], ids: list[int]) -> "pa.RecordBatch":
    """Build a RecordBatch whose ``pk`` column is tagged ``partition_by``.

    Drives :meth:`FolderPath._write_arrow_batches` into the partition
    split branch — one ``pk=<v>/`` directory per distinct value.
    """
    schema = pa.schema([
        pa.field("pk", pa.int64(), metadata={b"t:partition_by": b"True"}),
        pa.field("id", pa.int64()),
    ])
    return pa.record_batch(
        [pa.array(part_values, pa.int64()), pa.array(ids, pa.int64())],
        schema=schema,
    )


class TestPartitionWriteModes:
    """Mode handling on partitioned writes (Hive-layout overwrite, ignore, …).

    Bug surface: previous behaviour returned from the partition branch
    before applying mode handling at the folder level, so OVERWRITE
    left untouched partition directories on disk and IGNORE /
    ERROR_IF_EXISTS silently became "append at partition level".
    """

    def test_overwrite_clears_stale_partitions(self, tmp_path) -> None:
        from yggdrasil.data.enums import Mode
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        # Round 1: lands two partitions (pk=1, pk=2).
        folder.write_arrow_batches(
            (_partitioned_batch([1, 2], [10, 20]),),
            options=FolderOptions(mode=Mode.APPEND),
        )
        assert (tmp_path / "pk=1").is_dir()
        assert (tmp_path / "pk=2").is_dir()
        # Round 2: OVERWRITE writes a different partition (pk=3).
        # The previous pk=1 / pk=2 trees must be gone — otherwise a
        # subsequent read returns stale rows from those partitions.
        folder.write_arrow_batches(
            (_partitioned_batch([3], [30]),),
            options=FolderOptions(mode=Mode.OVERWRITE),
        )
        assert not (tmp_path / "pk=1").exists()
        assert not (tmp_path / "pk=2").exists()
        assert (tmp_path / "pk=3").is_dir()

    def test_ignore_short_circuits_when_any_partition_present(self, tmp_path) -> None:
        from yggdrasil.data.enums import Mode
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches(
            (_partitioned_batch([1], [10]),),
            options=FolderOptions(mode=Mode.APPEND),
        )
        # Second write under IGNORE must NOT land a new partition.
        folder.write_arrow_batches(
            (_partitioned_batch([2], [20]),),
            options=FolderOptions(mode=Mode.IGNORE),
        )
        assert (tmp_path / "pk=1").is_dir()
        assert not (tmp_path / "pk=2").exists()

    def test_error_if_exists_raises(self, tmp_path) -> None:
        from yggdrasil.data.enums import Mode
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches(
            (_partitioned_batch([1], [10]),),
            options=FolderOptions(mode=Mode.APPEND),
        )
        with pytest.raises(FileExistsError):
            folder.write_arrow_batches(
                (_partitioned_batch([2], [20]),),
                options=FolderOptions(mode=Mode.ERROR_IF_EXISTS),
            )


class TestClearTabularChildren:
    """``_clear_tabular_children`` must remove partition directories too."""

    def test_removes_partition_subtrees_not_just_files(self, tmp_path) -> None:
        from yggdrasil.data.enums import Mode
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches(
            (_partitioned_batch([1, 2], [10, 20]),),
            options=FolderOptions(mode=Mode.APPEND),
        )
        assert (tmp_path / "pk=1").is_dir()
        assert (tmp_path / "pk=2").is_dir()
        folder._clear_tabular_children()
        assert not (tmp_path / "pk=1").exists()
        assert not (tmp_path / "pk=2").exists()
        # ``.ygg/`` sidecar (dot-prefixed) survives — the persist
        # hook will overwrite it on the next write.
        assert (tmp_path / ".ygg").is_dir()


class TestChildHelpers:
    """Direct child constructors — ``child_file`` and ``child_folder``.

    Both helpers mint a sub-:class:`Tabular` at ``self.path / name``
    without calling :meth:`iter_children` / :meth:`Path.iterdir` and
    without probing for existence on disk. They are the lazy
    counterpart to the eager :meth:`iter_children` walk — useful when
    the caller already knows the name (Hive partition probing,
    deterministic part-file paths, sidecar lookups).
    """

    def test_child_file_resolves_leaf_class_by_extension(self, tmp_path) -> None:
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        folder = FolderPath(path=str(tmp_path))
        leaf = folder.child_file("trades.parquet")

        assert isinstance(leaf, ParquetFile)
        assert leaf._parent.url == folder.path.url.joinpath("trades.parquet")
        # No I/O at mint time: the file doesn't exist on disk yet.
        assert not (tmp_path / "trades.parquet").exists()

    def test_child_file_falls_back_to_bytes_io_for_unknown_extension(
        self, tmp_path,
    ) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        folder = FolderPath(path=str(tmp_path))
        leaf = folder.child_file("notes.unknown_extension")

        # Unregistered extension → caller still gets a usable handle.
        assert isinstance(leaf, BytesIO)

    def test_child_folder_returns_lazy_subfolder(self, tmp_path) -> None:
        folder = FolderPath(path=str(tmp_path))
        sub = folder.child_folder("partition_key=42")

        assert isinstance(sub, FolderPath)
        assert sub.path.url == folder.path.url.joinpath("partition_key=42")
        # No I/O at mint time — the sub-folder doesn't exist yet, and
        # a downstream read against it must short-circuit to empty.
        assert not (tmp_path / "partition_key=42").exists()
        assert list(sub.iter_children()) == []

    def test_child_folder_preserves_subclass(self, tmp_path) -> None:
        # ``child_folder`` mints a child of the same concrete class so
        # subclasses (e.g. DeltaFolder) chain through correctly. The
        # ``mime_type = None`` opt-out keeps the subclass off the
        # format registry so the test doesn't clash with the parent
        # FolderPath slot.
        class _SubFolder(FolderPath):
            mime_type = None  # type: ignore[assignment]

        sub_root = _SubFolder(path=str(tmp_path))
        child = sub_root.child_folder("partition_key=0")
        assert type(child) is _SubFolder

    def test_child_folder_adopts_child(self, tmp_path) -> None:
        # Adoption stamps tabular_parent so the child shares schema /
        # free-cols caches with the parent (the same contract
        # ``iter_children`` provides for yielded children).
        folder = FolderPath(path=str(tmp_path))
        sub = folder.child_folder("partition_key=0")
        assert sub.tabular_parent is folder
