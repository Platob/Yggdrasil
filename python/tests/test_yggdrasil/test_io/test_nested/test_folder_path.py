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




class TestCheckedCast:
    """``CastOptions.checked_cast=True`` opts out of per-batch schema
    re-binding and casts.

    The folder partition / cache write path runs every batch through
    :meth:`CastOptions.check_source` (rebuilds a yggdrasil :class:`Field`
    from the batch's :class:`pa.Schema`) and :meth:`cast_arrow_tabular`
    (per-batch cast) by default. ``checked_cast=True`` short-circuits
    both — the caller guarantees the batch already matches the
    target. Used by :meth:`FolderPath._write_parts` when the parent
    already resolved the schema via :meth:`_schema_for_arrow`.
    """

    def test_check_source_short_circuits_when_checked(self) -> None:
        from yggdrasil.data.options import CastOptions
        import pyarrow as pa

        schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
        opts = CastOptions(checked_cast=True)
        # With ``checked_cast=True`` the peek does not run — source
        # stays None even though we passed a peekable schema.
        result = opts.check_source(schema, copy=False)
        assert result is opts
        assert result.source is None

    def test_cast_arrow_tabular_short_circuits_when_checked(self) -> None:
        from yggdrasil.data.options import CastOptions
        import pyarrow as pa

        # Build a target so ``cast_arrow_tabular`` would normally run.
        opts = CastOptions(checked_cast=False).check_target(
            pa.schema([("a", pa.int32())]),
        )
        batch = pa.record_batch([pa.array([1, 2, 3])], names=["a"])
        # Without ``checked_cast`` the cast pass runs (no-op cast here
        # but the dispatch fires); with it the input passes through
        # by identity.
        same_id = opts.copy(checked_cast=True).cast_arrow_tabular(batch)
        assert same_id is batch

    def test_write_arrow_batches_with_checked_cast_uses_first_batch_schema(
        self, tmp_path,
    ) -> None:
        # End-to-end: a caller that owns the source schema (an
        # ``HTTPResponse.values_to_arrow_batch`` projection, a
        # ``pa.RecordBatchReader``, etc.) writes with
        # ``checked_cast=True`` and the leaf write succeeds without
        # touching the cast machinery.
        import pyarrow as pa
        from yggdrasil.io.nested.folder_path import FolderOptions

        batch = pa.record_batch(
            [pa.array([1, 2, 3]), pa.array(["x", "y", "z"])],
            names=["a", "b"],
        )

        folder = FolderPath(path=str(tmp_path / "checked"))
        folder.write_arrow_batches(
            (batch,),
            options=FolderOptions(checked_cast=True),
        )

        # Round-trip: the bytes landed and read back match.
        reread = folder.read_arrow_table()
        assert reread.num_rows == 3
        assert reread.column_names == ["a", "b"]


class TestUniqueTagDedup:
    """``Field.unique`` triggers a client-side dedup on read.

    Contract:

    * Target column flagged ``unique=True`` and the source side
      doesn't carry the same flag → :meth:`Tabular.read_arrow_batches`
      collapses duplicate rows on that column before yielding.
    * Both sides flagged unique → no dedup runs (source already
      guarantees the invariant on disk).
    * Neither side flagged → no dedup runs (no contract to enforce).

    Verifies the contract holds across :meth:`read_arrow_table`,
    :meth:`read_arrow_batches`, :meth:`read_arrow_batch_reader`,
    and on the upsert / merge rewrite path so a writer maintaining
    the invariant via ``match_by`` doesn't get a redundant dedup
    pass on subsequent reads.
    """

    def _target(self):
        from yggdrasil.data import field, schema

        return schema(fields=[
            field("id", pa.int64(), nullable=False, tags={"unique": True}),
            field("v", pa.int64(), nullable=False),
        ])

    def _duplicate_batch(self) -> "pa.RecordBatch":
        # 5 rows with id ∈ {1, 2, 3} — 2 duplicates.
        return pa.record_batch(
            [pa.array([1, 2, 1, 3, 2]), pa.array([10, 20, 11, 30, 21])],
            names=["id", "v"],
        )

    def test_read_dedups_when_only_target_marks_unique(self, tmp_path) -> None:
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((self._duplicate_batch(),))

        # No options → no dedup (no target with unique).
        plain = folder.read_arrow_table()
        assert plain.num_rows == 5

        # Target carries unique → dedup pass runs.
        deduped = folder.read_arrow_table(options=FolderOptions(target=self._target()))
        assert deduped.num_rows == 3
        assert set(deduped.column("id").to_pylist()) == {1, 2, 3}

    def test_read_dedup_via_arrow_batches(self, tmp_path) -> None:
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((self._duplicate_batch(),))

        batches = list(folder.read_arrow_batches(
            options=FolderOptions(target=self._target()),
        ))
        total = sum(b.num_rows for b in batches)
        assert total == 3

    def test_read_dedup_via_arrow_batch_reader(self, tmp_path) -> None:
        from yggdrasil.data import field, schema
        from yggdrasil.io.nested.folder_path import FolderOptions

        # ``read_arrow_batch_reader`` strict-checks each yielded batch
        # against ``options.merged`` — use a nullable target so the
        # cast plane matches whatever the on-disk source carries
        # (the dedup contract is about column identity, not nullability).
        target = schema(fields=[
            field("id", pa.int64(), nullable=True, tags={"unique": True}),
            field("v", pa.int64(), nullable=True),
        ])

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((self._duplicate_batch(),))

        reader = folder.read_arrow_batch_reader(options=FolderOptions(target=target))
        table = reader.read_all()
        assert table.num_rows == 3

    def test_dedup_short_circuits_when_source_also_unique(self, tmp_path) -> None:
        # When the on-disk schema *also* declares the column unique,
        # the dedup pass is a no-op — we still get every row back.
        # The bench-side concern: the dedup wraps yields the original
        # iterator unchanged in this branch (no materialisation).
        from yggdrasil.io.nested.folder_path import FolderOptions

        # Build the batch off the target's arrow schema so the
        # ``t:unique`` metadata round-trips onto disk.
        target = self._target()
        arrow_schema = target.to_arrow_schema()
        batch = pa.record_batch(
            [pa.array([1, 2, 3]), pa.array([10, 20, 30])],
            schema=arrow_schema,
        )

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((batch,))

        # The on-disk part file carries the ``t:unique`` metadata, so
        # the dedup short-circuits and every distinct row comes back.
        out = folder.read_arrow_table(options=FolderOptions(target=target))
        assert out.num_rows == 3

    def test_dedup_columns_on_read_skips_when_source_unique(self) -> None:
        from yggdrasil.data import field, schema
        from yggdrasil.data.options import CastOptions

        target = schema(fields=[
            field("id", pa.int64(), tags={"unique": True}),
            field("v", pa.int64()),
        ])
        # Source already carries ``unique`` → dedup is the writer's
        # job (or already guaranteed), so the reader does nothing.
        source = schema(fields=[
            field("id", pa.int64(), tags={"unique": True}),
            field("v", pa.int64()),
        ])
        opts = CastOptions(source=source, target=target)
        assert opts.dedup_columns_on_read() == []

    def test_dedup_columns_on_read_picks_only_target_unique_fields(self) -> None:
        from yggdrasil.data import field, schema
        from yggdrasil.data.options import CastOptions

        target = schema(fields=[
            field("id", pa.int64(), tags={"unique": True}),
            field("alt", pa.int64(), tags={"unique": True}),
            field("v", pa.int64()),
        ])
        # Source declares ``id`` unique but not ``alt``.
        source = schema(fields=[
            field("id", pa.int64(), tags={"unique": True}),
            field("alt", pa.int64()),
            field("v", pa.int64()),
        ])
        opts = CastOptions(source=source, target=target)
        assert opts.dedup_columns_on_read() == ["alt"]

    def test_upsert_match_by_keeps_unique_invariant(self, tmp_path) -> None:
        from yggdrasil.data import field, schema
        from yggdrasil.data.enums import Mode
        from yggdrasil.io.nested.folder_path import FolderOptions

        target = self._target()
        arrow_schema = target.to_arrow_schema()

        # Round 1: APPEND lands 3 distinct rows.
        first = pa.record_batch(
            [pa.array([1, 2, 3]), pa.array([10, 20, 30])],
            schema=arrow_schema,
        )
        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((first,), options=FolderOptions(mode=Mode.APPEND))

        # Round 2: UPSERT with match_by=["id"] updates id=2 and adds id=4.
        second = pa.record_batch(
            [pa.array([2, 4]), pa.array([99, 40])],
            schema=arrow_schema,
        )
        folder.write_arrow_batches(
            (second,),
            options=FolderOptions(
                mode=Mode.UPSERT,
                match_by=[field("id", pa.int64())],
            ),
        )

        # Read back through the same unique-tagged target: the on-disk
        # rows are id ∈ {1, 2, 3, 4} after the merge rewrite. The
        # ``unique`` round-trip on the source schema makes the read-side
        # dedup a no-op (source declares it too), so every row survives.
        out = folder.read_arrow_table(options=FolderOptions(target=target))
        assert out.num_rows == 4
        assert sorted(out.column("id").to_pylist()) == [1, 2, 3, 4]
        # id=2's value must reflect the UPSERT (99, not 20).
        rows_by_id = {
            row["id"]: row["v"] for row in out.to_pylist()
        }
        assert rows_by_id[2] == 99
        assert rows_by_id[4] == 40

    def test_upsert_match_by_collapses_duplicates_within_incoming(
        self, tmp_path,
    ) -> None:
        # ``unique`` is a schema-level contract; the read-time dedup
        # protects against duplicates landed by an APPEND that never
        # set match_by. Verify the path: write duplicates with APPEND,
        # then a unique-tagged read collapses them.
        from yggdrasil.data import field
        from yggdrasil.data.enums import Mode
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        # Two writes against id=1 with different ``v`` values — would
        # leave duplicate id=1 rows on disk under plain APPEND.
        folder.write_arrow_batches(
            (pa.record_batch([pa.array([1, 2]), pa.array([10, 20])], names=["id", "v"]),),
            options=FolderOptions(mode=Mode.APPEND),
        )
        folder.write_arrow_batches(
            (pa.record_batch([pa.array([1, 3]), pa.array([11, 30])], names=["id", "v"]),),
            options=FolderOptions(mode=Mode.APPEND),
        )

        # Sanity: both id=1 rows are on disk before dedup.
        raw = folder.read_arrow_table()
        assert raw.column("id").to_pylist().count(1) == 2

        # With unique target → read collapses the duplicate.
        out = folder.read_arrow_table(options=FolderOptions(target=self._target()))
        assert set(out.column("id").to_pylist()) == {1, 2, 3}
