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


class TestUniqueByDedup:
    """``CastOptions.unique_by`` triggers a client-side dedup on read.

    Contract:

    * ``unique_by`` set → :meth:`Tabular.read_arrow_batches` collapses
      duplicate rows on the listed columns before yielding (first
      occurrence wins, original row order preserved).
    * Unset / empty → no dedup runs.

    The option carries the dedup keys as Fields — same shape as
    :attr:`CastOptions.match_by` — so the public API stays uniform
    across all "list of columns to act on" knobs.
    """

    def _duplicate_batch(self) -> "pa.RecordBatch":
        return pa.record_batch(
            [pa.array([1, 2, 1, 3, 2]), pa.array([10, 20, 11, 30, 21])],
            names=["id", "v"],
        )

    def test_read_dedups_when_unique_by_set(self, tmp_path) -> None:
        from yggdrasil.data import field
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((self._duplicate_batch(),))

        plain = folder.read_arrow_table()
        assert plain.num_rows == 5

        out = folder.read_arrow_table(options=FolderOptions(
            unique_by=[field("id", pa.int64())],
        ))
        assert out.num_rows == 3
        assert set(out.column("id").to_pylist()) == {1, 2, 3}

    def test_read_dedup_via_arrow_batches(self, tmp_path) -> None:
        from yggdrasil.data import field
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((self._duplicate_batch(),))

        batches = list(folder.read_arrow_batches(
            options=FolderOptions(unique_by=[field("id", pa.int64())]),
        ))
        assert sum(b.num_rows for b in batches) == 3

    def test_unique_by_accepts_strings_normalised_to_fields(self) -> None:
        from yggdrasil.data.options import CastOptions

        # Plain string keys get coerced to Fields in ``__post_init__``,
        # mirroring the ``match_by`` shape so the rest of the
        # cast machinery sees a uniform ``list[Field]`` surface.
        opts = CastOptions(unique_by=["id", "name"])
        assert opts.dedup_columns_on_read() == ["id", "name"]
        assert all(hasattr(f, "name") for f in opts.unique_by or [])

    def test_unique_by_unset_short_circuits(self) -> None:
        from yggdrasil.data.options import CastOptions

        opts = CastOptions()
        assert opts.dedup_columns_on_read() == []

    def test_upsert_match_by_then_unique_by_read(self, tmp_path) -> None:
        # End-to-end: APPEND lands duplicates, UPSERT with match_by
        # collapses them on disk, then a unique_by read returns one
        # row per id even if the on-disk shape still carries dupes.
        from yggdrasil.data import field
        from yggdrasil.data.enums import Mode
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches(
            (pa.record_batch([pa.array([1, 2]), pa.array([10, 20])], names=["id", "v"]),),
            options=FolderOptions(mode=Mode.APPEND),
        )
        folder.write_arrow_batches(
            (pa.record_batch([pa.array([1, 3]), pa.array([11, 30])], names=["id", "v"]),),
            options=FolderOptions(mode=Mode.APPEND),
        )
        # Two id=1 rows on disk pre-dedup.
        assert folder.read_arrow_table().column("id").to_pylist().count(1) == 2

        out = folder.read_arrow_table(options=FolderOptions(
            unique_by=[field("id", pa.int64())],
        ))
        assert set(out.column("id").to_pylist()) == {1, 2, 3}


class TestTimeSampleByResample:
    """``CastOptions.time_sample_by`` triggers a client-side resample on read.

    Each entry's :attr:`Field.metadata` carries a non-tag
    ``b"time_sampling"`` key whose value is the ISO-8601 duration the
    column should be snapped to. Unset → no resample.
    """

    def _make_quarter_hour_batch(self) -> "pa.RecordBatch":
        import datetime as dt
        epoch = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        ts = pa.array(
            [epoch + dt.timedelta(minutes=15 * i) for i in range(12)],
            type=pa.timestamp("us", "UTC"),
        )
        v = pa.array(list(range(12)))
        return pa.record_batch([ts, v], names=["ts", "v"])

    def _hourly_field(self):
        from yggdrasil.data import field
        from yggdrasil.data.options import timedelta_to_iso_duration
        import datetime as dt

        iso = timedelta_to_iso_duration(dt.timedelta(hours=1))
        return field(
            "ts", pa.timestamp("us", "UTC"),
            metadata={b"time_sampling": iso.encode()},
        )

    def test_read_resamples_to_target_grid(self, tmp_path) -> None:
        import datetime as dt
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((self._make_quarter_hour_batch(),))

        plain = folder.read_arrow_table()
        assert plain.num_rows == 12

        out = folder.read_arrow_table(options=FolderOptions(
            time_sample_by=[self._hourly_field()],
        ))
        assert out.num_rows == 3
        expected = [
            dt.datetime(2024, 1, 1, h, 0, tzinfo=dt.timezone.utc)
            for h in range(3)
        ]
        assert out.column("ts").to_pylist() == expected
        # ``first`` per bucket — the :00-minute row of each hour wins.
        assert out.column("v").to_pylist() == [0, 4, 8]

    def test_resample_unset_short_circuits(self, tmp_path) -> None:
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((self._make_quarter_hour_batch(),))
        out = folder.read_arrow_table(options=FolderOptions())
        assert out.num_rows == 12

    def test_resample_on_read_picks_first_valid_entry(self) -> None:
        from yggdrasil.data import field
        from yggdrasil.data.options import CastOptions, timedelta_to_iso_duration
        import datetime as dt

        # Field with no ``time_sampling`` metadata is skipped; the
        # second entry's PT1H wins. ``partition_by`` falls back to ``[]``
        # because no target schema is bound.
        bare = field("ts1", pa.timestamp("us", "UTC"))
        tagged = field(
            "ts2", pa.timestamp("us", "UTC"),
            metadata={b"time_sampling": timedelta_to_iso_duration(
                dt.timedelta(hours=1),
            ).encode()},
        )
        opts = CastOptions(time_sample_by=[bare, tagged])
        assert opts.resample_on_read() == ("ts2", 3600, [], "ffill")

    def test_timedelta_to_iso_duration_round_trip(self) -> None:
        from yggdrasil.data.options import timedelta_to_iso_duration
        from yggdrasil.data.types.primitive.temporal import _parse_iso_duration
        import datetime as dt

        cases = [
            dt.timedelta(hours=1),
            dt.timedelta(days=1),
            dt.timedelta(minutes=5, seconds=30),
            dt.timedelta(0),
        ]
        for td in cases:
            iso = timedelta_to_iso_duration(td)
            assert _parse_iso_duration(iso) == td

    def test_unique_by_and_time_sample_by_chain(self, tmp_path) -> None:
        # Both passes run in sequence: resample first (downsample to
        # hourly), then dedup on the resampled column. The hourly grid
        # already gives one row per hour so the dedup is effectively a
        # no-op here, but the chain has to compose.
        from yggdrasil.data import field
        from yggdrasil.io.nested.folder_path import FolderOptions

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((self._make_quarter_hour_batch(),))

        out = folder.read_arrow_table(options=FolderOptions(
            time_sample_by=[self._hourly_field()],
            unique_by=[field("ts", pa.timestamp("us", "UTC"))],
        ))
        assert out.num_rows == 3

    def test_partition_by_derived_from_target_primary_keys(self) -> None:
        # ``resample_on_read`` defaults ``partition_by`` to the
        # target schema's :attr:`primary_key` fields, minus the
        # time column when it's also primary.
        from yggdrasil.data import field, schema
        from yggdrasil.data.options import CastOptions, timedelta_to_iso_duration
        import datetime as dt

        target = schema(fields=[
            field("symbol", pa.string(), tags={"primary_key": True}),
            field("ts", pa.timestamp("us", "UTC"), tags={"primary_key": True}),
            field("price", pa.float64()),
        ])
        opts = CastOptions(
            target=target,
            time_sample_by=[field(
                "ts", pa.timestamp("us", "UTC"),
                metadata={b"time_sampling": timedelta_to_iso_duration(
                    dt.timedelta(hours=1),
                ).encode()},
            )],
        )
        assert opts.resample_on_read() == ("ts", 3600, ["symbol"], "ffill")

    def test_partition_by_skips_when_only_time_is_primary(self) -> None:
        # When the only primary key IS the time column, there's
        # nothing to partition on — fall back to a flat resample.
        from yggdrasil.data import field, schema
        from yggdrasil.data.options import CastOptions, timedelta_to_iso_duration
        import datetime as dt

        target = schema(fields=[
            field("ts", pa.timestamp("us", "UTC"), tags={"primary_key": True}),
            field("v", pa.int64()),
        ])
        opts = CastOptions(
            target=target,
            time_sample_by=[field(
                "ts", pa.timestamp("us", "UTC"),
                metadata={b"time_sampling": timedelta_to_iso_duration(
                    dt.timedelta(hours=1),
                ).encode()},
            )],
        )
        assert opts.resample_on_read() == ("ts", 3600, [], "ffill")

    def test_partition_by_buckets_each_entity_independently(
        self, tmp_path,
    ) -> None:
        # End-to-end: two symbols, each with quarter-hour rows. The
        # resample partitions on ``symbol`` so each symbol's timeline
        # buckets independently — both yield 3 hourly rows.
        import datetime as dt
        from yggdrasil.data import field, schema
        from yggdrasil.data.options import timedelta_to_iso_duration
        from yggdrasil.io.nested.folder_path import FolderOptions

        epoch = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        # 12 rows per symbol, interleaved on disk.
        rows: list[tuple[str, dt.datetime, int]] = []
        for i in range(12):
            t = epoch + dt.timedelta(minutes=15 * i)
            rows.append(("A", t, i))
            rows.append(("B", t, 100 + i))
        symbols = pa.array([r[0] for r in rows])
        ts = pa.array([r[1] for r in rows], type=pa.timestamp("us", "UTC"))
        vals = pa.array([r[2] for r in rows])
        batch = pa.record_batch([symbols, ts, vals], names=["symbol", "ts", "v"])

        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches((batch,))

        target = schema(fields=[
            field("symbol", pa.string(), tags={"primary_key": True}),
            field("ts", pa.timestamp("us", "UTC"), tags={"primary_key": True}),
            field("v", pa.int64()),
        ])
        ts_field = field(
            "ts", pa.timestamp("us", "UTC"),
            metadata={b"time_sampling": timedelta_to_iso_duration(
                dt.timedelta(hours=1),
            ).encode()},
        )
        out = folder.read_arrow_table(options=FolderOptions(
            target=target, time_sample_by=[ts_field],
        ))
        # 3 hourly rows × 2 symbols = 6 rows total.
        assert out.num_rows == 6
        rows_out = out.to_pylist()
        per_symbol = {"A": [], "B": []}
        for r in rows_out:
            per_symbol[r["symbol"]].append(r["ts"])
        expected = [
            dt.datetime(2024, 1, 1, h, 0, tzinfo=dt.timezone.utc)
            for h in range(3)
        ]
        assert sorted(per_symbol["A"]) == expected
        assert sorted(per_symbol["B"]) == expected


class TestFillStrategyOnResample:
    """``fill_strategy`` controls how nulls left by resample's ``first`` agg are filled.

    The resample's bucket collapse picks the first row per bucket
    verbatim — null cells stay null. ``fill_strategy="ffill"``
    (default) propagates the last non-null value forward into
    subsequent nulls within the same partition; ``"bfill"`` does
    the same backward; ``"none"`` is a no-op. Cross-partition
    leaks are never allowed: a partition whose first non-null is
    null stays null until a non-null arrives within that partition.
    """

    def test_ffill_flat_carries_last_non_null_forward(self) -> None:
        import datetime as dt
        from yggdrasil.arrow.ops import resample_arrow_table

        ts = pa.array(
            [dt.datetime(2024, 1, 1, h) for h in range(4)],
            type=pa.timestamp("us"),
        )
        v = pa.array([1, None, None, 4])
        t = pa.table({"ts": ts, "v": v})
        out = resample_arrow_table(t, time_column="ts", sampling_seconds=3600)
        assert out.column("v").to_pylist() == [1, 1, 1, 4]

    def test_ffill_per_partition_does_not_leak(self) -> None:
        import datetime as dt
        from yggdrasil.arrow.ops import resample_arrow_table

        # Symbol A: [1, None, None] — ffill within A yields [1, 1, 1].
        # Symbol B: [None, 5, None] — the leading null has no prior
        # non-null in B's partition, so it stays null; trailing null
        # ffills from 5.
        ts = pa.array(
            [dt.datetime(2024, 1, 1, h) for h in range(6)],
            type=pa.timestamp("us"),
        )
        sym = pa.array(["A", "A", "A", "B", "B", "B"])
        v = pa.array([1, None, None, None, 5, None])
        t = pa.table({"ts": ts, "sym": sym, "v": v})
        out = resample_arrow_table(
            t, time_column="ts", sampling_seconds=3600,
            partition_by=["sym"],
        )
        rows = out.to_pylist()
        rows.sort(key=lambda r: (r["sym"], r["ts"]))
        assert [r["v"] for r in rows] == [1, 1, 1, None, 5, 5]

    def test_bfill_propagates_next_non_null_backward(self) -> None:
        import datetime as dt
        from yggdrasil.arrow.ops import resample_arrow_table

        ts = pa.array(
            [dt.datetime(2024, 1, 1, h) for h in range(4)],
            type=pa.timestamp("us"),
        )
        v = pa.array([None, None, 3, None])
        t = pa.table({"ts": ts, "v": v})
        out = resample_arrow_table(
            t, time_column="ts", sampling_seconds=3600,
            fill_strategy="bfill",
        )
        # Trailing null stays — nothing to bfill from.
        assert out.column("v").to_pylist() == [3, 3, 3, None]

    def test_none_strategy_leaves_nulls_in_place(self) -> None:
        import datetime as dt
        from yggdrasil.arrow.ops import resample_arrow_table

        ts = pa.array(
            [dt.datetime(2024, 1, 1, h) for h in range(4)],
            type=pa.timestamp("us"),
        )
        v = pa.array([1, None, None, 4])
        t = pa.table({"ts": ts, "v": v})
        out = resample_arrow_table(
            t, time_column="ts", sampling_seconds=3600,
            fill_strategy="none",
        )
        assert out.column("v").to_pylist() == [1, None, None, 4]

    def test_invalid_strategy_raises(self) -> None:
        import datetime as dt
        import pytest
        from yggdrasil.arrow.ops import resample_arrow_table

        ts = pa.array([dt.datetime(2024, 1, 1)], type=pa.timestamp("us"))
        t = pa.table({"ts": ts, "v": pa.array([1])})
        with pytest.raises(ValueError, match="fill_strategy"):
            resample_arrow_table(
                t, time_column="ts", sampling_seconds=3600,
                fill_strategy="zfill",
            )

    def test_fill_arrow_table_nested_columns_skipped(self) -> None:
        from yggdrasil.arrow.ops import fill_arrow_table

        # Struct + list payload with nulls in a scalar column.
        t = pa.table({
            "sym": pa.array(["A", "A", "A"]),
            "ts": pa.array([1, 2, 3]),
            "v": pa.array([1, None, 3]),
            "addr": pa.array(
                [{"city": "x", "zip": 1}, {"city": "y", "zip": 2}, {"city": "z", "zip": 3}],
                type=pa.struct([("city", pa.string()), ("zip", pa.int32())]),
            ),
        })
        out = fill_arrow_table(t, sort_by="ts", partition_by=["sym"], fill_strategy="ffill")
        # Scalar column gets filled; nested column rides through unchanged.
        assert out.column("v").to_pylist() == [1, 1, 3]
        assert out.column("addr").to_pylist() == t.column("addr").to_pylist()


class TestComplexNestedTypeDedupResample:
    """Dedup / resample on tables with list, struct, list-of-struct columns.

    pyarrow's ``group_by`` doesn't support nested types as group keys,
    but it accepts them as non-key columns. The dedup / resample
    contract holds as long as the keys / time column themselves are
    scalar — the nested payload columns ride through ``take`` /
    ``filter`` unchanged.

    These tests stress the pure-arrow ops at the shapes :class:`Response`
    / :class:`PreparedRequest` actually carry (headers as
    ``map<string, string>``, request URL bits as ``struct<…>``,
    user_info as a nested struct, body as ``binary``) so a regression
    in the underlying group_by / take wiring shows up here.
    """

    @staticmethod
    def _build_table_with_list(rows: int) -> "pa.Table":
        """``id`` (int64) + ``items`` (list<int64>) — list payload."""
        return pa.table({
            "id": pa.array([i % (rows // 2 or 1) for i in range(rows)]),
            "v": pa.array([i for i in range(rows)]),
            "items": pa.array(
                [[i, i + 1, i + 2] for i in range(rows)],
                type=pa.list_(pa.int64()),
            ),
        })

    @staticmethod
    def _build_table_with_struct(rows: int) -> "pa.Table":
        """``id`` (int64) + ``addr`` (struct<city, zip>) — struct payload."""
        struct_type = pa.struct([("city", pa.string()), ("zip", pa.int32())])
        return pa.table({
            "id": pa.array([i % (rows // 2 or 1) for i in range(rows)]),
            "v": pa.array([i for i in range(rows)]),
            "addr": pa.array(
                [{"city": f"city-{i}", "zip": i * 100} for i in range(rows)],
                type=struct_type,
            ),
        })

    @staticmethod
    def _build_table_with_list_of_struct(rows: int) -> "pa.Table":
        """``id`` (int64) + ``orders`` (list<struct<sku, qty>>) — nested payload."""
        item_type = pa.struct([("sku", pa.string()), ("qty", pa.int32())])
        return pa.table({
            "id": pa.array([i % (rows // 2 or 1) for i in range(rows)]),
            "orders": pa.array(
                [
                    [{"sku": f"sku-{i}-{k}", "qty": k} for k in range(3)]
                    for i in range(rows)
                ],
                type=pa.list_(item_type),
            ),
        })

    @staticmethod
    def _build_table_with_map(rows: int) -> "pa.Table":
        """``id`` (int64) + ``headers`` (map<string, string>) — map payload.

        Matches the shape :class:`Response` uses for its headers
        column; the cache layer's dedup walks tables with this exact
        payload shape.
        """
        map_type = pa.map_(pa.string(), pa.string())
        return pa.table({
            "id": pa.array([i % (rows // 2 or 1) for i in range(rows)]),
            "headers": pa.array(
                [
                    [(f"k{i}", f"v{i}"), ("type", "json")]
                    for i in range(rows)
                ],
                type=map_type,
            ),
        })

    def test_dedup_preserves_list_payload(self) -> None:
        from yggdrasil.arrow.ops import dedup_arrow_table

        t = self._build_table_with_list(rows=6)
        # 6 rows → 3 distinct ids (i % 3) → dedup to 3 rows.
        out = dedup_arrow_table(t, ["id"])
        assert out.num_rows == 3
        # Payload preserved — first occurrence per id wins.
        assert out.column("items").to_pylist() == [
            [0, 1, 2], [1, 2, 3], [2, 3, 4],
        ]

    def test_dedup_preserves_struct_payload(self) -> None:
        from yggdrasil.arrow.ops import dedup_arrow_table

        t = self._build_table_with_struct(rows=6)
        out = dedup_arrow_table(t, ["id"])
        assert out.num_rows == 3
        addrs = out.column("addr").to_pylist()
        # First occurrence's struct content rides through unchanged.
        assert addrs == [
            {"city": "city-0", "zip": 0},
            {"city": "city-1", "zip": 100},
            {"city": "city-2", "zip": 200},
        ]

    def test_dedup_preserves_list_of_struct_payload(self) -> None:
        from yggdrasil.arrow.ops import dedup_arrow_table

        t = self._build_table_with_list_of_struct(rows=6)
        out = dedup_arrow_table(t, ["id"])
        assert out.num_rows == 3
        orders = out.column("orders").to_pylist()
        assert orders[0] == [
            {"sku": "sku-0-0", "qty": 0},
            {"sku": "sku-0-1", "qty": 1},
            {"sku": "sku-0-2", "qty": 2},
        ]

    def test_dedup_preserves_map_payload(self) -> None:
        from yggdrasil.arrow.ops import dedup_arrow_table

        t = self._build_table_with_map(rows=6)
        out = dedup_arrow_table(t, ["id"])
        assert out.num_rows == 3
        # ``pa.array(...).to_pylist()`` for a map column returns
        # ``list[tuple[K, V]]`` per row.
        first_row_headers = dict(out.column("headers")[0].as_py())
        assert first_row_headers == {"k0": "v0", "type": "json"}

    def test_resample_preserves_list_of_struct_payload(self, tmp_path) -> None:
        from yggdrasil.arrow.ops import resample_arrow_table
        import datetime as dt

        # Build a quarter-hour series with a list<struct> payload.
        epoch = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        item_type = pa.struct([("sku", pa.string()), ("qty", pa.int32())])
        ts = pa.array(
            [epoch + dt.timedelta(minutes=15 * i) for i in range(12)],
            type=pa.timestamp("us", "UTC"),
        )
        orders = pa.array(
            [
                [{"sku": f"row-{i}-{k}", "qty": k} for k in range(2)]
                for i in range(12)
            ],
            type=pa.list_(item_type),
        )
        t = pa.table({"ts": ts, "orders": orders})

        out = resample_arrow_table(t, time_column="ts", sampling_seconds=3600)
        assert out.num_rows == 3
        # The :00 row of each hour wins ("first" aggregator), so
        # the surviving orders are i ∈ {0, 4, 8}.
        survivors = out.column("orders").to_pylist()
        assert survivors[0] == [
            {"sku": "row-0-0", "qty": 0}, {"sku": "row-0-1", "qty": 1},
        ]
        assert survivors[1] == [
            {"sku": "row-4-0", "qty": 0}, {"sku": "row-4-1", "qty": 1},
        ]
        assert survivors[2] == [
            {"sku": "row-8-0", "qty": 0}, {"sku": "row-8-1", "qty": 1},
        ]

    def test_resample_with_partition_by_struct_payload(self) -> None:
        # Per-entity resample with a struct payload column. The
        # partition_by column ('symbol') is scalar; the struct rides
        # through ``take`` unchanged.
        from yggdrasil.arrow.ops import resample_arrow_table
        import datetime as dt

        epoch = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        struct_type = pa.struct([("price", pa.float64()), ("qty", pa.int32())])
        rows: list[tuple[str, dt.datetime, dict]] = []
        for i in range(12):
            t = epoch + dt.timedelta(minutes=15 * i)
            rows.append(("A", t, {"price": float(i), "qty": i}))
            rows.append(("B", t, {"price": float(i * 2), "qty": i * 2}))
        syms = pa.array([r[0] for r in rows])
        ts = pa.array([r[1] for r in rows], type=pa.timestamp("us", "UTC"))
        ticks = pa.array([r[2] for r in rows], type=struct_type)
        table = pa.table({"symbol": syms, "ts": ts, "tick": ticks})

        out = resample_arrow_table(
            table,
            time_column="ts",
            sampling_seconds=3600,
            partition_by=["symbol"],
        )
        # 3 hourly buckets × 2 symbols = 6 rows total.
        assert out.num_rows == 6
        rows_by_symbol = {"A": [], "B": []}
        for row in out.to_pylist():
            rows_by_symbol[row["symbol"]].append(row["tick"])
        # First-occurrence row of each hour for "A" is i ∈ {0, 4, 8}.
        assert rows_by_symbol["A"] == [
            {"price": 0.0, "qty": 0},
            {"price": 4.0, "qty": 4},
            {"price": 8.0, "qty": 8},
        ]
        # And for "B" — multiplied by 2.
        assert rows_by_symbol["B"] == [
            {"price": 0.0, "qty": 0},
            {"price": 8.0, "qty": 8},
            {"price": 16.0, "qty": 16},
        ]

    def test_folder_round_trip_with_nested_payload_and_dedup(
        self, tmp_path,
    ) -> None:
        # End-to-end: write a parquet folder with a list<struct>
        # payload, then read back with ``unique_by`` set.
        from yggdrasil.data import field
        from yggdrasil.io.nested.folder_path import FolderOptions
        from yggdrasil.data.enums import MediaTypes

        t = self._build_table_with_list_of_struct(rows=8)
        folder = FolderPath(path=str(tmp_path))
        folder.write_arrow_batches(
            (t.to_batches()[0],),
            options=FolderOptions(child_media_type=MediaTypes.PARQUET),
        )

        # 8 rows on disk → dedup to 4 distinct ids.
        out = folder.read_arrow_table(options=FolderOptions(
            unique_by=[field("id", pa.int64())],
        ))
        assert out.num_rows == 4
        # ``orders`` column survives the round trip (parquet preserves
        # list<struct>) and the dedup ``take``.
        orders = out.column("orders").to_pylist()
        assert orders[0] == [
            {"sku": "sku-0-0", "qty": 0},
            {"sku": "sku-0-1", "qty": 1},
            {"sku": "sku-0-2", "qty": 2},
        ]


class TestPartitionDataCache:
    """``FolderPath._PARTITION_DATA_CACHE`` short-circuits leaf reads.

    The cache lives on :class:`FolderPath` (process-wide ExpiringDict)
    and stores the unfiltered batches for any partition-leaf folder
    (one whose ``static_values`` were populated by Hive parsing). A
    second read of the same partition skips the file open + IPC parse
    entirely; the row-level predicate filter still runs on the cached
    batches in pyarrow's C++ kernel.
    """

    def _seed_partitioned_folder(self, tmp_path, *, n: int = 6):
        from yggdrasil.data import field, schema
        epoch = pa.array(list(range(n)))
        part = pa.array([f"p{i % 3}" for i in range(n)])
        sch = schema(fields=[
            field("part", pa.string(), tags={"partition_by": True}),
            field("v", pa.int64()),
        ])
        folder = FolderPath(path=str(tmp_path))
        folder._persist_schema(sch)
        batch = pa.record_batch([part, epoch], names=["part", "v"])
        folder.write_arrow_batches([batch])
        return folder

    def test_partition_cache_populates_on_first_read(self, tmp_path) -> None:
        folder = self._seed_partitioned_folder(tmp_path)
        FolderPath._PARTITION_DATA_CACHE.clear()
        # First read: populates cache for each existing partition leaf.
        assert folder.read_arrow_table().num_rows == 6
        # 3 distinct ``part=`` values → 3 leaf entries.
        assert len(FolderPath._PARTITION_DATA_CACHE._store) == 3

    def test_partition_cache_hit_skips_file_open(self, tmp_path) -> None:
        """A cached partition leaf never opens its leaf files again.

        Wraps :class:`ArrowIPCFile._read_arrow_batches` with a counter
        so the second read can prove every leaf was answered from
        memory: count rises by one per file on the first (cold) read
        and stays put on the second.
        """
        from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile

        folder = self._seed_partitioned_folder(tmp_path)
        FolderPath._PARTITION_DATA_CACHE.clear()
        _ = folder.read_arrow_table()  # warm — opens 3 leaf files

        orig = ArrowIPCFile._read_arrow_batches
        opened = [0]

        def counting(self, options):
            opened[0] += 1
            return orig(self, options)

        ArrowIPCFile._read_arrow_batches = counting
        try:
            out = folder.read_arrow_table()
        finally:
            ArrowIPCFile._read_arrow_batches = orig
        assert out.num_rows == 6
        assert opened[0] == 0, (
            f"expected 0 leaf-file reads on cached path, got {opened[0]}"
        )

        # Predicate-bearing reads still hit the cache; the row-level
        # filter runs on the cached batches in pyarrow's C++ kernel.
        from yggdrasil.io.tabular.execution.expr import col
        filtered = folder.filter(col("v") >= 3).read_arrow_table()
        assert sorted(filtered.column("v").to_pylist()) == [3, 4, 5]

    def test_partition_cache_invalidates_on_write(self, tmp_path) -> None:
        folder = self._seed_partitioned_folder(tmp_path)
        FolderPath._PARTITION_DATA_CACHE.clear()
        _ = folder.read_arrow_table()  # warm
        assert len(FolderPath._PARTITION_DATA_CACHE._store) == 3

        # A write into one of the partitions invalidates its cache
        # entry on the way in (the partition leaf's
        # ``_write_arrow_batches`` calls ``_invalidate_partition_cache``).
        extra = pa.record_batch(
            [pa.array(["p0"]), pa.array([99])], names=["part", "v"],
        )
        folder.write_arrow_batches([extra])
        # The next read of ``part=p0`` re-populates the entry with
        # the freshly-written row included.
        out = folder.read_arrow_table()
        assert out.num_rows == 7
        assert 99 in out.column("v").to_pylist()
