"""Tests for :class:`yggdrasil.delta.io.DeltaFolder` end-to-end."""
from __future__ import annotations

import os

from yggdrasil.data.data_field import Field
from yggdrasil.enums import Mode
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type, StringType
from yggdrasil.delta.io import DeltaOptions
from yggdrasil.delta.tests import DeltaTestCase


def _partition_schema() -> Schema:
    s = Schema()
    s.with_field(Field(name="id", dtype=Int64Type()))
    s.with_field(
        Field(name="region", dtype=StringType()).with_partition_by(True)
    )
    s.with_field(Field(name="val", dtype=StringType()))
    return s


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip(DeltaTestCase):
    def test_unpartitioned_round_trip(self) -> None:
        d = self.delta_io()
        t = self.pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        d.write_arrow_table(t)

        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.version, 0)
        self.assertEqual(snap.num_active_files(), 1)

        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column_names), {"id", "val"})

    def test_partitioned_round_trip(self) -> None:
        d = self.delta_io()
        t = self.pa.table({
            "id": [1, 2, 3, 4],
            "region": ["us", "us", "eu", "eu"],
            "val": ["a", "b", "c", "d"],
        })
        d.write_arrow_table(
            t, options=DeltaOptions(target=_partition_schema())
        )
        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.partition_columns, ["region"])
        self.assertEqual(snap.num_active_files(), 2)

        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, 4)
        self.assertIn("region", out.column_names)
        self.assertEqual(set(out.column("region").to_pylist()), {"us", "eu"})


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


class TestModes(DeltaTestCase):
    def test_append(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))
        d.write_arrow_batches(
            self.pa.table({"id": [3, 4]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )
        self.assertEqual(d.snapshot(fresh=True).num_active_files(), 2)
        self.assertEqual(d.read_arrow_table().num_rows, 4)

    def test_overwrite_emits_removes(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))
        d.write_arrow_batches(
            self.pa.table({"id": [3]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )
        d.write_arrow_table(
            self.pa.table({"id": [99]}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )
        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.num_active_files(), 1)
        self.assertEqual(d.read_arrow_table().column("id").to_pylist(), [99])

    def test_ignore_skips_when_non_empty(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1]}))
        d.write_arrow_batches(
            self.pa.table({"id": [2]}).to_batches(),
            options=DeltaOptions(mode=Mode.IGNORE),
        )
        self.assertEqual(d.read_arrow_table().column("id").to_pylist(), [1])

    def test_error_if_exists_raises(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1]}))
        with self.assertRaises(FileExistsError):
            d.write_arrow_batches(
                self.pa.table({"id": [2]}).to_batches(),
                options=DeltaOptions(mode=Mode.ERROR_IF_EXISTS),
            )


# ---------------------------------------------------------------------------
# Time-travel
# ---------------------------------------------------------------------------


class TestTimeTravel(DeltaTestCase):
    def test_read_at_old_version(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))
        d.write_arrow_batches(
            self.pa.table({"id": [3]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )
        d.write_arrow_batches(
            self.pa.table({"id": [4, 5]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )

        v0 = d.read_arrow_table(options=DeltaOptions(version=0))
        self.assertEqual(v0.column("id").to_pylist(), [1, 2])
        v1 = d.read_arrow_table(options=DeltaOptions(version=1))
        self.assertEqual(sorted(v1.column("id").to_pylist()), [1, 2, 3])
        head = d.read_arrow_table()
        self.assertEqual(sorted(head.column("id").to_pylist()), [1, 2, 3, 4, 5])


# ---------------------------------------------------------------------------
# Partition pruning
# ---------------------------------------------------------------------------


class TestPartitionPruning(DeltaTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.d = self.delta_io()
        self.d.write_arrow_table(
            self.pa.table({
                "id": [1, 2, 3, 4],
                "region": ["us", "us", "eu", "ap"],
                "val": ["a", "b", "c", "d"],
            }),
            options=DeltaOptions(target=_partition_schema()),
        )

    def test_prune_to_single_value(self) -> None:
        from yggdrasil.execution.expr import col as expr_col
        out = self.d.read_arrow_table(
            options=DeltaOptions(predicate=expr_col("region") == "us"),
        )
        self.assertEqual(out.num_rows, 2)
        self.assertEqual(set(out.column("region").to_pylist()), {"us"})

    def test_prune_unknown_value_returns_empty(self) -> None:
        from yggdrasil.execution.expr import col as expr_col
        out = self.d.read_arrow_table(
            options=DeltaOptions(predicate=expr_col("region") == "antarctica"),
        )
        self.assertEqual(out.num_rows, 0)


# ---------------------------------------------------------------------------
# Predicate-driven partition pruning — same file-skip behavior as
# explicit ``prune_values``, but derived from the row-level
# ``options.predicate``. Lets callers ship one Predicate that drives
# both the partition prune *and* the row-level filter downstream.
# ---------------------------------------------------------------------------


class TestPredicatePartitionPruning(DeltaTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.d = self.delta_io()
        self.d.write_arrow_table(
            self.pa.table({
                "id": [1, 2, 3, 4, 5],
                "region": ["us", "us", "eu", "ap", "br"],
                "val": ["a", "b", "c", "d", "e"],
            }),
            options=DeltaOptions(target=_partition_schema()),
        )

    def _files_read(self, options: DeltaOptions) -> int:
        """Count partition files surviving the prune (pre-parquet open)."""
        from yggdrasil.io.nested.delta.delta_folder import _partition_prune_values  # noqa: E501

        snap = self.d.snapshot(fresh=False)
        prune = _partition_prune_values(
            options.predicate, snap.partition_columns,
        )
        return sum(1 for _ in snap.prune_files(prune_values=prune))

    def test_eq_predicate_prunes_to_one_file(self) -> None:
        from yggdrasil.execution.expr import col as expr_col

        opts = DeltaOptions(predicate=(expr_col("region") == "us"))
        # Four partitions exist (us, us is one file because the writer
        # groups by partition value), only one matches.
        self.assertEqual(self._files_read(opts), 1)
        out = self.d.read_arrow_table(options=opts)
        self.assertEqual(out.num_rows, 2)
        self.assertEqual(set(out.column("region").to_pylist()), {"us"})

    def test_or_predicate_collapses_to_in_set(self) -> None:
        from yggdrasil.execution.expr import col as expr_col

        opts = DeltaOptions(
            predicate=(expr_col("region") == "us")
            | (expr_col("region") == "eu"),
        )
        # Simplify turns the OR into ``region IN ('us', 'eu')`` →
        # two partition files match.
        self.assertEqual(self._files_read(opts), 2)
        out = self.d.read_arrow_table(options=opts)
        self.assertEqual(set(out.column("region").to_pylist()), {"us", "eu"})

    def test_predicate_with_non_partition_column_still_prunes_files(self) -> None:
        # The ``id > 1`` half can't drive partition pruning, but the
        # ``region == "us"`` half still picks one file. The row-level
        # filter takes care of the ``id > 1`` rejection downstream.
        from yggdrasil.execution.expr import col as expr_col

        opts = DeltaOptions(
            predicate=(expr_col("region") == "us") & (expr_col("id") > 1),
        )
        self.assertEqual(self._files_read(opts), 1)
        out = self.d.read_arrow_table(options=opts)
        self.assertEqual(out.num_rows, 1)
        self.assertEqual(out.column("id").to_pylist(), [2])

    def test_predicate_in_list_prunes_to_subset(self) -> None:
        from yggdrasil.execution.expr import col as expr_col

        opts = DeltaOptions(
            predicate=expr_col("region").is_in(["us", "eu"]),
        )
        self.assertEqual(self._files_read(opts), 2)
        out = self.d.read_arrow_table(options=opts)
        self.assertEqual(set(out.column("region").to_pylist()), {"us", "eu"})

    def test_predicate_alone_with_unknown_value_returns_empty(self) -> None:
        from yggdrasil.execution.expr import col as expr_col

        opts = DeltaOptions(predicate=(expr_col("region") == "antarctica"))
        self.assertEqual(self._files_read(opts), 0)
        out = self.d.read_arrow_table(options=opts)
        self.assertEqual(out.num_rows, 0)

    def test_range_predicate_does_not_drive_file_prune(self) -> None:
        # ``id > 1`` is not extractable for partition pruning (range,
        # not partition column anyway) — every file survives the
        # prune, then the row-level filter strips the rejected rows.
        from yggdrasil.execution.expr import col as expr_col

        opts = DeltaOptions(predicate=(expr_col("id") > 3))
        self.assertEqual(self._files_read(opts), 4)
        out = self.d.read_arrow_table(options=opts)
        self.assertEqual(out.num_rows, 2)
        self.assertEqual(sorted(out.column("id").to_pylist()), [4, 5])


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


class TestCheckpoints(DeltaTestCase):
    def _write_n(self, d, n: int, *, kind: str = "v1", interval: int = 5) -> None:
        for i in range(n):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                self.pa.table({"id": [i]}).to_batches(),
                options=DeltaOptions(
                    mode=mode,
                    checkpoint_interval=interval,
                    checkpoint_kind=kind,
                ),
            )

    def test_v1_checkpoint_emitted_on_interval(self) -> None:
        d = self.delta_io()
        self._write_n(d, 6, kind="v1", interval=5)
        log_dir = os.path.join(str(d.path), "_delta_log")
        names = os.listdir(log_dir)
        self.assertIn("00000000000000000004.checkpoint.parquet", names)
        self.assertIn("_last_checkpoint", names)

    def test_v1_checkpoint_replay(self) -> None:
        d = self.delta_io()
        self._write_n(d, 6, kind="v1", interval=5)
        # Re-open with a fresh log so the checkpoint is the only way
        # the snapshot can recover the active set up to the
        # checkpoint version.
        from yggdrasil.delta.io import DeltaFolder
        d2 = DeltaFolder(path=str(d.path))
        out = d2.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), list(range(6)))

    def test_v2_checkpoint_replay(self) -> None:
        d = self.delta_io()
        self._write_n(d, 6, kind="v2", interval=5)
        log_dir = os.path.join(str(d.path), "_delta_log")
        sidecar_dir = os.path.join(log_dir, "_sidecars")
        self.assertTrue(os.path.isdir(sidecar_dir))
        self.assertTrue(any(
            n.endswith(".parquet") for n in os.listdir(sidecar_dir)
        ))
        from yggdrasil.delta.io import DeltaFolder
        d2 = DeltaFolder(path=str(d.path))
        out = d2.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), list(range(6)))


# ---------------------------------------------------------------------------
# Caching — metadata fetches collapse
# ---------------------------------------------------------------------------


class TestMetadataCaching(DeltaTestCase):
    def test_repeat_reads_dont_reopen_log(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))

        from yggdrasil.delta.log import DeltaLog
        original = DeltaLog._list_log_dir
        calls = {"n": 0}

        def counting(self):  # type: ignore[no-redef]
            calls["n"] += 1
            return original(self)

        DeltaLog._list_log_dir = counting  # type: ignore[assignment]
        try:
            d.read_arrow_table()
            d.read_arrow_table()
            d.collect_schema()
        finally:
            DeltaLog._list_log_dir = original  # type: ignore[assignment]

        # First read may walk the listing twice (segment + a follow-up
        # in the snapshot resolution); subsequent reads collapse to the
        # cached listing — three reads must not multiply the count.
        self.assertLessEqual(calls["n"], 2)

    def test_refresh_invalidates(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1]}))
        before = d.snapshot().num_active_files()
        d.write_arrow_batches(
            self.pa.table({"id": [2]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )
        # No refresh → cache returns stale snapshot ... but the writer
        # already invalidates after committing, so this should reflect
        # the new write without an explicit refresh.
        after = d.snapshot().num_active_files()
        self.assertEqual(before + 1, after)


# ---------------------------------------------------------------------------
# Schema fidelity
# ---------------------------------------------------------------------------


class TestSchema(DeltaTestCase):
    def test_collect_schema(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1], "name": ["a"]}))
        s = d.collect_schema()
        names = [f.name for f in s.fields]
        self.assertEqual(names, ["id", "name"])
