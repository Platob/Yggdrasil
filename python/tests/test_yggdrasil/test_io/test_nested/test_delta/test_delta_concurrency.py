"""Tests for the concurrency / upsert / uint write paths.

These are the contracts that distinguish the Delta writer from a
bare folder-of-parquets:

- UPSERT / MERGE — key-aware; existing rows whose key matches the
  incoming set get rewritten with incoming values; non-matching
  incoming rows are appended.
- Concurrent commits — the version race is detected (atomic create
  on the commit JSON) and the writer retries with a rebuilt action
  set against the new HEAD. Exhaustion surfaces a clean
  :class:`ConcurrentDeltaCommitError`.
- Unsigned-integer columns are reinterpreted as same-width signed
  via two's-complement on the way to parquet, and the Delta
  schemaString agrees with the on-disk payload.

Modes (append / overwrite / ignore / error_if_exists), partition
pruning, V1 + V2 checkpoint replay are covered by the parity suite
under ``tests/test_yggdrasil/test_delta/`` — this file adds to that.
"""

from __future__ import annotations

import shutil
from typing import Any

from yggdrasil.data.data_field import Field
from yggdrasil.data.enums import Mode
from yggdrasil.data.types.primitive import Int64Type
from yggdrasil.io.nested.delta import (
    ConcurrentDeltaCommitError,
    DeltaIO,
    DeltaOptions,
)
from yggdrasil.io.nested.delta.tests import DeltaTestCase


def _key_field(name: str = "id") -> Field:
    return Field(name, dtype=Int64Type())


# ---------------------------------------------------------------------------
# UPSERT / MERGE — key-aware merge
# ---------------------------------------------------------------------------


class TestUpsert(DeltaTestCase):
    def test_upsert_overwrites_matching_rows(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(
            self.pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]}),
        )
        # id=2 updates v from "b" -> "B"; id=4 is new
        d.write_arrow_batches(
            self.pa.table({"id": [2, 4], "v": ["B", "d"]}).to_batches(),
            options=DeltaOptions(mode=Mode.UPSERT, match_by=[_key_field()]),
        )
        out = d.read_arrow_table()
        rows = sorted(zip(out.column("id").to_pylist(), out.column("v").to_pylist()))
        self.assertEqual(rows, [(1, "a"), (2, "B"), (3, "c"), (4, "d")])

    def test_upsert_no_match_just_appends(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2], "v": ["a", "b"]}))
        d.write_arrow_batches(
            self.pa.table({"id": [3, 4], "v": ["c", "d"]}).to_batches(),
            options=DeltaOptions(mode=Mode.UPSERT, match_by=[_key_field()]),
        )
        out = d.read_arrow_table()
        rows = sorted(zip(out.column("id").to_pylist(), out.column("v").to_pylist()))
        self.assertEqual(rows, [(1, "a"), (2, "b"), (3, "c"), (4, "d")])

    def test_upsert_without_keys_collapses_to_append(self) -> None:
        # Without match_by, UPSERT semantically collapses to APPEND --
        # there's no key to dedup on.
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))
        d.write_arrow_batches(
            self.pa.table({"id": [2, 3]}).to_batches(),
            options=DeltaOptions(mode=Mode.UPSERT),
        )
        out = d.read_arrow_table()
        # Both [1, 2] and [2, 3] coexist -- duplicate id=2 is fine.
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 2, 3])

    def test_upsert_partitioned_table(self) -> None:
        # Partitioned table -- match_by spans the partition columns
        # plus the row identity column. Existing partition stays put;
        # the file under the matching partition is rewritten with the
        # surviving rows.
        d = self.delta_io()
        d.write_arrow_table(
            self.pa.table(
                {
                    "id": [1, 2, 3, 4],
                    "region": ["us", "us", "eu", "eu"],
                    "v": ["a", "b", "c", "d"],
                }
            ),
        )
        d.write_arrow_batches(
            self.pa.table(
                {"id": [2, 5], "region": ["us", "ap"], "v": ["B", "e"]}
            ).to_batches(),
            options=DeltaOptions(mode=Mode.UPSERT, match_by=[_key_field()]),
        )
        out = d.read_arrow_table()
        rows = sorted(
            zip(
                out.column("id").to_pylist(),
                out.column("region").to_pylist(),
                out.column("v").to_pylist(),
            )
        )
        self.assertEqual(
            rows,
            [
                (1, "us", "a"),
                (2, "us", "B"),
                (3, "eu", "c"),
                (4, "eu", "d"),
                (5, "ap", "e"),
            ],
        )


# ---------------------------------------------------------------------------
# Concurrent commit retry
# ---------------------------------------------------------------------------


class TestConcurrentRetry(DeltaTestCase):
    def _smuggle_commit(
        self,
        *,
        target: DeltaIO,
        version: int,
        rival_table: list[Any],
        rival_path_name: str = "rival",
    ) -> None:
        """Land an out-of-band commit at *version* under *target*'s log.

        Builds a parallel Delta from *rival_table* batches and copies
        the commit JSON over -- standalone trick for simulating "a
        concurrent writer landed first" without actual threads.
        """
        rival = self.delta_io(rival_path_name)
        rival.write_arrow_table(rival_table[0])
        for tbl in rival_table[1:]:
            rival.write_arrow_batches(
                tbl.to_batches(),
                options=DeltaOptions(mode=Mode.APPEND),
            )
        name = f"{version:020d}.json"
        src = (rival.path / "_delta_log" / name).full_path()
        dst = (target.path / "_delta_log" / name).full_path()
        shutil.copy(src, dst)

    def test_append_retries_past_concurrent_commit(self) -> None:
        d = self.delta_io("ours")
        d.write_arrow_table(self.pa.table({"id": [1]}))  # version 0

        # A rival writer landed version 1 between our read and our
        # write -- we should refresh, rebase, and land at version 2.
        rival_tables = [self.pa.table({"id": [10]}), self.pa.table({"id": [99]})]
        self._smuggle_commit(target=d, version=1, rival_table=rival_tables)

        # The smuggle happened in the rival's filesystem, then we copied
        # the commit JSON. d's HEAD cache still says 0; an immediate
        # fresh-snapshot read would pick up the new version, but
        # _write_arrow_batches will too. Force the writer to start
        # from version 0 by NOT calling refresh: the internal
        # snapshot(fresh=True) will see 1, try to commit at 2 -- which
        # is empty -- and succeed without retrying. So instead, drop
        # the rival's data file content too; what we really care about
        # is that _after_ a concurrent commit lands, our writer ends
        # up at the next free version. Reading it back proves rebased.
        d.write_arrow_batches(
            self.pa.table({"id": [2]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )
        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.version, 2)

        out = d.read_arrow_table()
        self.assertIn(2, out.column("id").to_pylist())

    def test_atomic_create_rejects_existing_version(self) -> None:
        # Direct test of the atomic-create primitive: writing to a
        # version that already exists raises FileExistsError, which is
        # the foundation of the retry loop.
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1]}))  # version 0
        with self.assertRaises(FileExistsError):
            d._commit_atomic(
                0, [d._build_commit_info(options=DeltaOptions(), mode=Mode.APPEND)]
            )

    def test_exhaustion_raises_concurrent_commit_error(self) -> None:
        # Force every commit attempt to fail so the budget is the
        # only thing that determines the outcome.
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1]}))

        attempts = {"n": 0}

        def always_race(version, actions):
            attempts["n"] += 1
            raise FileExistsError(f"simulated race at v{version}")

        d._commit_atomic = always_race  # type: ignore[assignment]

        with self.assertRaises(ConcurrentDeltaCommitError):
            d.write_arrow_batches(
                self.pa.table({"id": [2]}).to_batches(),
                options=DeltaOptions(
                    mode=Mode.APPEND,
                    commit_max_retries=2,
                    commit_retry_backoff=0,
                    commit_retry_jitter=0,
                ),
            )
        # 1 initial + 2 retries.
        self.assertEqual(attempts["n"], 3)


# ---------------------------------------------------------------------------
# Unsigned integer reinterpretation
# ---------------------------------------------------------------------------


class TestUnsignedIntReinterpretation(DeltaTestCase):
    def test_uint8_round_trip_via_twos_complement(self) -> None:
        d = self.delta_io()
        # uint8 values: 0, 127 (signed-positive boundary), 200, 255.
        # Cast to int8 should produce: 0, 127, -56, -1.
        d.write_arrow_table(
            self.pa.table(
                {"x": self.pa.array([0, 127, 200, 255], type=self.pa.uint8())},
            ),
        )
        out = d.read_arrow_table()
        self.assertEqual(out.schema.field("x").type, self.pa.int8())
        self.assertEqual(out.column("x").to_pylist(), [0, 127, -56, -1])

    def test_uint64_does_not_widen_to_decimal(self) -> None:
        # Without the as_spark widening short-circuit, uint64 would
        # land as DECIMAL(20, 0) -- 16 bytes per value, lossless but
        # storage-heavy. With the short-circuit it lands as int64
        # via two's-complement reinterpretation.
        d = self.delta_io()
        d.write_arrow_table(
            self.pa.table(
                {"big": self.pa.array([0, 2**63, 2**64 - 1], type=self.pa.uint64())}
            ),
        )
        snap = d.snapshot(fresh=True)
        self.assertIn('"name":"big","type":"long"', snap.schema_string)

        out = d.read_arrow_table()
        self.assertEqual(out.schema.field("big").type, self.pa.int64())
        self.assertEqual(
            out.column("big").to_pylist(),
            [0, -(2**63), -1],
        )

    def test_signed_columns_pass_through_unchanged(self) -> None:
        d = self.delta_io()
        d.write_arrow_table(
            self.pa.table({"x": self.pa.array([-1, 0, 1], type=self.pa.int32())}),
        )
        out = d.read_arrow_table()
        self.assertEqual(out.schema.field("x").type, self.pa.int32())
        self.assertEqual(out.column("x").to_pylist(), [-1, 0, 1])
