"""Concurrent inserts / merges against one table under thread fan-out.

Skipped unless ``DATABRICKS_HOST`` (and credentials) are set — see
:class:`SQLIntegrationCase`.
"""
from __future__ import annotations

import concurrent.futures as cf
from typing import ClassVar

import pyarrow as pa

from yggdrasil.enums import Mode

from ._sql_integration import SQLIntegrationCase


class TestSQLConcurrentWrites(SQLIntegrationCase):
    """Drive the same Delta target from N threads at once.

    Delta serializes commits at the table level — each writer that
    races a successful commit retries against the latest snapshot, so
    the final state must match a serial run of the same workload.
    These tests assert that contract through the public ``insert``
    surface: no lost rows on disjoint appends, no duplicate keys on
    overlapping upserts.
    """

    PARALLELISM: ClassVar[int] = 4
    ROWS_PER_WRITER: ClassVar[int] = 25

    @staticmethod
    def _writer_chunk(writer_id: int, n: int, *, key_offset: int = 0) -> pa.Table:
        """Build a deterministic chunk for *writer_id*.

        ``key_offset`` lets two writers either own disjoint keys
        (offset = writer_id * n) or share keys (offset = 0).
        """
        ids = list(range(key_offset, key_offset + n))
        return pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "writer": pa.array([writer_id] * n, type=pa.int32()),
                "amount": pa.array(
                    [float(writer_id * 1000 + i) for i in range(n)],
                    type=pa.float64(),
                ),
            }
        )

    @staticmethod
    def _empty_schema() -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("writer", pa.int32()),
                pa.field("amount", pa.float64()),
            ]
        )

    def _run_in_parallel(self, fn, args_iter):
        """Run ``fn`` concurrently and re-raise the first exception."""
        with cf.ThreadPoolExecutor(max_workers=self.PARALLELISM) as pool:
            futures = [pool.submit(fn, *args) for args in args_iter]
            for fut in cf.as_completed(futures):
                fut.result()  # surface exceptions from the workers

    # ------------------------------------------------------------------
    # Disjoint appends — no row should be lost
    # ------------------------------------------------------------------

    def test_concurrent_appends_disjoint_keys_preserve_all_rows(self) -> None:
        table = self._unique_table("concurrent_append")
        self._ensure_table(table, self._empty_schema())

        def append(writer_id: int) -> None:
            chunk = self._writer_chunk(
                writer_id,
                self.ROWS_PER_WRITER,
                key_offset=writer_id * self.ROWS_PER_WRITER,
            )
            table.insert(chunk, mode=Mode.APPEND)

        self._run_in_parallel(
            append, [(i,) for i in range(self.PARALLELISM)],
        )

        count = self.engine.execute(
            f"SELECT COUNT(*) AS n FROM {table.full_name(safe=True)}"
        ).to_arrow_table().column("n")[0].as_py()
        self.assertEqual(count, self.PARALLELISM * self.ROWS_PER_WRITER)

        per_writer = self.engine.execute(
            f"SELECT writer, COUNT(*) AS n FROM {table.full_name(safe=True)} "
            "GROUP BY writer ORDER BY writer"
        ).to_arrow_table().to_pylist()
        self.assertEqual(
            per_writer,
            [{"writer": i, "n": self.ROWS_PER_WRITER} for i in range(self.PARALLELISM)],
        )

    # ------------------------------------------------------------------
    # Overlapping upserts — final state has one row per key
    # ------------------------------------------------------------------

    def test_concurrent_upserts_overlapping_keys_no_duplicates(self) -> None:
        """Every writer upserts the same key range. After all commits
        land, every key must be present exactly once and the surviving
        ``writer`` value is one of the writer ids in [0, PARALLELISM)."""
        table = self._unique_table("concurrent_upsert")
        self._ensure_table(table, self._empty_schema())

        # Seed so the first MERGE has a non-empty target — exercises the
        # WHEN MATCHED branch on at least one writer.
        seed = self._writer_chunk(-1, self.ROWS_PER_WRITER, key_offset=0)
        table.insert(seed, mode=Mode.OVERWRITE)

        def upsert(writer_id: int) -> None:
            chunk = self._writer_chunk(
                writer_id, self.ROWS_PER_WRITER, key_offset=0,
            )
            table.insert(chunk, mode=Mode.UPSERT, match_by=["id"])

        self._run_in_parallel(
            upsert, [(i,) for i in range(self.PARALLELISM)],
        )

        rows = self.engine.execute(
            f"SELECT id, writer FROM {table.full_name(safe=True)} ORDER BY id"
        ).to_arrow_table().to_pylist()

        self.assertEqual(len(rows), self.ROWS_PER_WRITER)
        self.assertEqual(
            [r["id"] for r in rows], list(range(self.ROWS_PER_WRITER)),
        )
        valid_writers = set(range(self.PARALLELISM))
        for r in rows:
            self.assertIn(
                r["writer"], valid_writers,
                f"row {r!r} kept writer={r['writer']!r} which never wrote",
            )

    # ------------------------------------------------------------------
    # Mixed APPEND + UPSERT — neither writer should lose rows
    # ------------------------------------------------------------------

    def test_concurrent_mixed_append_and_upsert(self) -> None:
        """One writer appends a disjoint key range; another upserts an
        overlapping range. After both commit, the appender's keys are
        all present and the upsert range has exactly one row per key."""
        table = self._unique_table("concurrent_mixed")
        self._ensure_table(table, self._empty_schema())

        # Seed with keys [0, ROWS_PER_WRITER) so the upsert path has
        # something to update.
        seed = self._writer_chunk(0, self.ROWS_PER_WRITER, key_offset=0)
        table.insert(seed, mode=Mode.OVERWRITE)

        def appender() -> None:
            chunk = self._writer_chunk(
                100, self.ROWS_PER_WRITER,
                key_offset=10 * self.ROWS_PER_WRITER,
            )
            table.insert(chunk, mode=Mode.APPEND)

        def upserter() -> None:
            chunk = self._writer_chunk(
                200, self.ROWS_PER_WRITER, key_offset=0,
            )
            table.insert(chunk, mode=Mode.UPSERT, match_by=["id"])

        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(appender), pool.submit(upserter)]
            for fut in cf.as_completed(futures):
                fut.result()

        # Total rows = original seed (all upserted in place) + appended chunk.
        count = self.engine.execute(
            f"SELECT COUNT(*) AS n FROM {table.full_name(safe=True)}"
        ).to_arrow_table().column("n")[0].as_py()
        self.assertEqual(count, 2 * self.ROWS_PER_WRITER)

        # Upsert range survived as a single row per key, all written by 200.
        upserted = self.engine.execute(
            f"SELECT writer FROM {table.full_name(safe=True)} "
            f"WHERE id < {self.ROWS_PER_WRITER}"
        ).to_arrow_table().column("writer").to_pylist()
        self.assertEqual(len(upserted), self.ROWS_PER_WRITER)
        self.assertTrue(all(w == 200 for w in upserted))

        # Append range fully landed.
        appended = self.engine.execute(
            f"SELECT COUNT(*) AS n FROM {table.full_name(safe=True)} "
            f"WHERE id >= {10 * self.ROWS_PER_WRITER}"
        ).to_arrow_table().column("n")[0].as_py()
        self.assertEqual(appended, self.ROWS_PER_WRITER)
