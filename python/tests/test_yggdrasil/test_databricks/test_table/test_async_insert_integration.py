"""Live-integration tests for the async-insert drop → load pipeline.

Covers the three supported save modes end to end against a real Unity Catalog
table:

- **APPEND** — staged rows accumulate across drops.
- **OVERWRITE** — the latest drop supersedes everything staged before it.
- **MERGE / UPSERT** — keyed drops update matched rows and insert new ones.

Two surfaces are exercised:

- the **loader driven directly** (:func:`load_async`) — fast, deterministic data
  assertions for every mode (no serverless job in the loop);
- the **deployed file-arrival job** end to end (:meth:`Table.async_job` builds +
  uploads the ygg wheel, :meth:`Job.run` triggers + waits) — the real
  "drop an insert, the job loads it" path, plus a check that the run's captured
  stdout is reachable for debugging.

Skipped wholesale unless ``DATABRICKS_HOST`` is set. The deployed-job test
additionally skips when the wheel can't be built (no ``pip`` / no index access
in the test environment). Tables + jobs are throw-away and cleaned up.
"""
from __future__ import annotations

import secrets
import subprocess
import unittest
from typing import ClassVar

import pyarrow as pa
from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied

from yggdrasil.enums import Mode

from .. import DatabricksIntegrationCase

_CATALOG = "trading"
_SCHEMA = "unittest"
_WAIT_SECONDS = 420

_SCHEMA_FIELDS = pa.schema(
    [("id", pa.int64()), ("label", pa.string()), ("amount", pa.float64())]
)


def _data(rows: list[tuple[int, str, float]]) -> pa.Table:
    return pa.table(
        {
            "id": pa.array([r[0] for r in rows], pa.int64()),
            "label": pa.array([r[1] for r in rows], pa.string()),
            "amount": pa.array([r[2] for r in rows], pa.float64()),
        }
    )


class _AsyncFixture(DatabricksIntegrationCase):
    catalog_name: ClassVar[str] = _CATALOG
    schema_name: ClassVar[str] = _SCHEMA

    def setUp(self) -> None:
        super().setUp()
        self._tables: list[str] = []
        self._jobs: list[int] = []

    def tearDown(self) -> None:
        for full in self._tables:
            try:
                self.client.tables.table(full).delete(missing_ok=True)
            except Exception:
                pass
        for job_id in self._jobs:
            try:
                self.client.workspace_client().jobs.delete(job_id=job_id)
            except Exception:
                pass
        super().tearDown()

    def _new_table(self):
        full = f"{self.catalog_name}.{self.schema_name}.yg_async_{secrets.token_hex(4)}"
        self._tables.append(full)
        try:
            tbl = self.client.tables.table(full)
            tbl.ensure_created(_SCHEMA_FIELDS)
            tbl.insert(_data([]), mode=Mode.OVERWRITE)  # start empty
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"cannot create async test table: {exc}")
        return tbl

    def _rows(self, tbl) -> dict[int, tuple[str, float]]:
        res = self.client.sql(
            catalog_name=self.catalog_name, schema_name=self.schema_name,
        ).execute(
            f"SELECT id, label, amount FROM {tbl.full_name(safe=True)} ORDER BY id"
        ).to_arrow_table()
        return {r["id"]: (r["label"], r["amount"]) for r in res.to_pylist()}


@unittest.skipUnless(__import__("os").getenv("DATABRICKS_HOST"), "needs DATABRICKS_HOST")
class TestAsyncInsertModes(_AsyncFixture):
    """append / overwrite / merge data outcomes, loader driven."""

    def test_append_overwrite_merge_round_trip(self):
        from yggdrasil.databricks.table.insert import (
            load_async, logs_path, stage_async_insert,
        )

        tbl = self._new_table()

        def _load():
            return load_async(self.client.tables, logs_path(tbl), wait=True)

        try:
            # APPEND — rows land
            stage_async_insert(tbl, _data([(1, "a", 1.0), (2, "b", 2.0)]), mode=Mode.APPEND)
            _load()
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"async insert needs volume write access: {exc}")
        self.assertEqual(self._rows(tbl), {1: ("a", 1.0), 2: ("b", 2.0)})

        # APPEND again — accumulates
        stage_async_insert(tbl, _data([(3, "c", 3.0)]), mode=Mode.APPEND)
        _load()
        self.assertEqual(set(self._rows(tbl)), {1, 2, 3})

        # OVERWRITE — supersedes everything before it
        stage_async_insert(tbl, _data([(9, "z", 9.0)]), mode=Mode.OVERWRITE)
        _load()
        self.assertEqual(self._rows(tbl), {9: ("z", 9.0)})

        # MERGE/UPSERT on id — update 9, insert 10
        stage_async_insert(
            tbl, _data([(9, "Z", 99.0), (10, "k", 10.0)]),
            mode=Mode.MERGE, match_by=["id"],
        )
        _load()
        self.assertEqual(self._rows(tbl), {9: ("Z", 99.0), 10: ("k", 10.0)})

    def test_insert_wait_false_routes_merge_to_async(self):
        # Table.insert(wait=False, mode=MERGE, match_by=...) returns the staged
        # op-log path (async path), not None (the synchronous return).
        from yggdrasil.databricks.fs.volume_path import VolumePath
        from yggdrasil.databricks.table.insert import load_async, logs_path

        tbl = self._new_table()
        try:
            tbl.insert(_data([(1, "a", 1.0)]), mode=Mode.APPEND, wait=False)
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"async insert needs volume write access: {exc}")
        log = tbl.insert(
            _data([(1, "A", 11.0), (2, "b", 2.0)]),
            mode=Mode.MERGE, match_by=["id"], wait=False,
        )
        self.assertIsInstance(log, VolumePath)
        load_async(self.client.tables, logs_path(tbl), wait=True)
        self.assertEqual(self._rows(tbl), {1: ("A", 11.0), 2: ("b", 2.0)})

    def test_async_merge_requires_keys(self):
        from yggdrasil.databricks.table.insert import stage_async_insert

        tbl = self._new_table()
        with self.assertRaises(ValueError):
            stage_async_insert(tbl, _data([(1, "a", 1.0)]), mode=Mode.MERGE)

    def test_merge_on_partitioned_table_is_correct(self):
        # A keyed merge on a partitioned target auto-derives the partition cols
        # and inlines a literal IN filter on the MERGE ON (pruning the scan) —
        # the result must still be correct: matched rows update, new rows insert.
        from yggdrasil.databricks.table.insert import (
            load_async, logs_path, stage_async_insert,
        )

        name = f"{self.catalog_name}.{self.schema_name}.yg_part_{secrets.token_hex(4)}"
        self._tables.append(name)
        sql = self.client.sql(
            catalog_name=self.catalog_name, schema_name=self.schema_name,
        )

        def _pdata(rows):
            return pa.table({
                "id": pa.array([r[0] for r in rows], pa.int64()),
                "d": pa.array([r[1] for r in rows], pa.string()),
                "v": pa.array([r[2] for r in rows], pa.float64()),
            })

        try:
            sql.execute(
                f"CREATE TABLE {name} (id BIGINT, d STRING, v DOUBLE) PARTITIONED BY (d)"
            )
            sql.execute(
                f"INSERT INTO {name} VALUES (1,'2024-01-01',1.0),(2,'2024-01-02',2.0)"
            )
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"cannot create partitioned table: {exc}")

        tbl = self.client.tables.table(name)
        # the partition filter is literal (no subquery — invalid in a MERGE ON),
        # derived by listing the source's distinct partition values
        self.assertEqual(
            tbl.merge_partition_filters("SELECT 'x' AS d"),
            ["T.`d` IN ('x')"],
        )

        stage_async_insert(
            tbl, _pdata([(2, "2024-01-02", 22.0), (3, "2024-01-03", 33.0)]),
            mode=Mode.MERGE, match_by=["id"],
        )
        # prune_partitions=True → the loader lists distinct values + filters the
        # MERGE (as the deployed job does); the live path would skip it.
        load_async(self.client.tables, logs_path(tbl), wait=True, prune_partitions=True)

        rows = {
            x["id"]: (x["d"], x["v"])
            for x in sql.execute(
                f"SELECT id, d, v FROM {name} ORDER BY id"
            ).to_arrow_table().to_pylist()
        }
        self.assertEqual(rows, {
            1: ("2024-01-01", 1.0),     # untouched
            2: ("2024-01-02", 22.0),    # matched → updated
            3: ("2024-01-03", 33.0),    # new → inserted
        })

    def test_several_merge_file_arrivals_aggregate_latest_wins(self):
        # Several independent async merge drops (each its own "file arrival" /
        # op-log) targeting one table are aggregated by the loader into a single
        # MERGE INTO over the deduped union. Keys staged in more than one drop
        # resolve to the LATEST drop's row (incoming-wins, deterministic).
        from yggdrasil.databricks.table.insert import (
            load_async, logs_path, stage_async_insert,
        )

        tbl = self._new_table()
        try:
            tbl.insert(_data([(1, "seed1", 1.0), (2, "seed2", 2.0)]), mode=Mode.APPEND)
            # three drops, in arrival order; overlapping keys across them:
            #   id 3 appears in all three → drop C must win
            #   id 2 updated by A; id 1 updated by C; id 4 inserted by B
            stage_async_insert(
                tbl, _data([(2, "A2", 20.0), (3, "A3", 30.0)]),
                mode=Mode.MERGE, match_by=["id"],
            )
            stage_async_insert(
                tbl, _data([(3, "B3", 31.0), (4, "B4", 40.0)]),
                mode=Mode.MERGE, match_by=["id"],
            )
            stage_async_insert(
                tbl, _data([(1, "C1", 11.0), (3, "C3", 33.0)]),
                mode=Mode.MERGE, match_by=["id"],
            )
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"async insert needs volume write access: {exc}")

        # One aggregated load consumes all three op-logs.
        processed = load_async(self.client.tables, logs_path(tbl), wait=True)
        self.assertEqual(processed, 3)

        self.assertEqual(
            self._rows(tbl),
            {
                1: ("C1", 11.0),   # seeded, updated by drop C
                2: ("A2", 20.0),   # seeded, updated by drop A
                3: ("C3", 33.0),   # in A, B, C → latest (C) wins
                4: ("B4", 40.0),   # inserted by drop B
            },
        )


@unittest.skipUnless(__import__("os").getenv("DATABRICKS_HOST"), "needs DATABRICKS_HOST")
class TestAsyncInsertDeployedJob(_AsyncFixture):
    """The real deployed file-arrival loader job runs the staged insert."""

    def test_deployed_job_loads_staged_insert_and_exposes_stdout(self):
        from yggdrasil.databricks.table.insert import stage_async_insert

        tbl = self._new_table()
        try:
            stage_async_insert(tbl, _data([(1, "a", 1.0), (2, "b", 2.0)]), mode=Mode.APPEND)
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"async insert needs volume write access: {exc}")

        # Build + upload the ygg wheel and deploy the loader job. The wheel build
        # shells out to ``pip wheel`` — skip cleanly where that isn't available.
        try:
            job = tbl.async_job(rebuild=True)
        except (DatabricksError, PermissionDenied, FileNotFoundError,
                subprocess.CalledProcessError) as exc:
            self.skipTest(f"loader job deploy unavailable (wheel build): {exc}")
        self._jobs.append(job.job_id)

        # Trigger the loader now (instead of waiting on the file-arrival trigger)
        # and block until it finishes.
        run = job.run(wait=_WAIT_SECONDS, raise_error=False)

        # The convenience debug surface is always reachable — this is what the
        # test really guarantees: a single-node DAG and a printable dump that
        # carries per-task state + captured stderr.
        dump = run.debug()
        self.assertIn("async-load", dump)
        self.assertEqual(run.dag().keys, ["async-load"])

        if run.is_failed:
            # The ygg image ships the pure-python ygg wheel by path and resolves
            # its deps from the workspace index, so a serverless install should
            # succeed. If a workspace nonetheless can't resolve the image (no
            # index access for a dep, etc.) that's an environment limitation,
            # not an async-insert defect — skip with the cause debug() surfaced
            # rather than failing the suite.
            haystack = f"{run.stderr}\n{run.state_message}\n{dump}".lower()
            if "installation" in haystack or "wheel" in haystack:
                self.skipTest(f"serverless image install unavailable:\n{dump}")
            self.fail(f"loader run failed:\n{dump}")

        # On success: the staged insert was loaded by the job, and the loader
        # CLI task's stdout is reachable for debugging.
        self.assertEqual(self._rows(tbl), {1: ("a", 1.0), 2: ("b", 2.0)})
        self.assertTrue(run.stdout.strip(), "expected loader task stdout")
