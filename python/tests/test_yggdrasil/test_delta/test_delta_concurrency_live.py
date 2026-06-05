"""Live Databricks integration tests for Delta optimistic-concurrency rebase
and clustering-aware pruning.

These exercise the two enhancements end-to-end against a real external
Delta table on S3, so the commit semantics are validated against what
Databricks itself reads/writes — not just the local fixture:

- **Concurrent DeltaFolder appends converge** — N threads append to one
  external table; every commit lands (version advances by exactly N, all
  rows present, no lost writes), proving the rebase commutes concurrent
  blind appends rather than dropping the losers.
- **A true conflict still raises** — a concurrent OVERWRITE against a
  table another writer just changed surfaces ``ConcurrentDeltaCommitError``
  with a logical-conflict reason.
- **DeltaFolder append interleaved with a Databricks SQL INSERT** — both
  land and both are visible to both readers (extends the cross-write
  coverage in ``test_delta_databricks.py``).
- **Clustering-aware pruning** — a ``CLUSTER BY`` table written via SQL +
  INSERT surfaces its clustering columns on the snapshot, and a predicate
  on a clustering column prunes files while matching a full scan exactly.

Requires:
    DATABRICKS_HOST, DATABRICKS_TOKEN (or auth profile)
    External-location CREATE grant. Set ``YGG_TEST_EXTERNAL_LOCATION`` to a
    writable base prefix; otherwise these skip cleanly (an environment
    grant, not a code defect).

Run:
    python -m pytest tests/test_yggdrasil/test_delta/test_delta_concurrency_live.py \\
        -v -s -m integration
"""
from __future__ import annotations

import os
import secrets
import threading
import unittest

import pyarrow as pa
import pytest

from tests.test_yggdrasil.test_delta.test_delta_databricks import (
    _DeltaSQLBase, _has_databricks,
)

def _external_base() -> "str | None":
    return os.environ.get("YGG_TEST_EXTERNAL_LOCATION") or None


@pytest.mark.integration
@unittest.skipUnless(_has_databricks(), "DATABRICKS_HOST not set")
class TestConcurrencyLive(_DeltaSQLBase):
    """Concurrent DeltaFolder writes against one real external Delta table."""

    def _external_delta_table(self, tag: str, definition):
        from databricks.sdk.errors import DatabricksError
        from databricks.sdk.errors.platform import PermissionDenied
        from databricks.sdk.service.catalog import TableType

        base = _external_base()
        if not base:
            raise unittest.SkipTest("no external-location grant (set YGG_TEST_EXTERNAL_LOCATION)")
        tbl = self._table(tag)
        runid = secrets.token_hex(4)
        location = f"{base.rstrip('/')}/{runid}/{tbl.table_name}"
        try:
            tbl.create(definition, table_type=TableType.EXTERNAL, storage_location=location)
        except (DatabricksError, PermissionDenied, NotImplementedError, ValueError) as exc:
            raise unittest.SkipTest(f"cannot provision external table: {exc}") from exc
        return tbl

    def _purge(self, tbl) -> None:
        try:
            sp = tbl.storage_path()
            if sp is not None:
                sp.remove(recursive=True, missing_ok=True)
        except Exception:
            pass

    def test_concurrent_appends_converge(self) -> None:
        from yggdrasil.data.schema import Schema, field
        from yggdrasil.enums import Mode
        from yggdrasil.io.delta import DeltaOptions

        definition = Schema([field("id", "int64"), field("tid", "int64")])
        tbl = self._external_delta_table("conc_app", definition)
        try:
            # Seed a base commit so all threads contend on top of it. The
            # CREATE may already have written v0, so capture the post-seed
            # version and assert the *delta* rather than an absolute number.
            seed = self._delta_folder(tbl)
            seed.write_arrow_table(
                pa.table({"id": pa.array([-1], pa.int64()),
                          "tid": pa.array([-1], pa.int64())}),
                mode=Mode.APPEND,
            )
            base_version = seed.snapshot(fresh=True).version

            threads, appends = 4, 5
            errors: list[BaseException] = []

            def _worker(t: int) -> None:
                # Each thread opens its own DeltaFolder over the same storage.
                d = self._delta_folder(tbl)
                for i in range(appends):
                    key = t * 1000 + i
                    try:
                        d.write_arrow_batches(
                            pa.table({"id": pa.array([key], pa.int64()),
                                      "tid": pa.array([t], pa.int64())}).to_batches(),
                            options=DeltaOptions(mode=Mode.APPEND,
                                                 checkpoint_interval=0,
                                                 commit_max_retries=200,
                                                 commit_retry_backoff=0.02,
                                                 commit_retry_jitter=0.05,
                                                 commit_retry_max_delay=1.0),
                        )
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc); return

            ts = [threading.Thread(target=_worker, args=(t,)) for t in range(threads)]
            for t in ts: t.start()
            for t in ts: t.join()
            self.assertEqual(errors, [])

            snap = self._delta_folder(tbl).snapshot(fresh=True)
            # Version advanced by exactly one per concurrent commit — every
            # racing append landed, none clobbered another (atomic commit +
            # rebase).
            self.assertEqual(snap.version, base_version + threads * appends)

            out = self._delta_folder(tbl).read_arrow_table()
            ids = sorted(out.column("id").to_pylist())
            expected = sorted([-1] + [t * 1000 + i for t in range(threads)
                                      for i in range(appends)])
            self.assertEqual(ids, expected)  # no lost writes
        finally:
            self._purge(tbl)

    def test_concurrent_overwrite_raises_logical_conflict(self) -> None:
        from yggdrasil.data.schema import Schema, field
        from yggdrasil.enums import Mode
        from yggdrasil.io.delta import ConcurrentDeltaCommitError, DeltaOptions
        import yggdrasil.io.delta.delta_folder as df

        definition = Schema([field("id", "int64")])
        tbl = self._external_delta_table("conc_ow", definition)
        try:
            d = self._delta_folder(tbl)
            d.write_arrow_table(pa.table({"id": pa.array([1, 2, 3], pa.int64())}),
                                mode=Mode.APPEND)

            # Force our overwrite onto the rebase path: fail its first atomic
            # create after a rival append has landed.
            other = self._delta_folder(tbl)
            attempts = {"n": 0}
            orig = d._commit_atomic

            def _flaky(version, actions):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    other.write_arrow_batches(
                        pa.table({"id": pa.array([99], pa.int64())}).to_batches(),
                        options=DeltaOptions(mode=Mode.APPEND, checkpoint_interval=0),
                    )
                    d.refresh()
                    raise FileExistsError(f"race at v{version}")
                return orig(version, actions)

            d._commit_atomic = _flaky  # type: ignore[assignment]
            with self.assertRaises(ConcurrentDeltaCommitError) as ctx:
                d.write_arrow_batches(
                    pa.table({"id": pa.array([1000], pa.int64())}).to_batches(),
                    options=DeltaOptions(mode=Mode.OVERWRITE, checkpoint_interval=0,
                                         commit_retry_backoff=0),
                )
            self.assertEqual(ctx.exception.conflict, "overwrite-vs-concurrent-write")
        finally:
            self._purge(tbl)

    def test_deltafolder_append_interleaved_with_sql_insert(self) -> None:
        from yggdrasil.data.schema import Schema, field
        from yggdrasil.enums import Mode

        definition = Schema([field("id", "int64")])
        tbl = self._external_delta_table("conc_xwrite", definition)
        try:
            d = self._delta_folder(tbl)
            d.write_arrow_table(pa.table({"id": pa.array([1, 2], pa.int64())}),
                                mode=Mode.APPEND)
            # Databricks SQL appends a version on top of ours.
            self._execute(f"INSERT INTO {tbl.full_name()} VALUES (3), (4)")
            # Our DeltaFolder appends again, rebasing past the SQL commit.
            d.refresh().write_arrow_batches(
                pa.table({"id": pa.array([5], pa.int64())}).to_batches(),
                mode=Mode.APPEND,
            )
            # Both readers see all rows.
            sql_ids = sorted(r["id"] for r in self._read_sql_arrow(
                f"SELECT id FROM {tbl.full_name()}").to_pylist())
            ygg_ids = sorted(self._delta_folder(tbl).read_arrow_table()
                             .column("id").to_pylist())
            self.assertEqual(sql_ids, [1, 2, 3, 4, 5])
            self.assertEqual(ygg_ids, [1, 2, 3, 4, 5])
        finally:
            self._purge(tbl)


@pytest.mark.integration
@unittest.skipUnless(_has_databricks(), "DATABRICKS_HOST not set")
class TestClusteringPruningLive(_DeltaSQLBase):
    """``CLUSTER BY`` tables: clustering metadata + clustering-aware pruning."""

    def _clustered_external(self, tag: str):
        base = _external_base()
        if not base:
            raise unittest.SkipTest("no external-location grant (set YGG_TEST_EXTERNAL_LOCATION)")
        tbl = self._table(tag)
        runid = secrets.token_hex(4)
        loc = f"{base.rstrip('/')}/{runid}/{tbl.table_name}"
        try:
            self._execute(
                f"CREATE TABLE {tbl.full_name()} (id BIGINT, region STRING, val STRING) "
                f"USING DELTA CLUSTER BY (region, id) LOCATION '{loc}'"
            )
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"cannot provision clustered external table: {exc}") from exc
        return tbl

    def _purge(self, tbl) -> None:
        try:
            sp = tbl.storage_path()
            if sp is not None:
                sp.remove(recursive=True, missing_ok=True)
        except Exception:
            pass

    def test_clustering_columns_surface_from_sql_table(self) -> None:
        tbl = self._clustered_external("clu_meta")
        try:
            for region in ("us", "eu", "ap"):
                vals = ",".join(f"({i}, '{region}', 'v{i}')" for i in range(10))
                self._execute(f"INSERT INTO {tbl.full_name()} VALUES {vals}")
            snap = self._delta_folder(tbl).snapshot(fresh=True)
            self.assertEqual(snap.clustering_columns, ["region", "id"])
        finally:
            self._purge(tbl)

    def test_clustering_predicate_prunes_and_matches_full_scan(self) -> None:
        from yggdrasil.execution.expr import col
        from yggdrasil.io.delta import DeltaOptions
        import yggdrasil.io.delta.delta_folder as df

        tbl = self._clustered_external("clu_prune")
        try:
            # Separate single-region inserts so files are region-disjoint
            # (liquid clustering co-locates by region in practice).
            for region in ("us", "eu", "ap"):
                vals = ",".join(f"({i}, '{region}', 'v{i}')" for i in range(20))
                self._execute(f"INSERT INTO {tbl.full_name()} VALUES {vals}")

            folder = self._delta_folder(tbl)
            total_files = folder.snapshot(fresh=True).num_active_files()

            seen = {}
            orig = df._data_skip_adds
            def _spy(snap, adds, predicate):
                kept = list(orig(snap, adds, predicate))
                seen["kept"] = len(kept)
                return iter(kept)
            df._data_skip_adds = _spy
            try:
                pruned = folder.read_arrow_table(
                    options=DeltaOptions(predicate=col("region") == "us"))
            finally:
                df._data_skip_adds = orig

            # Pruned read opened fewer files than the table holds.
            self.assertLess(seen["kept"], total_files)
            # And the result equals a full-scan filter.
            full = folder.read_arrow_table()
            manual = sorted(i for i, r in zip(full.column("id").to_pylist(),
                                              full.column("region").to_pylist())
                            if r == "us")
            self.assertEqual(sorted(pruned.column("id").to_pylist()), manual)
            self.assertEqual(set(pruned.column("region").to_pylist()), {"us"})
        finally:
            self._purge(tbl)
