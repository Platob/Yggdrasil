"""Databricks-compatibility guards for :class:`DeltaFolder`.

These tests pin the on-disk shape that makes yggdrasil's Delta output
readable by Databricks SQL / Photon and the ``deltalake`` Rust reader,
learned from diffing against real Databricks-generated tables:

- **Timestamps are stored as parquet ``TIMESTAMP(MICROS)``.** Databricks'
  reader rejects nanosecond timestamps outright
  (``Unsupported time unit in Parquet TimestampType``); the writer must
  down-coerce ``timestamp[ns]`` / ``timestamp[s]`` to microseconds while
  preserving the zone.
- **AddFile stats match the Delta data-skipping format** — timestamp
  min/max as ISO-8601 with millisecond precision (trailing ``Z`` for the
  UTC-anchored ``timestamp`` type, none for ``timestamp_ntz``), plus a
  ``tightBounds`` flag.
- **tz-naive → ``timestamp_ntz``, tz-aware → ``timestamp``** in the Delta
  schema string, matching the ``deltalake`` writer.

The live cross-read against a real workspace lives in
``test_delta_databricks_live.py`` (gated on ``DATABRICKS_HOST``).
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.io.delta.tests import DeltaTestCase


class TestTimestampPhysicalType(DeltaTestCase):
    def _written_parquet_schema(self, table: pa.Table) -> pa.Schema:
        d = self.new_table(table, name="ts")
        snap = d.snapshot(fresh=True)
        add = next(iter(snap.active_files.values()))
        leaf = (d.path / add.path)
        return pq.read_schema(leaf.full_path())

    def test_nanosecond_timestamp_written_as_micros(self) -> None:
        # The reference shape: id + a tz-naive ns timestamp.
        t = self.pa.table({
            "id": self.pa.array(["a", "b"], self.pa.string()),
            "updated_at": self.pa.array([0, 86_400_000_000_000],
                                        self.pa.timestamp("ns")),
        })
        schema = self._written_parquet_schema(t)
        ts = schema.field("updated_at").type
        self.assertTrue(pa.types.is_timestamp(ts))
        self.assertEqual(ts.unit, "us")
        self.assertIsNone(ts.tz)

    def test_second_timestamp_written_as_micros(self) -> None:
        t = self.pa.table({
            "id": self.pa.array(["a"], self.pa.string()),
            "ts": self.pa.array([1], self.pa.timestamp("s", tz="UTC")),
        })
        schema = self._written_parquet_schema(t)
        ts = schema.field("ts").type
        self.assertEqual(ts.unit, "us")
        self.assertEqual(ts.tz, "Etc/UTC")

    def test_micros_timestamp_unchanged(self) -> None:
        t = self.pa.table({
            "ts": self.pa.array([1, 2], self.pa.timestamp("us", tz="UTC")),
        })
        schema = self._written_parquet_schema(t)
        self.assertEqual(schema.field("ts").type.unit, "us")

    def test_timestamp_values_preserved_through_coercion(self) -> None:
        t = self.pa.table({
            "ts": self.pa.array([0, 86_400_000_000_000],
                                self.pa.timestamp("ns", tz="UTC")),
        })
        d = self.new_table(t, name="tsval")
        out = d.read_arrow_table()
        # ns -> us is exact for whole-day values; verify the instants survive.
        got = out.column("ts").cast(self.pa.timestamp("us", tz="UTC")).to_pylist()
        self.assertEqual(
            [v.replace(tzinfo=None) for v in got],
            [v.replace(tzinfo=None)
             for v in t.column("ts").cast(self.pa.timestamp("us", tz="UTC")).to_pylist()],
        )


class TestStatsFormat(DeltaTestCase):
    def _stats(self, table: pa.Table) -> dict:
        d = self.new_table(table, name="stats")
        snap = d.snapshot(fresh=True)
        add = next(iter(snap.active_files.values()))
        return json.loads(add.stats)

    def test_tight_bounds_flag_present(self) -> None:
        stats = self._stats(self.pa.table({"id": [1, 2, 3]}))
        self.assertIs(stats.get("tightBounds"), True)

    def test_aware_timestamp_stats_iso_millis_with_z(self) -> None:
        t = self.pa.table({
            "ts": self.pa.array([0, 86_400_000_000],
                                self.pa.timestamp("us", tz="UTC")),
        })
        stats = self._stats(t)
        self.assertEqual(stats["minValues"]["ts"], "1970-01-01T00:00:00.000Z")
        self.assertEqual(stats["maxValues"]["ts"], "1970-01-02T00:00:00.000Z")

    def test_naive_timestamp_stats_iso_millis_no_z(self) -> None:
        t = self.pa.table({
            "ts": self.pa.array([0, 86_400_000_000], self.pa.timestamp("us")),
        })
        stats = self._stats(t)
        self.assertEqual(stats["minValues"]["ts"], "1970-01-01T00:00:00.000")
        self.assertEqual(stats["maxValues"]["ts"], "1970-01-02T00:00:00.000")
        self.assertNotIn("Z", stats["maxValues"]["ts"])

    def test_date_stats_iso(self) -> None:
        import datetime
        t = self.pa.table({
            "d": self.pa.array([datetime.date(2020, 1, 1),
                                datetime.date(2020, 12, 31)], self.pa.date32()),
        })
        stats = self._stats(t)
        self.assertEqual(stats["minValues"]["d"], "2020-01-01")
        self.assertEqual(stats["maxValues"]["d"], "2020-12-31")

    def test_null_counts_emitted(self) -> None:
        t = self.pa.table({
            "v": self.pa.array([1.0, None, 3.0, None], self.pa.float64()),
        })
        stats = self._stats(t)
        self.assertEqual(stats["nullCount"]["v"], 2)
        self.assertEqual(stats["numRecords"], 4)


class TestSchemaStringTimestampMapping(DeltaTestCase):
    def _schema_string(self, table: pa.Table) -> dict:
        d = self.new_table(table, name="sch")
        return json.loads(d.snapshot(fresh=True).metadata.schema_string)

    def test_naive_timestamp_maps_to_timestamp_ntz(self) -> None:
        t = self.pa.table({"ts": self.pa.array([0], self.pa.timestamp("us"))})
        fields = {f["name"]: f["type"] for f in self._schema_string(t)["fields"]}
        self.assertEqual(fields["ts"], "timestamp_ntz")

    def test_utc_timestamp_maps_to_timestamp(self) -> None:
        t = self.pa.table({"ts": self.pa.array([0], self.pa.timestamp("us", tz="UTC"))})
        fields = {f["name"]: f["type"] for f in self._schema_string(t)["fields"]}
        self.assertEqual(fields["ts"], "timestamp")


class TestDataSkipping(DeltaTestCase):
    """Per-file stats data-skipping never changes results, only file reads.

    A predicate that no file's min/max can satisfy returns zero rows
    without opening parquet; a partial predicate prunes the
    non-overlapping files while the surviving rows match an unpruned
    full scan exactly.
    """

    def _multi_file_table(self):
        # Three single-file commits with disjoint id ranges so each AddFile
        # carries non-overlapping min/max stats.
        d = self.delta_io("skip")
        d.write_arrow_table(self.pa.table({
            "id": self.pa.array([1, 2, 3], self.pa.int64()),
            "v": self.pa.array(["a", "b", "c"], self.pa.string()),
        }))
        d.write_arrow_table(self.pa.table({
            "id": self.pa.array([10, 11, 12], self.pa.int64()),
            "v": self.pa.array(["d", "e", "f"], self.pa.string()),
        }))
        d.write_arrow_table(self.pa.table({
            "id": self.pa.array([100, 101, 102], self.pa.int64()),
            "v": self.pa.array(["g", "h", "i"], self.pa.string()),
        }))
        return d

    def test_skipping_matches_full_scan_gt(self) -> None:
        from yggdrasil.saga.expr import col
        from yggdrasil.io.delta import DeltaOptions
        d = self._multi_file_table()
        out = d.read_arrow_table(options=DeltaOptions(predicate=col("id") > 50))
        self.assertEqual(sorted(out.column("id").to_pylist()), [100, 101, 102])

    def test_skipping_matches_full_scan_eq(self) -> None:
        from yggdrasil.saga.expr import col
        from yggdrasil.io.delta import DeltaOptions
        d = self._multi_file_table()
        out = d.read_arrow_table(options=DeltaOptions(predicate=col("id") == 11))
        self.assertEqual(out.column("id").to_pylist(), [11])

    def test_skipping_empty_when_no_file_matches(self) -> None:
        from yggdrasil.saga.expr import col
        from yggdrasil.io.delta import DeltaOptions
        d = self._multi_file_table()
        out = d.read_arrow_table(options=DeltaOptions(predicate=col("id") > 1000))
        self.assertEqual(out.num_rows, 0)

    def test_skipping_range_between(self) -> None:
        from yggdrasil.saga.expr import col
        from yggdrasil.io.delta import DeltaOptions
        d = self._multi_file_table()
        out = d.read_arrow_table(
            options=DeltaOptions(predicate=(col("id") >= 10) & (col("id") <= 12)))
        self.assertEqual(sorted(out.column("id").to_pylist()), [10, 11, 12])

    def test_extract_constraints_conjunction(self) -> None:
        from yggdrasil.saga.expr import col
        from yggdrasil.io.delta.delta_folder import _extract_range_constraints
        c = _extract_range_constraints((col("id") > 5) & (col("v") == "x"))
        self.assertIn("id", c)
        self.assertIn("v", c)
        self.assertIn((">", 5), c["id"])

    def test_extract_constraints_disjunction_drops_bounds(self) -> None:
        from yggdrasil.saga.expr import col
        from yggdrasil.io.delta.delta_folder import _extract_range_constraints
        # OR must not yield a binding bound — either branch could match.
        c = _extract_range_constraints((col("id") > 5) | (col("id") < 1))
        self.assertIsNone(c)


class TestClusteringAwarePruning(DeltaTestCase):
    """Liquid-clustering metadata strengthens (and never weakens) pruning.

    A ``CLUSTER BY`` table written through DeltaFolder must stamp the
    ``delta.clustering`` domain-metadata so ``Snapshot.clustering_columns``
    round-trips, always stat the clustering columns even past the indexed-col
    cap, and prune files on a clustering-column predicate — identical results
    to a full scan, fewer files opened.
    """

    def _clustered_table(self, *, region_col_position_last: bool = False):
        from yggdrasil.io.delta import DeltaOptions
        d = self.delta_io("clustered")
        opts = DeltaOptions(mode=self.Mode.APPEND, cluster_by=("region",))
        # Three single-file commits, each a disjoint region — clustering
        # co-locates rows by region, which is exactly what we simulate by
        # giving each file one region value.
        for region, ids in (("ap", [1, 2]), ("eu", [10, 11]), ("us", [100, 101])):
            d.write_arrow_batches(self.pa.table({
                "id": self.pa.array(ids, self.pa.int64()),
                "region": self.pa.array([region] * len(ids), self.pa.string()),
                "v": self.pa.array(["x"] * len(ids), self.pa.string()),
            }).to_batches(), options=opts)
        return d

    @property
    def Mode(self):
        from yggdrasil.enums import Mode
        return Mode

    def test_cluster_by_round_trips_via_domain_metadata(self) -> None:
        d = self._clustered_table()
        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.clustering_columns, ["region"])
        # The writer feature + reader/writer versions Databricks requires.
        self.assertIn("clustering", snap.protocol.writer_features)
        self.assertIn("domainMetadata", snap.protocol.writer_features)
        self.assertGreaterEqual(snap.protocol.min_writer_version, 7)

    def test_clustering_column_past_cap_still_stat(self) -> None:
        # A clustering column beyond the indexed-col cap must still be
        # stat'd — that's the stat the pruner leans on.
        import json
        from yggdrasil.io.delta import DeltaOptions
        d = self.delta_io("wide")
        # 3 leading cols + region as the 4th; cap at 2 drops cols 2,3 but
        # region (clustering) is kept regardless.
        tbl = self.pa.table({
            "a": self.pa.array([1], self.pa.int64()),
            "b": self.pa.array([2], self.pa.int64()),
            "c": self.pa.array([3], self.pa.int64()),
            "region": self.pa.array(["eu"], self.pa.string()),
        })
        d.write_arrow_batches(tbl.to_batches(), options=DeltaOptions(
            mode=self.Mode.APPEND, cluster_by=("region",), stats_num_indexed_cols=2))
        snap = d.snapshot(fresh=True)
        stats = json.loads(next(iter(snap.active_files.values())).stats)
        self.assertIn("region", stats["minValues"])  # past the cap, still stat'd
        self.assertNotIn("c", stats["minValues"])     # past the cap, dropped

    def test_clustering_predicate_prunes_files(self) -> None:
        import yggdrasil.io.delta.delta_folder as df
        from yggdrasil.saga.expr import col
        from yggdrasil.io.delta import DeltaOptions
        d = self._clustered_table()

        seen = {}
        orig = df._data_skip_adds
        def _spy(snap, adds, predicate):
            kept = list(orig(snap, adds, predicate))
            seen["kept"] = len(kept)
            return iter(kept)
        df._data_skip_adds = _spy
        try:
            out = d.read_arrow_table(
                options=DeltaOptions(predicate=col("region") == "us"))
        finally:
            df._data_skip_adds = orig

        # Pruned to the single 'us' file; result equals a full-scan filter.
        self.assertEqual(seen["kept"], 1)
        self.assertEqual(sorted(out.column("id").to_pylist()), [100, 101])
        self.assertEqual(set(out.column("region").to_pylist()), {"us"})

    def test_clustering_pruning_matches_full_scan(self) -> None:
        from yggdrasil.saga.expr import col
        from yggdrasil.io.delta import DeltaOptions
        d = self._clustered_table()
        pruned = d.read_arrow_table(
            options=DeltaOptions(predicate=col("region") == "eu"))
        full = d.read_arrow_table()
        manual = [i for i, r in zip(full.column("id").to_pylist(),
                                    full.column("region").to_pylist()) if r == "eu"]
        self.assertEqual(sorted(pruned.column("id").to_pylist()), sorted(manual))
