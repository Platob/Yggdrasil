"""DeltaIO integration tests against a Unity Catalog **external** table.

External tables: the customer owns the storage location. Both the
SQL warehouse and our :class:`DeltaIO` can write to them safely as
long as they don't run concurrently.

This test file exercises the **full** read+write surface:

- SQL writes → DeltaIO reads (parity baseline).
- DeltaIO writes (APPEND, OVERWRITE) → SQL reads.
- Round-trip: SQL → DeltaIO read → DeltaIO append → SQL read.
- Deletion vectors emitted by either side.
- S3-content checks: log files, data files, DV ``.bin`` files.

Prerequisites
-------------

The external location at :data:`EXTERNAL_LOCATION` must already
exist in Unity Catalog (registered via ``CREATE EXTERNAL LOCATION``)
and the test runner's principal must have ``CREATE EXTERNAL TABLE``
permissions on it. Without that, ``setUpClass`` fails with a clear
error from the SQL warehouse.

This file is fully self-contained — no inheritance from
:mod:`test_managed`. Helpers are duplicated between the two files
on purpose: each file should read top-to-bottom without chasing
imports across modules.
"""

import pyarrow as pa

from yggdrasil.aws.fs.path import S3Path
from yggdrasil.io.buffer.nested.delta import DeltaIO, DeltaOptions
from yggdrasil.io.buffer.nested.delta.deletion_vector import (
    DeletionVectorDescriptor,
)
from yggdrasil.io.buffer.nested.delta.replay import (
    latest_commit_version,
    read_last_checkpoint,
)
from yggdrasil.io.enums import Mode
from yggdrasil.io.fs import Path
from yggdrasil.io.url import URL
from .._base import DatabricksCase


# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------

#: struct<a:int, s:string> — same schema used by the managed test file
#: so the parallel read tests are identical.
TABLE_SCHEMA: str = "'test': struct<a: int, s: string>"

#: Customer-owned S3 prefix where the external table's data lives.
#: Must be a registered UC external location.
EXTERNAL_LOCATION: str = "s3://my-test-bucket/yggdrasil-tests/"

#: External-table identity. Distinct from the managed test table so the
#: two suites can run in parallel against the same metastore.
EXTERNAL_TABLE_NAME: str = "trading_tgp_dev.src_monteleq.test_table_external"


class ExternalTableIntegrationCase(DatabricksCase):
    """Full read+write DeltaIO tests against a UC **external** Delta table.

    The table's data lives at :data:`EXTERNAL_LOCATION`; the SQL
    warehouse and our :class:`DeltaIO` are both supported as
    writers. Tests interleave SQL and DeltaIO writes to validate
    that each side produces output the other can read.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Sub-prefix per table so multiple test runs don't collide
        # at the same EXTERNAL_LOCATION root.
        location = (
            EXTERNAL_LOCATION.rstrip("/") + "/"
            + EXTERNAL_TABLE_NAME.split(".")[-1] + "/"
        )

        cls.table = (
            cls.client.sql.table(EXTERNAL_TABLE_NAME)
            .create(TABLE_SCHEMA, location=location)
        )
        # Best-effort wipe so the first test starts clean.
        try:
            cls.table.sql.execute(
                f"DELETE FROM {cls.table.full_name(safe=True)} WHERE true"
            )
        except Exception:
            pass

    # ==================================================================
    # Helpers
    # ==================================================================

    def _delta_io(self, write: bool = False) -> DeltaIO:
        """Get a DeltaIO pointed at the table's S3 storage."""
        from databricks.sdk.service.catalog import TableOperation

        op = TableOperation.READ_WRITE if write else TableOperation.READ
        path = self.table.storage_location(operation=op)
        return DeltaIO.from_path(path)

    def _truncate(self):
        """Wipe rows via SQL."""
        self.table.sql.execute(
            f"DELETE FROM {self.table.full_name(safe=True)} WHERE true"
        )

    def _sql_select_all(self) -> pa.Table:
        return self.table.to_arrow_dataset()

    # --- S3-content helpers --------------------------------------------

    def _storage_path(self, write: bool = False) -> S3Path:
        """The yggdrasil S3Path for the table's storage location."""
        from databricks.sdk.service.catalog import TableOperation

        op = TableOperation.READ_WRITE if write else TableOperation.READ
        path = self.table.storage_location(operation=op)
        self.assertIsInstance(
            path, S3Path,
            f"storage_location returned {type(path).__name__}, expected S3Path",
        )
        return path

    def _log_dir(self, write: bool = False) -> S3Path:
        return self._storage_path(write=write) / "_delta_log"

    def _list_keys(self, root: Path, *, recursive: bool = False) -> list[str]:
        return sorted(p.name for p in root.ls(recursive=recursive, allow_not_found=True))

    def _list_data_files(self) -> list[S3Path]:
        """Walk the storage tree; return every parquet leaf outside ``_delta_log/``."""
        root = self._storage_path()
        out: list[S3Path] = []
        for child in root.ls(recursive=True, allow_not_found=True):
            if "_delta_log" in child.url.path.split("/"):
                continue
            if child.name.lower().endswith(".parquet"):
                out.append(child)
        return out

    def _list_dv_files(self) -> list[S3Path]:
        """Walk the storage tree; return every ``.bin`` leaf — DV files."""
        root = self._storage_path()
        out: list[S3Path] = []
        for child in root.ls(recursive=True, allow_not_found=True):
            if child.name.lower().endswith(".bin"):
                out.append(child)
        return out

    # --- Inline assertions ---------------------------------------------

    def _assert_log_min_version(self, min_version: int) -> None:
        latest = latest_commit_version(self._log_dir())
        self.assertGreaterEqual(
            latest, min_version,
            f"Expected log latest version >= {min_version}, got {latest}",
        )

    def _assert_data_files_exist(self) -> list[S3Path]:
        files = self._list_data_files()
        self.assertGreater(
            len(files), 0,
            "Expected at least one .parquet data file outside _delta_log/",
        )
        for f in files:
            try:
                size = f.size
            except Exception as exc:
                self.fail(f"Could not stat data file {f!r}: {exc!r}")
            self.assertGreater(
                size, 0,
                f"Data file {f.name!r} has zero size",
            )
        return files

    def _replay(self) -> "object":
        from yggdrasil.io.buffer.nested.delta.replay import replay_log

        return replay_log(self._log_dir())

    def _table_supports_dvs(self) -> bool:
        replay = self._replay()
        protocol = replay.protocol
        if protocol is None:
            return False
        if "deletionVectors" in (protocol.reader_features or ()):
            return True
        if "deletionVectors" in (protocol.writer_features or ()):
            return True
        return False

    def _assert_dv_descriptor_resolves(
        self,
        io: DeltaIO,
        descriptor: DeletionVectorDescriptor,
    ) -> None:
        if descriptor.is_inline:
            from yggdrasil.io.buffer.nested.delta.deletion_vector import (
                decode_inline_descriptor,
            )
            bitmap = decode_inline_descriptor(descriptor)
            self.assertEqual(
                len(bitmap), descriptor.cardinality,
                "Inline DV bitmap cardinality mismatch with descriptor",
            )
            return

        if descriptor.storage_type == "p":
            bin_path = io._resolve_relative_path(descriptor.path_or_inline)
        elif descriptor.storage_type == "u":
            bin_path = io._resolve_uuid_dv_path(descriptor.path_or_inline)
        else:
            self.fail(f"Unhandled DV storageType {descriptor.storage_type!r}")
            return

        self.assertTrue(bin_path.exists())
        self.assertGreater(bin_path.size, 0)

        offset = descriptor.offset or 0
        header = bin_path.pread(n=5, pos=offset)
        self.assertEqual(len(header), 5)
        self.assertEqual(header[0], 1, "Unexpected DV format byte")

        bitmap = io._load_dv_bitmap_from_file(bin_path, descriptor)
        self.assertEqual(
            len(bitmap), descriptor.cardinality,
            f"Decoded bitmap cardinality {len(bitmap)} != "
            f"descriptor cardinality {descriptor.cardinality}",
        )

    # ==================================================================
    # SQL INSERT → DeltaIO read
    # ==================================================================

    def test_sql_insert_then_delta_read(self):
        """Rows inserted via SQL are readable through DeltaIO."""
        self._truncate()

        version_before = latest_commit_version(self._log_dir())
        self.table.insert([{"a": 1, "s": "hello"}, {"a": 2, "s": "world"}])

        self._assert_log_min_version(version_before + 1)
        self._assert_data_files_exist()

        delta = self._delta_io()
        result = delta.read_arrow_table()

        if "test" in result.column_names:
            test_col = result.column("test")
            a_vals = test_col.flatten()[0].to_pylist()
            s_vals = test_col.flatten()[1].to_pylist()
        else:
            a_vals = result.column("a").to_pylist() if "a" in result.column_names else []
            s_vals = result.column("s").to_pylist() if "s" in result.column_names else []

        self.assertIn(1, a_vals)
        self.assertIn(2, a_vals)
        self.assertIn("hello", s_vals)
        self.assertIn("world", s_vals)
        self.assertEqual(result.num_rows, 2)

        replay = self._replay()
        live_paths = sorted(a.path for a in replay.live_files)
        on_disk_relpaths = sorted(
            "/".join(f.relative_to(self._storage_path()).parts)
            for f in self._list_data_files()
        )
        for p in live_paths:
            self.assertIn(p, on_disk_relpaths)

    def test_sql_insert_schema_matches_delta_schema(self):
        """DeltaIO schema matches the SQL-created table columns."""
        self._truncate()
        self.table.insert([{"a": 1, "s": "x"}])

        self._assert_log_min_version(0)
        self._assert_data_files_exist()

        delta = self._delta_io()
        schema = delta.collect_schema()
        field_names = [f.name for f in schema.fields]

        self.assertIn("test", field_names)

        replay = self._replay()
        self.assertIsNotNone(replay.metadata)
        log_field_names = [f.name for f in replay.metadata.schema.fields]
        self.assertEqual(field_names, log_field_names)

    # ==================================================================
    # DeltaIO write → SQL read  (external-only path)
    # ==================================================================

    def test_delta_append_then_sql_read(self):
        """Rows appended via DeltaIO are visible to SQL SELECT."""
        self._truncate()

        version_before = latest_commit_version(self._log_dir(write=True))

        delta = self._delta_io(write=True)
        struct_type = pa.struct([("a", pa.int32()), ("s", pa.string())])
        tbl = pa.table(
            {"test": pa.array(
                [{"a": 10, "s": "delta_a"}, {"a": 20, "s": "delta_b"}],
                type=struct_type,
            )}
        )
        delta.write_arrow_table(tbl, options=DeltaOptions(mode=Mode.APPEND))

        self._assert_log_min_version(version_before + 1)
        files = self._assert_data_files_exist()
        self.assertGreaterEqual(
            len(files), 1,
            f"Expected at least 1 parquet file after APPEND; got {len(files)}",
        )

        result = self._sql_select_all()
        self.assertGreaterEqual(result.num_rows, 2)

    def test_delta_overwrite_then_sql_read(self):
        """DeltaIO OVERWRITE replaces all data; SQL sees only the new rows."""
        self._truncate()
        self.table.insert([{"a": 1, "s": "old"}])

        files_before = {f.name for f in self._list_data_files()}
        self.assertGreater(len(files_before), 0)

        version_before = latest_commit_version(self._log_dir(write=True))

        delta = self._delta_io(write=True)
        struct_type = pa.struct([("a", pa.int32()), ("s", pa.string())])
        tbl = pa.table(
            {"test": pa.array(
                [{"a": 777, "s": "only_this"}], type=struct_type,
            )}
        )
        delta.write_arrow_table(tbl, options=DeltaOptions(mode=Mode.OVERWRITE))

        self._assert_log_min_version(version_before + 1)
        replay = self._replay()
        live_relpaths = {a.path for a in replay.live_files}
        for old_name in files_before:
            self.assertFalse(
                any(old_name in p for p in live_relpaths),
                f"OVERWRITE didn't remove old file {old_name!r} from live set",
            )

        result = self._sql_select_all()
        self.assertEqual(result.num_rows, 1)

    def test_overwrite_emits_remove_actions(self):
        """OVERWRITE drops old AddFiles from the live set."""
        self._truncate()
        self.table.insert([{"a": 1, "s": "original"}])
        replay_before = self._replay()
        old_paths = {a.path for a in replay_before.live_files}
        self.assertGreater(len(old_paths), 0)

        delta = self._delta_io(write=True)
        struct_type = pa.struct([("a", pa.int32()), ("s", pa.string())])
        tbl = pa.table(
            {"test": pa.array([{"a": 999, "s": "fresh"}], type=struct_type)}
        )
        delta.write_arrow_table(tbl, options=DeltaOptions(mode=Mode.OVERWRITE))

        replay_after = self._replay()
        new_paths = {a.path for a in replay_after.live_files}

        leaked = old_paths & new_paths
        self.assertFalse(
            leaked,
            f"OVERWRITE left old AddFile(s) live: {leaked!r}",
        )

    # ==================================================================
    # Roundtrip: SQL → DeltaIO → SQL
    # ==================================================================

    def test_roundtrip_sql_delta_sql(self):
        """SQL INSERT → DeltaIO read → DeltaIO append → SQL read."""
        self._truncate()

        # 1. SQL insert.
        self.table.insert([{"a": 100, "s": "round"}])
        files_after_sql = {f.name for f in self._assert_data_files_exist()}

        # 2. DeltaIO read confirms it.
        delta = self._delta_io(write=True)
        mid = delta.read_arrow_table()
        self.assertGreaterEqual(mid.num_rows, 1)

        # 3. DeltaIO append.
        version_before_append = latest_commit_version(self._log_dir(write=True))
        struct_type = pa.struct([("a", pa.int32()), ("s", pa.string())])
        extra = pa.table(
            {"test": pa.array([{"a": 200, "s": "trip"}], type=struct_type)}
        )
        delta.write_arrow_table(extra, options=DeltaOptions(mode=Mode.APPEND))

        self._assert_log_min_version(version_before_append + 1)
        files_after_append = {f.name for f in self._list_data_files()}
        self.assertTrue(
            files_after_append > files_after_sql,
            f"DeltaIO APPEND didn't add new parquet files: "
            f"before={files_after_sql!r}, after={files_after_append!r}",
        )

        # 4. SQL sees both.
        final = self._sql_select_all()
        self.assertGreaterEqual(final.num_rows, 2)

    # ==================================================================
    # Deletion vectors
    # ==================================================================

    def test_delta_read_respects_deletion_vectors(self):
        """After SQL DELETE, DeltaIO must filter out deleted rows via DVs."""
        self._truncate()
        self.table.insert([{"a": 1, "s": "keep"}, {"a": 2, "s": "gone"}, {"a": 3, "s": "keep"}])

        self.table.sql.execute(
            f"DELETE FROM {self.table.full_name(safe=True)} WHERE test.a = 2"
        )

        delta = self._delta_io()
        result = delta.read_arrow_table()

        if "test" in result.column_names:
            a_vals = result.column("test").flatten()[0].to_pylist()
        else:
            a_vals = result.column("a").to_pylist()

        self.assertNotIn(2, a_vals)
        self.assertIn(1, a_vals)
        self.assertIn(3, a_vals)

        if self._table_supports_dvs():
            replay = self._replay()
            with_dv = [
                a for a in replay.live_files
                if a.deletion_vector is not None
                and not a.deletion_vector.is_empty
            ]
            self.assertGreater(
                len(with_dv), 0,
                "Table opts into DVs but no live AddFile carries one",
            )

            io = self._delta_io()
            self._assert_dv_descriptor_resolves(io, with_dv[0].deletion_vector)

    def test_dv_files_after_sql_delete(self):
        """When DVs are enabled, SQL DELETE produces .bin files (or inline DVs)."""
        self._truncate()
        self.table.insert([
            {"a": 1, "s": "keep1"},
            {"a": 2, "s": "delete_me"},
            {"a": 3, "s": "keep2"},
            {"a": 4, "s": "delete_me_too"},
        ])

        if not self._table_supports_dvs():
            self.skipTest(
                "Table protocol does not opt into deletionVectors; "
                "DELETE rewrites parquet files instead"
            )

        dv_files_before = {f.name for f in self._list_dv_files()}

        self.table.sql.execute(
            f"DELETE FROM {self.table.full_name(safe=True)} "
            "WHERE test.s LIKE 'delete_me%'"
        )

        dv_files_after = {f.name for f in self._list_dv_files()}
        new_dv_files = dv_files_after - dv_files_before

        if not new_dv_files:
            replay = self._replay()
            inline_dvs = [
                a for a in replay.live_files
                if a.deletion_vector is not None
                and a.deletion_vector.is_inline
                and not a.deletion_vector.is_empty
            ]
            self.assertGreater(
                len(inline_dvs), 0,
                "DELETE on a DV-capable table produced neither .bin nor inline DV",
            )
            io = self._delta_io()
            self._assert_dv_descriptor_resolves(io, inline_dvs[0].deletion_vector)
            return

        io = self._delta_io()
        replay = self._replay()
        external_dvs = [
            a for a in replay.live_files
            if a.deletion_vector is not None
            and not a.deletion_vector.is_inline
            and not a.deletion_vector.is_empty
        ]
        self.assertGreater(
            len(external_dvs), 0,
            f"Found new .bin files {new_dv_files!r} but no live AddFile "
            "references one",
        )
        self._assert_dv_descriptor_resolves(io, external_dvs[0].deletion_vector)

    def test_no_dv_files_when_dvs_disabled(self):
        """If DVs are disabled, no .bin files appear after DELETE."""
        self._truncate()
        self.table.insert([{"a": 1, "s": "a"}, {"a": 2, "s": "b"}])

        if self._table_supports_dvs():
            self.skipTest("Table opts into DVs; this test only applies otherwise")

        dv_files_before = {f.name for f in self._list_dv_files()}
        self.table.sql.execute(
            f"DELETE FROM {self.table.full_name(safe=True)} WHERE test.a = 1"
        )
        dv_files_after = {f.name for f in self._list_dv_files()}

        self.assertEqual(
            dv_files_before, dv_files_after,
            "Found new .bin files on a DV-disabled table",
        )

    # ==================================================================
    # Edge cases / read-only smoke
    # ==================================================================

    def test_delta_read_empty_after_truncate(self):
        """DeltaIO handles an empty table gracefully."""
        self._truncate()

        delta = self._delta_io()
        result = delta.read_arrow_table()
        self.assertEqual(result.num_rows, 0)

        log_dir = self._log_dir()
        self.assertTrue(log_dir.exists())
        self._assert_log_min_version(0)

    def test_delta_iter_children(self):
        """DeltaIO.iter_children yields one child IO per live AddFile."""
        self._truncate()
        self.table.insert([{"a": 1, "s": "frag"}])

        delta = self._delta_io()
        children = list(delta.iter_children())
        self.assertGreater(len(children), 0)

        on_disk = {f.url.path for f in self._list_data_files()}

        for child in children:
            self.assertTrue(child.path)
            self.assertIs(child.parent, delta)
            self.assertIn(
                child.path.url.path, on_disk,
                f"Child path {child.path!r} not in on-disk set",
            )

    def test_storage_location(self):
        """Smoke test: storage_location returns a valid path and DeltaIO opens it."""
        storage_location = self.table.storage_location()
        delta = DeltaIO.from_path(storage_location)
        schema = delta.collect_schema()
        self.assertIsNotNone(schema)

    # ==================================================================
    # Dedicated S3-content tests
    # ==================================================================

    def test_storage_location_is_s3path_with_service(self):
        """storage_location returns an S3Path with a usable service object."""
        path = self._storage_path()

        self.assertEqual(path.url.scheme, "s3")
        self.assertTrue(path.bucket)
        self.assertTrue(path.key)

        service = path.service
        self.assertIsNotNone(service)
        boto = service.boto_client
        self.assertIsNotNone(boto)

        self.assertTrue(
            path.exists(),
            f"Storage path {path!r} doesn't exist via S3Path.exists()",
        )

    def test_storage_location_creds_can_list(self):
        """The vended credentials must permit ``ls`` on the storage location."""
        path = self._storage_path()
        children = list(path.ls(recursive=False, allow_not_found=True))
        names = [c.name for c in children]
        self.assertIn(
            "_delta_log", names,
            f"_delta_log not visible under {path!r}; got {names!r}",
        )

    def test_storage_location_under_external_prefix(self):
        """The vended storage path lives under :data:`EXTERNAL_LOCATION`.

        Sanity-check that the table actually got placed in the
        external location and not in some Databricks-managed prefix
        (would mean the LOCATION clause silently failed).
        """
        external_url = URL.from_str(EXTERNAL_LOCATION)
        path = self._storage_path()

        self.assertEqual(
            path.bucket, external_url.host,
            f"Storage path bucket {path.bucket!r} doesn't match "
            f"EXTERNAL_LOCATION bucket {external_url.host!r}; the "
            "table may have been created managed instead of external",
        )

        external_prefix = external_url.path.lstrip("/").rstrip("/")
        self.assertTrue(
            path.key.startswith(external_prefix),
            f"Storage path key {path.key!r} does not start with "
            f"external prefix {external_prefix!r}",
        )

    def test_log_directory_structure(self):
        """The log directory has well-formed commit JSON files."""
        self._truncate()
        self.table.insert([{"a": 1, "s": "log_test"}])

        log_dir = self._log_dir()
        from yggdrasil.io.buffer.nested.delta.constants import COMMIT_FILE_RE

        commit_files: list[tuple[int, S3Path]] = []
        for child in log_dir.ls(recursive=False, allow_not_found=False):
            m = COMMIT_FILE_RE.match(child.name)
            if m is None:
                continue
            commit_files.append((int(m.group(1)), child))

        self.assertGreater(len(commit_files), 0)

        for version, child in commit_files:
            size = child.size
            self.assertGreater(
                size, 0,
                f"Commit file {child.name!r} (v{version}) has zero size",
            )

        versions = sorted(v for v, _ in commit_files)
        last_cp = read_last_checkpoint(log_dir)
        if last_cp is None:
            self.assertEqual(versions[0], 0)
        else:
            cp_version = int(last_cp["version"])
            self.assertGreaterEqual(versions[0], 0)
            self.assertIn(
                cp_version + 1, versions,
                f"Commit {cp_version + 1} (post-checkpoint) missing",
            )

        for prev, curr in zip(versions, versions[1:]):
            self.assertEqual(curr, prev + 1, f"Gap in log: {prev} → {curr}")

    def test_data_file_paths_are_valid_s3_paths(self):
        """Every live AddFile resolves to a real S3 object via S3Path."""
        self._truncate()
        self.table.insert([{"a": 1, "s": "p1"}, {"a": 2, "s": "p2"}])

        replay = self._replay()
        self.assertGreater(len(replay.live_files), 0)

        root = self._storage_path()
        for add in replay.live_files:
            child = root.joinpath(*add.path.split("/"))
            self.assertTrue(
                child.exists(),
                f"AddFile {add.path!r} not on disk at {child!r}",
            )
            actual_size = child.size
            self.assertEqual(
                actual_size, add.size,
                f"AddFile {add.path!r} size mismatch: "
                f"log={add.size}, S3={actual_size}",
            )