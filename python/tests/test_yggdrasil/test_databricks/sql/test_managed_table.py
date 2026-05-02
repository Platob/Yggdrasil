"""DeltaIO integration tests against a Unity Catalog **managed** table.

Managed tables are owned by Databricks: data files live in
Databricks-managed storage and only the SQL warehouse is supported
as a writer. Writing directly to the Delta log from outside the
warehouse can corrupt the table.

This test file therefore exercises only the **read** path:

- Writes happen via the SQL warehouse (``self.table.insert(...)``,
  ``self.table.sql.execute("DELETE ...")``).
- Reads go through our :class:`DeltaIO` — log replay, AddFile
  enumeration, deletion-vector decoding, S3-content checks.

The companion :mod:`test_external` exercises the full read+write
surface against a customer-owned external table where DeltaIO
writes are safe.

This file is fully self-contained — no inheritance from
``test_external``. Helpers are duplicated between the two files
on purpose: each file should read top-to-bottom without chasing
imports across modules.
"""

import pyarrow as pa

from yggdrasil.aws.fs.path import S3Path
from yggdrasil.io.buffer.nested.delta import DeltaIO
from yggdrasil.io.buffer.nested.delta.deletion_vector import (
    DeletionVectorDescriptor,
)
from yggdrasil.io.buffer.nested.delta.replay import (
    latest_commit_version,
    read_last_checkpoint,
)
from yggdrasil.io.fs import Path
from .._base import DatabricksCase


# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------

#: struct<a:int, s:string> — same schema used by the external test file
#: so the read tests are exactly parallel between the two cases.
TABLE_SCHEMA: str = "'test': struct<a: int, s: string>"

#: Fully-qualified name for the managed test table.
MANAGED_TABLE_NAME: str = "trading_tgp_dev.src_monteleq.test_table"


class ManagedTableIntegrationCase(DatabricksCase):
    """Read-only DeltaIO tests against a UC-managed Delta table.

    SQL warehouse writes; DeltaIO reads. Validates that:

    - rows inserted/deleted via SQL are visible through our log
      replay;
    - the schema we recover via :meth:`DeltaIO.collect_schema`
      matches what UC declared;
    - the parquet files and ``_delta_log`` JSON commits are
      reachable through our :class:`S3Path`;
    - deletion vectors emitted by Databricks decode through our
      DV reader.

    No test in this file calls ``DeltaIO.write_*`` —
    cross-implementation writes against a managed table are
    unsupported.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.table = cls.client.sql.table(MANAGED_TABLE_NAME).create(TABLE_SCHEMA)
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

    def _delta_io(self) -> DeltaIO:
        """Get a DeltaIO pointed at the table's S3 storage. Read-only."""
        from databricks.sdk.service.catalog import TableOperation

        path = self.table.storage_location(operation=TableOperation.READ)
        return DeltaIO.from_path(path)

    def _truncate(self):
        """Wipe rows via SQL. The only supported write path on a managed table."""
        self.table.sql.execute(
            f"DELETE FROM {self.table.full_name(safe=True)} WHERE true"
        )

    def _sql_select_all(self) -> pa.Table:
        return self.table.to_arrow_dataset()

    # --- S3-content helpers --------------------------------------------

    def _storage_path(self) -> S3Path:
        """The yggdrasil S3Path for the table's storage location."""
        from databricks.sdk.service.catalog import TableOperation

        path = self.table.storage_location(operation=TableOperation.READ)
        self.assertIsInstance(
            path, S3Path,
            f"storage_location returned {type(path).__name__}, expected S3Path",
        )
        return path

    def _log_dir(self) -> S3Path:
        """The ``_delta_log/`` subdirectory under the storage path."""
        return self._storage_path() / "_delta_log"

    def _list_keys(self, root: Path, *, recursive: bool = False) -> list[str]:
        """List children of *root* and return their basenames, sorted."""
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
        log_dir = self._log_dir()
        latest = latest_commit_version(log_dir)
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
        """Replay the table's log through the public replay function."""
        from yggdrasil.io.buffer.nested.delta.replay import replay_log

        return replay_log(self._log_dir())

    def _table_supports_dvs(self) -> bool:
        """Does this table's protocol opt into deletion vectors?"""
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
        """Resolve a DV descriptor end-to-end and verify the bitmap."""
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

        self.assertTrue(
            bin_path.exists(),
            f"DV descriptor points at {bin_path!r} which is missing",
        )
        size = bin_path.size
        self.assertGreater(
            size, 0,
            f"DV file {bin_path!r} has zero size",
        )

        # Verify framing: 1-byte format version + 4-byte LE size at the offset.
        offset = descriptor.offset or 0
        header = bin_path.pread(n=5, pos=offset)
        self.assertEqual(
            len(header), 5,
            f"Short pread on DV header at {bin_path!r}@{offset}",
        )
        format_byte = header[0]
        self.assertEqual(
            format_byte, 1,
            f"Unexpected DV format byte {format_byte!r}; expected 1",
        )

        # Full decode through the DeltaIO read path.
        bitmap = io._load_dv_bitmap_from_file(bin_path, descriptor)
        self.assertEqual(
            len(bitmap), descriptor.cardinality,
            f"Decoded bitmap cardinality {len(bitmap)} != descriptor "
            f"cardinality {descriptor.cardinality}",
        )

    # ==================================================================
    # SQL INSERT → DeltaIO read
    # ==================================================================

    def test_sql_insert_then_delta_read(self):
        """Rows inserted via SQL warehouse are readable through DeltaIO."""
        self._truncate()

        version_before = latest_commit_version(self._log_dir())
        self.table.insert([{"a": 1, "s": "hello"}, {"a": 2, "s": "world"}])

        # The insert advanced the log AND wrote at least one parquet file.
        self._assert_log_min_version(version_before + 1)
        self._assert_data_files_exist()

        delta = self._delta_io()
        result = delta.read_arrow_table()

        # struct<a, s> may surface as a single struct column or flattened.
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

        # Cross-check: every live AddFile.path must correspond to an on-disk file.
        replay = self._replay()
        live_paths = sorted(a.path for a in replay.live_files)
        on_disk_relpaths = sorted(
            "/".join(f.relative_to(self._storage_path()).parts)
            for f in self._list_data_files()
        )
        for p in live_paths:
            self.assertIn(
                p, on_disk_relpaths,
                f"Live AddFile path {p!r} not found on disk",
            )

    def test_sql_insert_schema_matches_delta_schema(self):
        """DeltaIO schema from the log matches the SQL-created table columns."""
        self._truncate()
        self.table.insert([{"a": 1, "s": "x"}])

        self._assert_log_min_version(0)
        self._assert_data_files_exist()

        delta = self._delta_io()
        schema = delta.collect_schema()
        field_names = [f.name for f in schema.fields]

        self.assertIn("test", field_names)

        # The Delta log's metadata schema must agree with what we just got.
        replay = self._replay()
        self.assertIsNotNone(replay.metadata)
        log_field_names = [f.name for f in replay.metadata.schema.fields]
        self.assertEqual(
            field_names, log_field_names,
            "DeltaIO.collect_schema() disagrees with replay metadata schema",
        )

    # ==================================================================
    # Deletion vectors
    # ==================================================================

    def test_delta_read_respects_deletion_vectors(self):
        """After SQL DELETE, DeltaIO must filter out deleted rows via DVs."""
        self._truncate()
        self.table.insert([{"a": 1, "s": "keep"}, {"a": 2, "s": "gone"}, {"a": 3, "s": "keep"}])

        # SQL DELETE — Databricks may emit a DV (depends on table's protocol).
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

        # If the table's protocol opts into DVs, verify the descriptor decodes.
        if self._table_supports_dvs():
            replay = self._replay()
            with_dv = [
                a for a in replay.live_files
                if a.deletion_vector is not None
                and not a.deletion_vector.is_empty
            ]
            self.assertGreater(
                len(with_dv), 0,
                "Table opts into DVs but no live AddFile carries one after DELETE",
            )

            io = self._delta_io()
            self._assert_dv_descriptor_resolves(io, with_dv[0].deletion_vector)

    def test_dv_files_after_sql_delete(self):
        """When the table opts into DVs, SQL DELETE produces .bin files (or inline DVs).

        Skipped on tables without DV support — Databricks' default for
        Unity Catalog tables varies by metastore version.
        """
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
                "DELETE rewrites parquet files instead of emitting DVs"
            )

        dv_files_before = {f.name for f in self._list_dv_files()}

        self.table.sql.execute(
            f"DELETE FROM {self.table.full_name(safe=True)} "
            "WHERE test.s LIKE 'delete_me%'"
        )

        dv_files_after = {f.name for f in self._list_dv_files()}
        new_dv_files = dv_files_after - dv_files_before

        # Inline DVs (no .bin file) for very small tables — fall back to log.
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
                "DELETE on a DV-capable table produced neither a .bin file "
                "nor an inline DV",
            )
            io = self._delta_io()
            self._assert_dv_descriptor_resolves(io, inline_dvs[0].deletion_vector)
            return

        # External DV — at least one new .bin file appeared.
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
            "references one. The DV may be from a now-removed file.",
        )
        self._assert_dv_descriptor_resolves(io, external_dvs[0].deletion_vector)

    def test_no_dv_files_when_dvs_disabled(self):
        """If the table doesn't opt into DVs, no .bin files appear after DELETE.

        Inverse of the previous test — proves DELETE went down the
        file-rewrite path. Skipped on DV-capable tables.
        """
        self._truncate()
        self.table.insert([{"a": 1, "s": "a"}, {"a": 2, "s": "b"}])

        if self._table_supports_dvs():
            self.skipTest(
                "Table opts into deletionVectors; this test only applies "
                "to DV-disabled tables (DELETE = file rewrite)"
            )

        dv_files_before = {f.name for f in self._list_dv_files()}
        self.table.sql.execute(
            f"DELETE FROM {self.table.full_name(safe=True)} WHERE test.a = 1"
        )
        dv_files_after = {f.name for f in self._list_dv_files()}

        self.assertEqual(
            dv_files_before, dv_files_after,
            "Found new .bin files on a DV-disabled table; "
            f"before={dv_files_before!r}, after={dv_files_after!r}",
        )

    # ==================================================================
    # Edge cases / read-only smoke
    # ==================================================================

    def test_delta_read_empty_after_truncate(self):
        """DeltaIO handles an empty table (all rows deleted) gracefully."""
        self._truncate()

        delta = self._delta_io()
        result = delta.read_arrow_table()
        self.assertEqual(result.num_rows, 0)

        # The log still exists (truncate is a write, not a wipe).
        log_dir = self._log_dir()
        self.assertTrue(
            log_dir.exists(),
            "Truncate dropped the _delta_log directory; should never happen",
        )
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
                f"Child path {child.path!r} not in on-disk set "
                f"{on_disk!r}",
            )

    def test_storage_location(self):
        """Smoke test: storage_location returns a valid path and DeltaIO opens it."""
        storage_location = self.table.storage_location()
        delta = DeltaIO.from_path(storage_location)
        # Should not raise — the log directory exists after table creation.
        schema = delta.collect_schema()
        self.assertIsNotNone(schema)

    # ==================================================================
    # Dedicated S3-content tests
    # ==================================================================

    def test_storage_location_is_s3path_with_service(self):
        """storage_location returns an S3Path with a usable service object."""
        path = self._storage_path()

        self.assertEqual(path.url.scheme, "s3")
        self.assertTrue(path.bucket, f"Empty bucket on {path!r}")
        self.assertTrue(path.key, f"Empty key on {path!r}")

        service = path.service
        self.assertIsNotNone(service)
        boto = service.boto_client
        self.assertIsNotNone(boto)

        # Vended creds work — the path exists.
        self.assertTrue(
            path.exists(),
            f"Storage path {path!r} doesn't exist via S3Path.exists() — "
            "the vended credentials can't see what they vended",
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

    def test_log_directory_structure(self):
        """The log directory has well-formed commit JSON files.

        Asserts: filenames match ``\\d{20}\\.json``, versions are
        contiguous from 0 to latest, every file is non-empty.
        """
        # Make sure we have at least one commit beyond 0 to test against.
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

        self.assertGreater(
            len(commit_files), 0,
            f"No commit files matching NN…NN.json in {log_dir!r}",
        )

        for version, child in commit_files:
            size = child.size
            self.assertGreater(
                size, 0,
                f"Commit file {child.name!r} (v{version}) has zero size",
            )

        versions = sorted(v for v, _ in commit_files)
        last_cp = read_last_checkpoint(log_dir)
        if last_cp is None:
            self.assertEqual(
                versions[0], 0,
                f"Without a checkpoint, expected log to start at 0; got {versions[0]}",
            )
        else:
            cp_version = int(last_cp["version"])
            self.assertGreaterEqual(versions[0], 0)
            self.assertIn(
                cp_version + 1, versions,
                f"Commit {cp_version + 1} (post-checkpoint) missing from log",
            )

        for prev, curr in zip(versions, versions[1:]):
            self.assertEqual(
                curr, prev + 1,
                f"Gap in log: {prev} → {curr}",
            )

    def test_data_file_paths_are_valid_s3_paths(self):
        """Every live AddFile resolves to a real S3 object via S3Path.

        Replay → for each AddFile, build the S3Path it points at →
        confirm exists() and content_length matches ``AddFile.size``.
        """
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
                f"AddFile {add.path!r} size mismatch: log says {add.size}, "
                f"S3 says {actual_size}",
            )