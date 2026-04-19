"""Unit tests for :class:`yggdrasil.databricks.sql.staging.StagingPath`."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from yggdrasil.databricks.sql.staging import StagingPath


def _fake_volume_path(str_path: str = "/Volumes/c/s/v/f.parquet"):
    """Minimal ``VolumePath`` stand-in — satisfies isinstance checks via monkey-patching.

    The real ``VolumePath`` needs a Databricks client; for unit tests we only
    care about the isinstance guard and the ``remove`` call that
    :meth:`StagingPath.cleanup` makes.
    """
    from yggdrasil.databricks.fs.path import VolumePath

    vp = MagicMock(spec=VolumePath)
    vp.__class__ = VolumePath
    vp.__str__ = lambda self=vp: str_path  # type: ignore[assignment]
    vp.sql_volume_or_table_parts.return_value = ("c", "s", "v", ("f.parquet",))
    return vp


class TestStagingPathOwnedFlag(unittest.TestCase):
    def test_default_construction_is_not_owned(self):
        path = _fake_volume_path()
        staging = StagingPath(
            path=path,
            catalog_name="c",
            schema_name="s",
            table_name="v",
            start_ts=0,
            end_ts=1,
            token="abc",
        )
        self.assertFalse(staging.owned)

    def test_cleanup_skips_remove_when_not_owned(self):
        path = _fake_volume_path()
        staging = StagingPath(
            path=path,
            catalog_name="c", schema_name="s", table_name="v",
            start_ts=0, end_ts=1, token="abc",
        )
        staging.cleanup()
        path.remove.assert_not_called()

    def test_cleanup_removes_when_owned(self):
        path = _fake_volume_path()
        staging = StagingPath(
            path=path,
            catalog_name="c", schema_name="s", table_name="v",
            start_ts=0, end_ts=1, token="abc",
            owned=True,
        )
        staging.cleanup()
        path.remove.assert_called_once_with(recursive=True, allow_not_found=True)


class TestStagingPathFromVolume(unittest.TestCase):
    def test_from_volume_wraps_existing_path_not_owned(self):
        path = _fake_volume_path("/Volumes/c/s/v/my_file.parquet")
        staging = StagingPath.from_volume(path)
        self.assertIs(staging.path, path)
        self.assertFalse(staging.owned)
        self.assertEqual(staging.catalog_name, "c")
        self.assertEqual(staging.schema_name, "s")
        self.assertEqual(staging.table_name, "v")

    def test_from_volume_rejects_non_volume(self):
        with self.assertRaises(TypeError):
            StagingPath.from_volume(object())  # type: ignore[arg-type]

    def test_from_volume_can_opt_into_ownership(self):
        path = _fake_volume_path()
        staging = StagingPath.from_volume(path, owned=True)
        self.assertTrue(staging.owned)


class TestStageExternalTablesOwnedHandoff(unittest.TestCase):
    """``_stage_external_tables`` respects the ``owned`` flag on StagingPaths."""

    def _make_engine(self):
        from yggdrasil.databricks.sql.engine import SQLEngine
        # Bind the real method onto a lightweight object; we don't need a
        # live Databricks client because the test never stages raw data.
        engine = object.__new__(SQLEngine)
        object.__setattr__(engine, "catalog_name", None)
        object.__setattr__(engine, "schema_name", None)
        return engine

    def test_owned_staging_is_returned_for_cleanup(self):
        engine = self._make_engine()
        path = _fake_volume_path()
        staging = StagingPath.from_volume(path, owned=True)

        substitutions, owned = engine._stage_external_tables({"src": staging})
        self.assertIn("src", substitutions)
        self.assertEqual(owned, [staging])

    def test_non_owned_staging_is_substituted_only(self):
        engine = self._make_engine()
        path = _fake_volume_path()
        staging = StagingPath.from_volume(path, owned=False)

        substitutions, owned = engine._stage_external_tables({"src": staging})
        self.assertIn("src", substitutions)
        self.assertEqual(owned, [])


class TestStagingWriteTableReadColumns(unittest.TestCase):
    """``read_columns`` populates ``last_read_frame`` from the buffer."""

    def _staging_with_recorder(self, written: list[bytes]):
        path = _fake_volume_path()
        # ``write_bytes`` is what staging calls after serializing to the
        # in-memory Parquet buffer — capture the payload, ignore the volume.
        path.write_bytes = lambda view: written.append(bytes(view))
        path.parent = MagicMock()
        return StagingPath(
            path=path,
            catalog_name="c", schema_name="s", table_name="v",
            start_ts=0, end_ts=1, token="abc",
        )

    def test_read_columns_projection_populates_last_read_frame(self):
        import pyarrow as pa

        written: list[bytes] = []
        staging = self._staging_with_recorder(written)

        data = pa.table(
            {
                "dt": ["2026-04-01", "2026-04-02", "2026-04-01"],
                "id": [1, 2, 3],
                "payload": ["a", "b", "c"],
            }
        )

        out = staging.write_table(data, read_columns=["dt"])

        self.assertIs(out, staging)
        self.assertEqual(len(written), 1)
        self.assertGreater(len(written[0]), 0)

        frame = staging.last_read_frame
        self.assertIsNotNone(frame)
        # Only the requested column should have been projected back.
        self.assertEqual(list(frame.columns), ["dt"])
        self.assertEqual(
            frame.get_column("dt").to_list(),
            ["2026-04-01", "2026-04-02", "2026-04-01"],
        )

    def test_without_read_columns_last_read_frame_is_none(self):
        import pyarrow as pa

        written: list[bytes] = []
        staging = self._staging_with_recorder(written)

        staging.write_table(pa.table({"dt": ["2026-04-01"], "id": [1]}))

        self.assertIsNone(staging.last_read_frame)

    def test_write_clears_previous_read_frame(self):
        import pyarrow as pa

        written: list[bytes] = []
        staging = self._staging_with_recorder(written)

        staging.write_table(
            pa.table({"dt": ["2026-04-01"]}),
            read_columns=["dt"],
        )
        self.assertIsNotNone(staging.last_read_frame)

        staging.write_table(pa.table({"dt": ["2026-04-02"]}))
        # No ``read_columns`` on the second write → frame cleared.
        self.assertIsNone(staging.last_read_frame)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
