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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
