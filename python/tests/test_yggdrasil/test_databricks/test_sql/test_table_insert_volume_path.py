"""Tests for :meth:`Table.insert_volume_path` — the staging-path
generator that the warehouse :meth:`Table.arrow_insert` path uses.

Lifting the path generation out of ``arrow_insert`` makes the
"local" half of the insert (everything that runs before the SQL
hits the warehouse) testable without a live workspace: we can
either drive ``insert_volume_path`` directly, or replace it on a
single instance to verify the full insert plumbs the staging
``VolumePath`` through unchanged.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from yggdrasil.databricks.fs import VolumePath
from yggdrasil.databricks.table.table import Table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table(
    catalog_name: str = "cat",
    schema_name: str = "sch",
    table_name: str = "tbl",
    *,
    workspace: object | None = None,
) -> Table:
    """Build a :class:`Table` with a mocked service so ``self.client``
    resolves without touching a real Databricks workspace."""
    service = MagicMock()
    service.client.workspace_client.return_value = workspace or MagicMock()
    return Table(
        service=service,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name=table_name,
    )


# ---------------------------------------------------------------------------
# insert_volume_path — path generation
# ---------------------------------------------------------------------------


class TestInsertVolumePath:

    def test_default_layout_uses_self_identity(self) -> None:
        tbl = _table("cat", "sch", "tbl")
        path = tbl.insert_volume_path()
        assert isinstance(path, VolumePath)
        full = path.full_path()
        assert full.startswith("/Volumes/cat/sch/tmp_tbl/.sql")
        assert full.endswith(".parquet")
        # Default keeps the staged Parquet temporary so the holder
        # cleans up after itself.
        assert path.temporary is True

    def test_temporary_flag_propagates(self) -> None:
        tbl = _table()
        assert tbl.insert_volume_path(temporary=False).temporary is False
        assert tbl.insert_volume_path(temporary=True).temporary is True

    def test_target_override_routes_under_other_table(self) -> None:
        """``target`` swaps the catalog/schema/name segments so a caller
        can stage rows under a sibling table's volume."""
        tbl = _table("cat", "sch", "primary")
        other = _table("cat2", "sch2", "extra")
        path = tbl.insert_volume_path(other)
        full = path.full_path()
        assert "/Volumes/cat2/sch2/tmp_extra/.sql" in full

    def test_unique_per_call(self) -> None:
        tbl = _table()
        a = tbl.insert_volume_path().full_path()
        b = tbl.insert_volume_path().full_path()
        # Filename carries epoch-ms + 8 bytes of randomness so two
        # successive calls never collide.
        assert a != b

    def test_workspace_resolved_from_client(self) -> None:
        """The staging path borrows the workspace client from
        ``self.client.workspace_client()`` — the same surface that
        the rest of the insert uses for Files API calls."""
        ws = MagicMock(name="workspace")
        tbl = _table(workspace=ws)
        assert tbl.insert_volume_path().workspace_client is ws


# ---------------------------------------------------------------------------
# Mocking — arrow_insert routes through insert_volume_path
# ---------------------------------------------------------------------------


class TestInsertVolumePathIsMockable:

    def test_delegates_to_volumepath_staging_path(self) -> None:
        """``insert_volume_path`` is a thin wrapper — patching
        :meth:`VolumePath.staging_path` should be enough to
        observe (or replace) the path it produces."""
        tbl = _table("cat", "sch", "tbl")
        sentinel = MagicMock(spec=VolumePath)
        with patch.object(
            VolumePath, "staging_path", return_value=sentinel,
        ) as staging:
            out = tbl.insert_volume_path(temporary=False)
        assert out is sentinel
        kwargs = staging.call_args.kwargs
        assert kwargs["catalog_name"] == "cat"
        assert kwargs["schema_name"] == "sch"
        assert kwargs["resource_name"] == "tbl"
        assert kwargs["temporary"] is False
        # ``self.client`` is forwarded so :meth:`VolumePath.staging_path`
        # can resolve the workspace lazily through the aggregator —
        # same shape it accepts in production.
        assert kwargs["client"] is tbl.client

    def test_instance_override_swaps_staging_target(self) -> None:
        """Replacing the bound method on a single instance lets a
        test pin the staging path (and its workspace) without
        monkey-patching the :class:`VolumePath` class."""
        tbl = _table()
        custom = VolumePath(
            "/Volumes/test/test/tmp/.sql/test/test/tbl/part-fixed.parquet",
            client=MagicMock(),
        )
        tbl.insert_volume_path = lambda *a, **kw: custom  # type: ignore[assignment]
        assert tbl.insert_volume_path() is custom
