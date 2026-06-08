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
from yggdrasil.databricks.volume import Volume


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
    ws = workspace or MagicMock()
    service.client.workspace_client.return_value = ws
    # ``Table.staging_volume`` builds a ``Volume`` from
    # ``service.volumes`` whose client must reach the same workspace
    # the table's client does — the minted VolumePath calls
    # ``files.upload`` through *Volume.client.workspace_client*.
    service.volumes.client.workspace_client.return_value = ws
    # ``Table.staging_volume`` runs ``client.safe_tag_value`` on the
    # table name to derive the ``<table>`` volume — mirror the
    # production behavior (strip/lower) so test assertions can match
    # the rendered volume name.
    service.client.safe_tag_value.side_effect = lambda v, repl="_": str(v).replace(
        "/", repl,
    )
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
        # ``staging_volume`` is ``<table>`` under the table's
        # catalog / schema; staging files land in ``tmp/``.
        assert full.startswith("/Volumes/cat/sch/tbl/tmp/")
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
        assert "/Volumes/cat2/sch2/extra/tmp/" in full

    def test_unique_per_call(self) -> None:
        tbl = _table()
        a = tbl.insert_volume_path().full_path()
        b = tbl.insert_volume_path().full_path()
        # Filename carries epoch-ms + 8 hex chars of randomness so two
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
# staging_volume — the per-table staging volume, derived + get-or-created
# ---------------------------------------------------------------------------


class TestStagingVolume:

    def test_none_when_table_has_no_infos(self) -> None:
        """The staging volume is isolated *under the table's own identity* — with
        no table info to derive it from, the property is ``None``."""
        tbl = _table("cat", "sch", "tbl")
        with patch.object(Table, "read_infos", return_value=None):
            assert tbl.staging_volume is None

    def test_builds_fast_external_candidate_sibling_of_table_data(self) -> None:
        """``staging_volume`` builds a fast ``_ygg_staging`` candidate location —
        a sibling of the table's own data (parent dir + table id) — and hands it
        to ``Volume.create`` (which owns the external-location validation), then
        caches the handle. No external-location lookup happens on the table."""
        from types import SimpleNamespace

        tbl = _table("cat", "sch", "tbl")
        infos = SimpleNamespace(
            storage_location="s3://bkt/apps/team/tbl", table_id="tid-1",
        )
        with patch.object(Table, "read_infos", return_value=infos), \
                patch.object(Volume, "get_or_create", return_value=None) as goc:
            vol = tbl.staging_volume
            again = tbl.staging_volume

        assert isinstance(vol, Volume)
        assert vol.volume_name == "tbl"           # named after the table
        assert again is vol                        # cached on the instance
        # The table does not consult external locations — that's Volume.create.
        tbl.service.client.external_locations.find_url.assert_not_called()
        loc = goc.call_args.kwargs["storage_location"]
        assert loc == "s3://bkt/apps/team/_ygg_staging/volumes/tid-1"


class TestTableStagingInvalidation:
    """Invalidation (what ``delete`` runs) drops the table's storage-derived
    caches — the staging-volume handle and the cached infos."""

    def test_invalidate_clears_staging_and_infos(self) -> None:
        tbl = _table("cat", "sch", "tbl")
        tbl._staging_volume = MagicMock()
        tbl.invalidate_singleton()
        assert tbl._staging_volume is None
        assert tbl._infos is None


# ---------------------------------------------------------------------------
# Mocking — arrow_insert routes through insert_volume_path
# ---------------------------------------------------------------------------


class TestInsertVolumePathIsMockable:

    def test_instance_override_swaps_staging_target(self) -> None:
        """Replacing the bound method on a single instance lets a
        test pin the staging path (and its workspace) without
        monkey-patching the underlying staging-volume plumbing."""
        tbl = _table()
        custom = VolumePath(
            "/Volumes/test/test/tbl/.sql/tmp/part-fixed.parquet",
            client=MagicMock(),
        )
        tbl.insert_volume_path = lambda *a, **kw: custom  # type: ignore[assignment]
        assert tbl.insert_volume_path() is custom
