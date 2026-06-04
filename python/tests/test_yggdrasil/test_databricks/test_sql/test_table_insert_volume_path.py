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

from unittest.mock import MagicMock, PropertyMock, patch

from databricks.sdk.service.catalog import TableType

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
        # catalog / schema; staging files land in ``.sql/tmp/``.
        assert full.startswith("/Volumes/cat/sch/tbl/.sql/tmp/")
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
        assert "/Volumes/cat2/sch2/extra/.sql/tmp/" in full

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
# staging_volume — external, rooted at the schema storage location
# ---------------------------------------------------------------------------


class TestStagingVolumeExternal:

    def test_property_is_a_cheap_handle(self) -> None:
        """Reading ``staging_volume`` never resolves infos or creates a
        volume — it just mints the handle (creation is deferred to
        :meth:`Table.ensure_staging_volume`)."""
        tbl = _table("cat", "sch", "tbl")
        with patch.object(Volume, "get_or_create") as goc:
            vol = tbl.staging_volume
            assert tbl.staging_volume is vol  # singleton
        assert isinstance(vol, Volume)
        goc.assert_not_called()

    def test_external_table_derives_root_and_stamps_table_property(self) -> None:
        """First staging on an external table derives ``<schema-staging>/<hash>``,
        records it on the table's ``ygg.staging_root`` TBLPROPERTY, and creates
        the external volume there — never the client default storage location."""
        import hashlib
        from types import SimpleNamespace

        from yggdrasil.databricks.schema.schema import UCSchema

        tbl = _table("cat", "sch", "tbl")
        # Table has no recorded staging root yet → derive + stamp.
        with patch.object(
            Table, "read_infos",
            return_value=SimpleNamespace(table_type=TableType.EXTERNAL),
        ), patch.object(
            Table, "infos", new_callable=PropertyMock,
        ) as infos, patch.object(
            UCSchema, "staging_location", return_value="s3://bkt/meta/uc/tables",
        ), patch.object(
            Table, "properties", new_callable=PropertyMock,
        ) as props, patch.object(
            Volume, "get_or_create",
        ) as goc:
            infos.return_value.properties = {}        # nothing recorded yet
            stored: dict = {}
            props.return_value = stored
            vol = tbl.ensure_staging_volume()

        key = hashlib.blake2b(b"cat.sch.tbl", digest_size=16).hexdigest()
        expected = f"s3://bkt/meta/uc/tables/{key}"
        # Stamped on the table's TBLPROPERTIES ...
        assert stored["ygg.staging_root"] == expected
        # ... and the external volume was created there.
        assert isinstance(vol, Volume)
        goc.assert_called_once()
        kw = goc.call_args.kwargs
        assert kw["volume_type"] == "EXTERNAL"
        assert kw["storage_location"] == expected

    def test_external_table_reuses_recorded_staging_root(self) -> None:
        # A recorded ``ygg.staging_root`` is used directly — no derivation.
        from types import SimpleNamespace

        from yggdrasil.databricks.schema.schema import UCSchema

        tbl = _table("cat", "sch", "tbl")
        with patch.object(
            Table, "read_infos",
            return_value=SimpleNamespace(table_type=TableType.EXTERNAL),
        ), patch.object(
            Table, "infos", new_callable=PropertyMock,
        ) as infos, patch.object(
            UCSchema, "staging_location",
        ) as staging, patch.object(
            Volume, "get_or_create",
        ) as goc:
            infos.return_value.properties = {"ygg.staging_root": "s3://pinned/loc"}
            tbl.ensure_staging_volume()

        staging.assert_not_called()                 # no re-derivation
        assert goc.call_args.kwargs["storage_location"] == "s3://pinned/loc"

    def test_external_table_without_staging_root_stays_managed(self) -> None:
        # No resolvable schema staging path → fall back to the managed default.
        from types import SimpleNamespace

        from yggdrasil.databricks.schema.schema import UCSchema

        tbl = _table("cat", "sch", "tbl")
        with patch.object(
            Table, "read_infos",
            return_value=SimpleNamespace(table_type=TableType.EXTERNAL),
        ), patch.object(
            Table, "infos", new_callable=PropertyMock,
        ) as infos, patch.object(
            UCSchema, "staging_location", return_value=None,
        ), patch.object(
            Volume, "get_or_create",
        ) as goc:
            infos.return_value.properties = {}
            tbl.ensure_staging_volume()

        goc.assert_not_called()

    def test_managed_table_skips_external_create(self) -> None:
        """A managed table leaves the staging volume to the default (managed)
        create path — no external get_or_create at the staging boundary."""
        tbl = _table("cat", "sch", "tbl")
        with patch.object(
            Table, "infos", new_callable=PropertyMock,
        ) as infos, patch.object(
            Volume, "get_or_create",
        ) as goc:
            infos.return_value.table_type = TableType.MANAGED
            vol = tbl.ensure_staging_volume()

        assert isinstance(vol, Volume)
        goc.assert_not_called()


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
