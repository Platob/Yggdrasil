"""Tests for :class:`Volume` and :class:`Volumes`.

The fast path here is "two callers asking for the same UC volume
collapse to one singleton with one cached VolumeInfo". The
secondary path is the 5-minute TTL refresh and the cascade
helpers (``_ensure_volume``).
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.volume import Volume, Volumes


class NotFound(Exception):
    pass


@pytest.fixture(autouse=True)
def reset_volume_singletons():
    Volume._INSTANCES.clear()
    yield
    Volume._INSTANCES.clear()


@pytest.fixture
def client():
    c = MagicMock()
    c.host = "https://example.cloud.databricks.com"
    return c


@pytest.fixture
def workspace(client):
    return client.workspace_client.return_value


def _info(
    *,
    catalog: str = "cat",
    schema: str = "sch",
    name: str = "vol",
    volume_id: str = "vid-1",
    storage_location: str = "s3://bkt/u/c/s/v",
):
    return SimpleNamespace(
        catalog_name=catalog,
        schema_name=schema,
        name=name,
        volume_id=volume_id,
        volume_type="MANAGED",
        storage_location=storage_location,
        full_name=f"{catalog}.{schema}.{name}",
        owner=None,
        comment=None,
    )


class TestVolumeSingleton:

    def test_same_coords_collapse_to_one_instance(self, workspace, client):
        v1 = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        v2 = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        assert v1 is v2

    def test_different_volume_yields_different_instance(self, workspace, client):
        svc = Volumes(client=client)
        a = svc.volume(catalog_name="cat", schema_name="sch", volume_name="a")
        b = svc.volume(catalog_name="cat", schema_name="sch", volume_name="b")
        assert a is not b

    def test_different_hosts_yield_different_instances(self, workspace):
        c1 = MagicMock(); c1.host = "https://host-a"
        c2 = MagicMock(); c2.host = "https://host-b"
        a = Volumes(client=c1).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        b = Volumes(client=c2).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        assert a is not b

    def test_construction_requires_full_coords(self, workspace, client):
        with pytest.raises(ValueError):
            Volume(service=Volumes(client=client), catalog_name="cat",
                   schema_name="sch", volume_name=None)


class TestVolumeInfoTTL:

    def test_first_access_hits_sdk_subsequent_hit_cache(self, workspace, client):
        workspace.volumes.read.return_value = _info()
        v = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        i1 = v.info
        i2 = v.info
        assert i1 is i2
        workspace.volumes.read.assert_called_once_with("cat.sch.vol")

    def test_refresh_forces_reread(self, workspace, client):
        workspace.volumes.read.side_effect = [_info(), _info()]
        v = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        v.read_info()
        v.read_info(refresh=True)
        assert workspace.volumes.read.call_count == 2

    def test_expired_entry_triggers_reread(self, workspace, client):
        workspace.volumes.read.side_effect = [_info(), _info()]
        v = Volume(
            service=Volumes(client=client),
            catalog_name="cat", schema_name="sch", volume_name="vol",
            infos_ttl=0.0,
        )
        v.read_info()
        # TTL=0 → immediately stale; second call re-reads.
        v.read_info()
        assert workspace.volumes.read.call_count == 2

    def test_default_ttl_is_five_minutes(self):
        assert Volume.DEFAULT_INFO_TTL == 300.0

    def test_seeded_info_skips_first_read(self, workspace, client):
        v = Volume(
            service=Volumes(client=client),
            catalog_name="cat", schema_name="sch", volume_name="vol",
            infos=_info(),
        )
        _ = v.info
        workspace.volumes.read.assert_not_called()


class TestVolumeNavigation:

    def test_path_builds_volume_path_at_volume_root(self, workspace, client):
        from yggdrasil.databricks.fs.volume_path import VolumePath
        v = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        p = v.path()
        assert isinstance(p, VolumePath)
        assert p.full_path() == "/Volumes/cat/sch/vol"

    def test_path_appends_sub(self, workspace, client):
        v = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        p = v.path("sub/x.parquet")
        assert p.full_path() == "/Volumes/cat/sch/vol/sub/x.parquet"

    def test_catalog_and_schema_cached(self, workspace, client):
        sentinel_cat = object()
        sentinel_sch = object()
        client.catalogs.catalog.return_value = sentinel_cat
        client.schemas.schema.return_value = sentinel_sch
        v = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        assert v.catalog is sentinel_cat
        assert v.catalog is sentinel_cat
        client.catalogs.catalog.assert_called_once_with("cat")
        assert v.schema is sentinel_sch
        client.schemas.schema.assert_called_once_with(
            catalog_name="cat", schema_name="sch",
        )


class TestVolumesDictAccess:

    def test_three_part_dotted_name(self, workspace, client):
        v = Volumes(client=client)["main.sales.uploads"]
        assert isinstance(v, Volume)
        assert (v.catalog_name, v.schema_name, v.volume_name) == (
            "main", "sales", "uploads",
        )

    def test_two_part_with_catalog_default(self, workspace, client):
        v = Volumes(client=client, catalog_name="main")["sales.uploads"]
        assert (v.catalog_name, v.schema_name, v.volume_name) == (
            "main", "sales", "uploads",
        )

    def test_one_part_with_full_default(self, workspace, client):
        v = Volumes(client=client, catalog_name="main", schema_name="sales")["uploads"]
        assert (v.catalog_name, v.schema_name, v.volume_name) == (
            "main", "sales", "uploads",
        )

    def test_one_part_without_defaults_raises(self, workspace, client):
        with pytest.raises(ValueError, match="catalog_name"):
            Volumes(client=client)["uploads"]


class TestVolumePathDelegation:
    """:class:`VolumePath` reads metadata via its :class:`Volume`
    singleton, so the SDK call count collapses to one per
    ``(catalog, schema, volume)`` no matter how many paths address
    that volume."""

    def test_two_paths_share_one_volume_info_read(self, workspace, client):
        from yggdrasil.databricks.fs.volume_path import VolumePath
        workspace.volumes.read.return_value = _info()
        p1 = VolumePath("/Volumes/cat/sch/vol/a", client=client)
        p2 = VolumePath("/Volumes/cat/sch/vol/b", client=client)
        assert p1.volume is p2.volume
        p1.volume_info()
        p2.volume_info()
        workspace.volumes.read.assert_called_once_with("cat.sch.vol")
