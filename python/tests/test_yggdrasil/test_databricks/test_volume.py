"""Tests for :class:`Volume` and :class:`Volumes`.

The fast path here is "two callers asking for the same UC volume
collapse to one singleton with one cached VolumeInfo". The
secondary path is the 5-minute TTL refresh and the cascade
"""
from __future__ import annotations

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


class _PicklableClient:
    """Minimal picklable stand-in for ``DatabricksClient`` — the MagicMock
    fixture used elsewhere in this file can't survive ``pickle.dumps``."""

    def __init__(self, host: str):
        self.host = host

    def __eq__(self, other):
        return isinstance(other, _PicklableClient) and self.host == other.host

    def __hash__(self):
        return hash((type(self), self.host))


class TestVolumePickle:
    """Pickle round-trip preserves the client (and thus the host /
    auth context), not just the three-part name."""

    def test_in_process_collapses_to_live_singleton(self):
        import pickle
        client = _PicklableClient(host="https://example.cloud.databricks.com")
        v = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        loaded = pickle.loads(pickle.dumps(v))
        assert loaded is v

    def test_cross_process_carries_source_client(self):
        import pickle
        client = _PicklableClient(host="https://example.cloud.databricks.com")
        v = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        payload = pickle.dumps(v)
        # Simulate a fresh process: drop the live singleton cache so
        # ``__new__`` has to rebuild from the pickle stream alone.
        Volume._INSTANCES.clear()
        loaded = pickle.loads(payload)
        assert loaded is not v
        # The client that travelled in the pickle is the one the
        # unpickled Volume uses — not ``Volumes.current()``.
        assert loaded.client == client
        assert loaded.client.host == client.host
        assert loaded.full_name() == "cat.sch.vol"


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

    def test_default_ttl_is_fifteen_minutes(self):
        assert Volume.DEFAULT_INFO_TTL == 900.0

    def test_seeded_info_skips_first_read(self, workspace, client):
        v = Volume(
            service=Volumes(client=client),
            catalog_name="cat", schema_name="sch", volume_name="vol",
            infos=_info(),
        )
        _ = v.info
        workspace.volumes.read.assert_not_called()

    def test_exists_ignores_fresh_cache_and_probes_live(self, workspace, client):
        # exists() is a liveness probe — a still-fresh cached VolumeInfo
        # must NOT shortcut it, or a volume dropped within the TTL keeps
        # reporting True. Seed a fresh cache, then have the live read
        # not-found: exists() must re-probe and return False.
        from databricks.sdk.errors import NotFound as SDKNotFound

        v = Volume(
            service=Volumes(client=client),
            catalog_name="cat", schema_name="sch", volume_name="vol",
            infos=_info(),
        )
        assert v._is_fresh()                       # cache says "exists"
        workspace.volumes.read.side_effect = SDKNotFound(
            "Volume 'cat.sch.vol' does not exist."
        )

        assert v.exists() is False                 # ground truth wins
        workspace.volumes.read.assert_called_once_with("cat.sch.vol")
        # Stale entry dropped so a later info access doesn't resurrect it.
        assert v._infos is None


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


class TestVolumesServiceCreate:
    """``Volumes.create(...)`` resolves the :class:`Volume` and delegates to
    :meth:`Volume.create`, which is **idempotent** (reads first) and
    auto-creates the missing schema / catalog parents through
    :meth:`UCSchema.get_or_create` when the read reports them missing — never
    punting the recovery to the caller."""

    def test_create_is_idempotent_when_volume_exists(self, workspace, client):
        # The volume already exists: the read succeeds, so create is a no-op
        # — no ``volumes.create`` call.
        workspace.volumes.read.return_value = _info()
        v = Volumes(client=client).create(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        assert isinstance(v, Volume)
        workspace.volumes.create.assert_not_called()

    def test_create_makes_missing_volume(self, workspace, client):
        # Volume missing, parents exist: the read NotFounds without naming the
        # schema, so a single ``volumes.create`` lands and no parents are made.
        workspace.volumes.read.side_effect = NotFound("Volume does not exist")
        workspace.volumes.create.return_value = _info()

        v = Volumes(client=client).create(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        assert isinstance(v, Volume)
        workspace.volumes.create.assert_called_once()
        workspace.schemas.create.assert_not_called()
        workspace.catalogs.create.assert_not_called()

    def test_create_ensures_parents_on_error_and_retries(self, workspace, client):
        # The volume is missing (read NotFounds); the first ``volumes.create``
        # then NotFounds because the schema is missing → ensure the parent
        # schema (cascading to the catalog — see the schema tests) through the
        # high-level ``client.schemas`` service, and retry the create.
        workspace.volumes.read.side_effect = NotFound("Volume does not exist")
        creates = [NotFound("Schema does not exist"), _info()]

        def vol_create(**_kw):
            r = creates.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.volumes.create.side_effect = vol_create

        v = Volumes(client=client).create(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        assert isinstance(v, Volume)
        client.schemas.schema.return_value.get_or_create.assert_called_once()
        assert workspace.volumes.create.call_count == 2

    def test_create_skips_parent_ensure_when_create_succeeds(self, workspace, client):
        # Volume missing, parents present: the create lands first try, so no
        # parent ensure and a single ``volumes.create``.
        workspace.volumes.read.side_effect = NotFound("Volume does not exist")
        workspace.volumes.create.return_value = _info()

        Volumes(client=client).create(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        client.schemas.schema.return_value.get_or_create.assert_not_called()
        workspace.volumes.create.assert_called_once()

    def test_create_refreshes_stale_missing_stat(self, workspace, client):
        # A prior probe cached the path stat as MISSING; create must refresh
        # the stat cache (in lock-step with read info) so a follow-up
        # is_dir()/exists() sees the volume — no stale MISSING.
        workspace.volumes.read.side_effect = NotFound("Volume does not exist")
        workspace.volumes.create.return_value = _info()
        v = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        assert v.is_dir() is False          # MISSING stat cached from the probe
        v.create()
        assert v.is_dir() is True           # create refreshed the stat cache


class TestVolumePathDelegation:
    """:class:`VolumePath` reads metadata via its :class:`Volume`
    singleton, so the SDK call count collapses to one per
    ``(catalog, schema, volume)`` no matter how many paths address
    that volume."""

    def test_two_paths_share_one_volume_info_read(self, workspace, client):
        from yggdrasil.databricks.fs.volume_path import VolumePath
        workspace.volumes.read.return_value = _info()
        service = Volumes(client=client)
        p1 = VolumePath("/Volumes/cat/sch/vol/a", service=service)
        p2 = VolumePath("/Volumes/cat/sch/vol/b", service=service)
        assert p1.volume is p2.volume
        p1.volume_info()
        p2.volume_info()
        workspace.volumes.read.assert_called_once_with("cat.sch.vol")


# --------------------------------------------------------------------------- #
# get_or_create — external (by storage URI) vs managed (by dotted name)
# --------------------------------------------------------------------------- #
from unittest.mock import patch  # noqa: E402


class TestGetOrCreate:
    def test_external_miss_creates_external_with_explicit_name(self, client):
        svc = Volumes(client=client, catalog_name="main", schema_name="ext")
        created = object()
        with patch.object(Volumes, "find", return_value=None) as find, \
             patch.object(Volumes, "create", return_value=created) as create:
            out = svc.get_or_create("s3://my-bucket/raw/events", volume_name="events")
        assert out is created
        find.assert_called_once_with(
            catalog_name="main", schema_name="ext", volume_name="events", raise_error=False,
        )
        ckw = create.call_args.kwargs
        assert (ckw["catalog_name"], ckw["schema_name"], ckw["volume_name"]) == ("main", "ext", "events")
        assert ckw["storage_location"] == "s3://my-bucket/raw/events"
        assert ckw["volume_type"] == "EXTERNAL"

    def test_external_hit_returns_existing_without_create(self, client):
        svc = Volumes(client=client, catalog_name="main", schema_name="ext")
        existing = object()
        with patch.object(Volumes, "find", return_value=existing), \
             patch.object(Volumes, "create") as create:
            out = svc.get_or_create("s3://b/p", volume_name="myvol")
        assert out is existing
        create.assert_not_called()

    def test_storage_location_kwarg_routes_external_with_name_override(self, client):
        svc = Volumes(client=client, catalog_name="main", schema_name="ext")
        with patch.object(Volumes, "find", return_value=None), \
             patch.object(Volumes, "create", return_value=object()) as create:
            svc.get_or_create("ignored", storage_location="s3://b/p/q", volume_name="myvol")
        ckw = create.call_args.kwargs
        assert ckw["volume_name"] == "myvol"              # explicit name
        assert ckw["storage_location"] == "s3://b/p/q"
        assert ckw["volume_type"] == "EXTERNAL"

    def test_external_requires_volume_name(self, client):
        svc = Volumes(client=client, catalog_name="main", schema_name="ext")
        with pytest.raises(ValueError, match="volume_name"):
            svc.get_or_create("s3://b/p")                 # no derived name anymore

    def test_external_requires_catalog_and_schema(self, client):
        svc = Volumes(client=client)                      # no defaults
        with pytest.raises(ValueError):
            svc.get_or_create("s3://b/p", volume_name="myvol")

    def test_managed_dotted_name_miss_creates_managed(self, client):
        svc = Volumes(client=client)
        with patch.object(Volumes, "find", return_value=None) as find, \
             patch.object(Volumes, "create", return_value=object()) as create:
            svc.get_or_create("main.sales.uploads")
        find.assert_called_once_with(
            catalog_name="main", schema_name="sales", volume_name="uploads", raise_error=False,
        )
        ckw = create.call_args.kwargs
        assert (ckw["catalog_name"], ckw["schema_name"], ckw["volume_name"]) == ("main", "sales", "uploads")
        assert ckw.get("volume_type") is None             # managed, not external


class TestVolumeDeleteClearsCaches:
    """Deleting a volume drops every storage-derived cache so a re-probe /
    rebind never reads stale state."""

    def test_delete_resets_info_storage_and_external_caches(self, workspace, client):
        from unittest.mock import MagicMock

        from yggdrasil.databricks.volume.volume import _UNRESOLVED

        v = Volumes(client=client).volume(
            catalog_name="cat", schema_name="sch", volume_name="vol",
        )
        # Seed every cache the way live use would.
        v._store_infos(_info())
        v._external_location = MagicMock()
        v._external_readable = True
        v._external_writable = True
        v._storage_paths = {True: MagicMock(), False: MagicMock()}

        v.delete()

        assert v._infos is None
        assert v._infos_fetched_at is None
        assert v._external_location is _UNRESOLVED
        assert v._external_readable is None
        assert v._external_writable is None
        assert v._storage_paths == {}
        # The UC volume was actually deleted.
        workspace.volumes.delete.assert_called_once_with(name="cat.sch.vol")
