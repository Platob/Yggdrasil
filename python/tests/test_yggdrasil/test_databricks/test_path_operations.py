"""Unit tests for path operations on Catalog / Schema / Volume / Table.

Covers the surfaces that are purely structural — URL parsing, scheme
dispatch, identity round-trips, navigation between resources — without
hitting a live workspace. The :class:`DatabricksPath` dispatcher and
each resource's ``from_url`` / ``full_name`` / ``full_path`` are the
contract these tests pin down: depth under ``/Volumes/`` resolves to
:class:`UCCatalog` (depth 1), :class:`UCSchema` (depth 2),
:class:`Volume` (depth 3), :class:`VolumePath` (depth 4+); explicit
``dbfs+catalog://`` / ``dbfs+schema://`` / ``dbfs+volume://`` /
``dbfs+workspace://`` / ``dbfs+dbfs://`` / ``dbfs+table://`` URLs
dispatch by scheme alone.

Live-workspace coverage lives in
``test_path_dispatch_integration.py`` — these tests stay mock-only so
they run on every push.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.catalog.catalog import UCCatalog
from yggdrasil.databricks.catalog.catalogs import Catalogs
from yggdrasil.databricks.fs.dbfs_path import DBFSPath
from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.databricks.fs.workspace_path import WorkspacePath
from yggdrasil.databricks.path import (
    DatabricksPath,
    _coerce_to_url_str,
    _looks_like_posix,
    _parse_posix,
    _resolve_databricks_subclass,
    _strip_dbfs_family_prefix,
)
from yggdrasil.databricks.schema.schema import UCSchema
from yggdrasil.databricks.schema.schemas import Schemas
from yggdrasil.databricks.table.table import Table
from yggdrasil.databricks.table.tables import Tables
from yggdrasil.databricks.volume.volume import Volume
from yggdrasil.databricks.volume.volumes import Volumes
from yggdrasil.enums import Scheme
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_databricks_singletons():
    """Drop the per-class singleton caches so a test can't see a
    UCCatalog / UCSchema / Volume / VolumePath built by a previous test
    even if the URL key collides."""
    UCCatalog._INSTANCES.clear()
    UCSchema._INSTANCES.clear()
    Volume._INSTANCES.clear()
    VolumePath._INSTANCES.clear()
    DBFSPath._INSTANCES.clear()
    WorkspacePath._INSTANCES.clear()
    DatabricksPath._INSTANCES.clear()
    yield
    UCCatalog._INSTANCES.clear()
    UCSchema._INSTANCES.clear()
    Volume._INSTANCES.clear()
    VolumePath._INSTANCES.clear()
    DBFSPath._INSTANCES.clear()
    WorkspacePath._INSTANCES.clear()
    DatabricksPath._INSTANCES.clear()


@pytest.fixture
def client():
    c = MagicMock()
    c.base_url = URL(scheme="https", host="ws.cloud.databricks.com", path="/")
    c.host = "ws.cloud.databricks.com"
    return c


@pytest.fixture
def catalogs_service(client):
    svc = MagicMock(spec=Catalogs)
    svc.client = client
    return svc


@pytest.fixture
def schemas_service(client):
    svc = MagicMock(spec=Schemas)
    svc.client = client
    return svc


@pytest.fixture
def volumes_service(client):
    svc = MagicMock(spec=Volumes)
    svc.client = client
    return svc


@pytest.fixture
def tables_service(client):
    svc = MagicMock(spec=Tables)
    svc.client = client
    return svc


# ===========================================================================
# Pure POSIX coercion helpers
# ===========================================================================


class TestPosixCoercion:
    """The module-private string helpers that turn ``/Volumes/...`` /
    ``/dbfs/...`` / ``/Workspace/...`` into canonical ``dbfs+…://`` URLs."""

    def test_looks_like_posix_recognises_volumes(self):
        assert _looks_like_posix("/Volumes/cat/sch/vol/x")

    def test_looks_like_posix_recognises_dbfs_lowercase(self):
        assert _looks_like_posix("/dbfs/tmp/x")

    def test_looks_like_posix_recognises_dbfs_mixed_case(self):
        # The namespace lookup is case-insensitive for ``/dbfs/`` —
        # ``/Dbfs/`` and ``/DBFS/`` both flow into the DBFS branch.
        assert _looks_like_posix("/DBFS/tmp/x")

    def test_looks_like_posix_recognises_workspace(self):
        assert _looks_like_posix("/Workspace/Users/me/x")

    def test_looks_like_posix_rejects_unrelated_paths(self):
        assert not _looks_like_posix("/etc/passwd")
        assert not _looks_like_posix("relative/path")
        assert not _looks_like_posix("")
        assert not _looks_like_posix("https://example.com/x")

    def test_parse_posix_volume(self):
        scheme, path = _parse_posix("/Volumes/cat/sch/vol/data.parquet")
        assert scheme == Scheme.DATABRICKS_VOLUME.value
        assert path == "/cat/sch/vol/data.parquet"

    def test_parse_posix_dbfs_case_insensitive(self):
        scheme_lower, _ = _parse_posix("/dbfs/x")
        scheme_upper, _ = _parse_posix("/DBFS/x")
        assert scheme_lower == scheme_upper == Scheme.DATABRICKS_DBFS.value

    def test_parse_posix_workspace_root(self):
        scheme, path = _parse_posix("/Workspace")
        assert scheme == Scheme.DATABRICKS_WORKSPACE.value
        assert path == "/"

    def test_parse_posix_rejects_non_databricks_namespace(self):
        with pytest.raises(ValueError):
            _parse_posix("/etc/passwd")

    def test_coerce_volume_string_to_url_form(self):
        assert (
            _coerce_to_url_str("/Volumes/cat/sch/vol/x")
            == f"{Scheme.DATABRICKS_VOLUME.value}:///cat/sch/vol/x"
        )

    def test_coerce_passes_through_unrelated(self):
        # ``_coerce_to_url_str`` is the boundary helper — anything that
        # doesn't match a known POSIX namespace flows through untouched
        # so the URL parser sees the original string.
        assert _coerce_to_url_str("https://example.com/x") == "https://example.com/x"
        assert _coerce_to_url_str(42) == 42


# ===========================================================================
# dbfs:// family prefix expansion
# ===========================================================================


class TestStripDbfsFamilyPrefix:
    """``dbfs:///...`` is the legacy un-qualified family URL.
    :func:`_strip_dbfs_family_prefix` rewrites it into the concrete
    ``dbfs+<surface>://`` form based on the leading path segment."""

    def test_volumes_prefix_becomes_volume_scheme(self):
        u = URL.from_("dbfs:///Volumes/cat/sch/vol")
        out = _strip_dbfs_family_prefix(u)
        assert out.scheme == Scheme.DATABRICKS_VOLUME.value
        assert out.path == "/cat/sch/vol"

    def test_workspace_prefix_becomes_workspace_scheme(self):
        u = URL.from_("dbfs:///Workspace/Users/me")
        out = _strip_dbfs_family_prefix(u)
        assert out.scheme == Scheme.DATABRICKS_WORKSPACE.value
        assert out.path == "/Users/me"

    def test_unprefixed_dbfs_becomes_dbfs_scheme(self):
        u = URL.from_("dbfs:///tmp/x")
        out = _strip_dbfs_family_prefix(u)
        assert out.scheme == Scheme.DATABRICKS_DBFS.value
        assert out.path == "/tmp/x"

    def test_already_qualified_passes_through(self):
        u = URL.from_("dbfs+volume:///cat/sch/vol/x")
        out = _strip_dbfs_family_prefix(u)
        assert out == u


# ===========================================================================
# Subclass dispatch by URL shape
# ===========================================================================


class TestResolveDatabricksSubclass:
    """:func:`_resolve_databricks_subclass` is the depth- and
    scheme-based switch every dispatch entry point ultimately funnels
    through."""

    def test_volume_path_depth_one_resolves_to_catalog(self):
        cls, _ = _resolve_databricks_subclass(data="/Volumes/cat")
        assert cls is UCCatalog

    def test_volume_path_depth_two_resolves_to_schema(self):
        cls, _ = _resolve_databricks_subclass(data="/Volumes/cat/sch")
        assert cls is UCSchema

    def test_volume_path_depth_three_resolves_to_volume(self):
        cls, _ = _resolve_databricks_subclass(data="/Volumes/cat/sch/vol")
        assert cls is Volume

    def test_volume_path_depth_four_resolves_to_volume_path(self):
        cls, _ = _resolve_databricks_subclass(data="/Volumes/cat/sch/vol/x")
        assert cls is VolumePath

    def test_volume_path_deep_resolves_to_volume_path(self):
        cls, _ = _resolve_databricks_subclass(
            data="/Volumes/cat/sch/vol/a/b/c.parquet",
        )
        assert cls is VolumePath

    def test_workspace_posix_resolves_to_workspace_path(self):
        cls, _ = _resolve_databricks_subclass(data="/Workspace/Users/me")
        assert cls is WorkspacePath

    def test_dbfs_posix_resolves_to_dbfs_path(self):
        cls, _ = _resolve_databricks_subclass(data="/dbfs/tmp/x")
        assert cls is DBFSPath

    def test_explicit_catalog_scheme_resolves_to_uc_catalog(self):
        cls, _ = _resolve_databricks_subclass(data="dbfs+catalog:///cat")
        assert cls is UCCatalog

    def test_explicit_schema_scheme_resolves_to_uc_schema(self):
        cls, _ = _resolve_databricks_subclass(data="dbfs+schema:///cat/sch")
        assert cls is UCSchema

    def test_table_scheme_resolves_to_table(self):
        cls, _ = _resolve_databricks_subclass(
            data="dbfs+table:///cat/sch/tbl",
        )
        assert cls is Table

    def test_unqualified_dbfs_family_expands_via_path(self):
        # Bare ``dbfs:///Volumes/...`` still routes to the depth-based
        # switch — ``_strip_dbfs_family_prefix`` runs first.
        cls, normalized = _resolve_databricks_subclass(
            data="dbfs:///Volumes/cat/sch/vol/x",
        )
        assert cls is VolumePath
        assert normalized.scheme == Scheme.DATABRICKS_VOLUME.value

    def test_url_keyword_takes_precedence_over_data(self):
        cls, _ = _resolve_databricks_subclass(
            data="/Volumes/cat/sch/vol/x",
            url=URL.from_("dbfs+workspace:///Users/me"),
        )
        assert cls is WorkspacePath

    def test_no_inputs_falls_back_to_dbfs_path(self):
        cls, normalized = _resolve_databricks_subclass()
        assert cls is DBFSPath
        assert normalized is None


# ===========================================================================
# DatabricksPath() end-to-end dispatch with mocked services
# ===========================================================================


class TestDatabricksPathDispatchFs:
    """End-to-end :class:`DatabricksPath` dispatch for the byte-shaped
    fs subclasses (DBFS / Volumes path / Workspace). The depth=4+
    branch is the typical user-facing call:
    ``DatabricksPath("/Volumes/cat/sch/vol/x")`` resolves to a
    :class:`VolumePath` without ever calling
    :meth:`DatabricksClient.current` when ``service=`` is supplied."""

    def test_volumes_depth_four_yields_volume_path(self, volumes_service):
        p = DatabricksPath(
            "/Volumes/cat/sch/vol/data.parquet",
            service=volumes_service,
        )
        assert isinstance(p, VolumePath)
        assert p.full_path() == "/Volumes/cat/sch/vol/data.parquet"

    def test_volumes_deep_path_yields_volume_path(self, volumes_service):
        p = DatabricksPath(
            "/Volumes/cat/sch/vol/year=2026/data.parquet",
            service=volumes_service,
        )
        assert isinstance(p, VolumePath)
        assert (
            p.full_path()
            == "/Volumes/cat/sch/vol/year=2026/data.parquet"
        )

    def test_workspace_posix_yields_workspace_path(self):
        from yggdrasil.databricks.workspaces.service import Workspaces
        svc = MagicMock(spec=Workspaces)
        p = DatabricksPath("/Workspace/Users/me/x", service=svc)
        assert isinstance(p, WorkspacePath)

    def test_dbfs_posix_yields_dbfs_path(self):
        from yggdrasil.databricks.fs.service import DBFSService
        svc = MagicMock(spec=DBFSService)
        p = DatabricksPath("/dbfs/tmp/x", service=svc)
        assert isinstance(p, DBFSPath)

    def test_dispatcher_volume_path_url_form(self, volumes_service):
        # Explicit ``dbfs+volume://`` URL with a deep path still
        # routes through the depth=4+ branch.
        p = DatabricksPath(
            "dbfs+volume:///cat/sch/vol/x",
            service=volumes_service,
        )
        assert isinstance(p, VolumePath)
        assert p.full_path() == "/Volumes/cat/sch/vol/x"


# ===========================================================================
# from_url round-trips
# ===========================================================================


class TestFromUrl:
    """Each resource's ``from_url`` must parse the canonical
    ``dbfs+<surface>://[host]/<segments>`` shape and bind the right
    service."""

    def test_uc_catalog_from_url(self, catalogs_service):
        cat = UCCatalog.from_url(
            "dbfs+catalog:///main",
            service=catalogs_service,
        )
        assert cat.catalog_name == "main"

    def test_uc_catalog_from_url_rejects_empty_path(self, catalogs_service):
        with pytest.raises(ValueError):
            UCCatalog.from_url("dbfs+catalog:///", service=catalogs_service)

    def test_uc_schema_from_url(self, schemas_service):
        sch = UCSchema.from_url(
            "dbfs+schema:///main/sales",
            service=schemas_service,
        )
        assert sch.catalog_name == "main"
        assert sch.schema_name == "sales"

    def test_uc_schema_from_url_rejects_one_segment(self, schemas_service):
        with pytest.raises(ValueError):
            UCSchema.from_url(
                "dbfs+schema:///main",
                service=schemas_service,
            )

    def test_volume_from_url(self, volumes_service):
        vol = Volume.from_url(
            "dbfs+volume:///main/sales/raw",
            service=volumes_service,
        )
        assert vol.catalog_name == "main"
        assert vol.schema_name == "sales"
        assert vol.volume_name == "raw"

    def test_volume_from_url_rejects_partial_coords(self, volumes_service):
        with pytest.raises(ValueError):
            Volume.from_url(
                "dbfs+volume:///main/sales",
                service=volumes_service,
            )

    def test_table_from_url(self, tables_service):
        tbl = Table.from_url(
            "dbfs+table:///main/sales/orders",
            service=tables_service,
        )
        assert tbl.catalog_name == "main"
        assert tbl.schema_name == "sales"
        assert tbl.table_name == "orders"


# ===========================================================================
# Identity round-trips
# ===========================================================================


class TestFullNameAndPath:
    """``full_name`` is the dotted SQL identifier; ``full_path`` is its
    POSIX projection. Both honour identity round-trips so the two never
    drift from the underlying ``catalog_name`` / ``schema_name`` /
    ``volume_name`` / ``table_name`` triple."""

    def test_uc_catalog_full_name(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert cat.full_name() == "main"

    def test_uc_catalog_full_name_quoted(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert cat.full_name(safe=True) == "`main`"

    def test_uc_schema_full_name(self, schemas_service):
        sch = UCSchema(
            service=schemas_service,
            catalog_name="main",
            schema_name="sales",
        )
        assert sch.full_name() == "main.sales"
        assert sch.full_path() == "/Schemas/main/sales"

    def test_uc_schema_full_name_quoted(self, schemas_service):
        sch = UCSchema(
            service=schemas_service,
            catalog_name="main",
            schema_name="sales",
        )
        assert sch.full_name(safe=True) == "`main`.`sales`"

    def test_volume_full_name(self, volumes_service):
        vol = Volume(
            service=volumes_service,
            catalog_name="main",
            schema_name="sales",
            volume_name="raw",
        )
        assert vol.full_name() == "main.sales.raw"
        assert vol.name == "raw"

    def test_volume_full_name_quoted(self, volumes_service):
        vol = Volume(
            service=volumes_service,
            catalog_name="main",
            schema_name="sales",
            volume_name="raw",
        )
        assert vol.full_name(safe=True) == "`main`.`sales`.`raw`"

    def test_table_full_name(self, tables_service):
        tbl = Table(
            service=tables_service,
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
        )
        assert tbl.full_name() == "main.sales.orders"
        assert tbl.name == "orders"


# ===========================================================================
# Navigation between resources
# ===========================================================================


class TestNavigation:
    """The path-shaped navigation surface — ``Volume.path()`` produces a
    :class:`VolumePath` rooted at the volume; ``Volume.catalog`` and
    ``Volume.schema`` walk back up the UC hierarchy."""

    def test_volume_path_at_root(self, volumes_service):
        vol = Volume(
            service=volumes_service,
            catalog_name="main",
            schema_name="sales",
            volume_name="raw",
        )
        p = vol.path()
        assert isinstance(p, VolumePath)
        assert p.full_path() == "/Volumes/main/sales/raw"

    def test_volume_path_with_sub_segment(self, volumes_service):
        vol = Volume(
            service=volumes_service,
            catalog_name="main",
            schema_name="sales",
            volume_name="raw",
        )
        p = vol.path("year=2026/data.parquet")
        assert isinstance(p, VolumePath)
        assert p.full_path() == "/Volumes/main/sales/raw/year=2026/data.parquet"

    def test_volume_path_strips_leading_slash_on_sub(self, volumes_service):
        vol = Volume(
            service=volumes_service,
            catalog_name="main",
            schema_name="sales",
            volume_name="raw",
        )
        # Leading slash on ``sub`` must not double up the joiner —
        # ``/Volumes/main/sales/raw/x`` not ``/Volumes/main/sales/raw//x``.
        p = vol.path("/x")
        assert p.full_path() == "/Volumes/main/sales/raw/x"

    def test_volume_catalog_navigates_to_uc_catalog(self, client):
        sentinel = object()
        client.catalogs.catalog.return_value = sentinel
        vol = Volumes(client=client).volume(
            catalog_name="main", schema_name="sales", volume_name="raw",
        )
        assert vol.catalog is sentinel
        # Cached on the per-instance slot — second access skips the call.
        assert vol.catalog is sentinel
        client.catalogs.catalog.assert_called_once_with("main")

    def test_volume_schema_navigates_to_uc_schema(self, client):
        sentinel = object()
        client.schemas.schema.return_value = sentinel
        vol = Volumes(client=client).volume(
            catalog_name="main", schema_name="sales", volume_name="raw",
        )
        assert vol.schema is sentinel
        assert vol.schema is sentinel
        client.schemas.schema.assert_called_once_with(
            catalog_name="main", schema_name="sales",
        )

    def test_uc_catalog_getitem_returns_schema(self, catalogs_service):
        # ``catalog["sch"]`` is the dict-like shortcut for
        # ``catalog.schema("sch")``; it builds a :class:`UCSchema`
        # bound to the same service the catalog uses, with the
        # catalog's name pre-filled.
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        sch = cat["sales"]
        assert isinstance(sch, UCSchema)
        assert sch.catalog_name == "main"
        assert sch.schema_name == "sales"
        assert sch.service is catalogs_service


# ===========================================================================
# Logical-resource byte ops must raise
# ===========================================================================


class TestLogicalResourceByteOpsRaise:
    """:class:`UCCatalog`, :class:`UCSchema`, and :class:`Table` are
    logical Unity Catalog resources — the byte-shaped Holder primitives
    must reject so misuse fails loudly with a hint at the right
    surface."""

    def test_uc_catalog_read_mv_raises(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        with pytest.raises(NotImplementedError):
            cat._read_mv(-1, 0)

    def test_uc_catalog_write_mv_raises(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        with pytest.raises(NotImplementedError):
            cat._write_mv(memoryview(b""), 0)

    def test_uc_schema_read_mv_raises(self, schemas_service):
        sch = UCSchema(
            service=schemas_service,
            catalog_name="main",
            schema_name="sales",
        )
        with pytest.raises(NotImplementedError):
            sch._read_mv(-1, 0)

    def test_table_read_mv_raises(self, tables_service):
        tbl = Table(
            service=tables_service,
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
        )
        with pytest.raises(NotImplementedError):
            tbl._read_mv(-1, 0)

    def test_table_truncate_raises(self, tables_service):
        tbl = Table(
            service=tables_service,
            catalog_name="main",
            schema_name="sales",
            table_name="orders",
        )
        with pytest.raises(NotImplementedError):
            tbl.truncate(0)


# ===========================================================================
# Singleton identity through the dispatcher
# ===========================================================================


class TestSingletonIdentityThroughDispatcher:
    """Repeated dispatch on the same POSIX seed must collapse onto the
    same singleton — every caller in the process should share the
    cached resource state."""

    def test_same_volume_path_collapses(self, volumes_service):
        a = DatabricksPath(
            "/Volumes/main/sales/raw/x.parquet",
            service=volumes_service,
        )
        b = DatabricksPath(
            "/Volumes/main/sales/raw/x.parquet",
            service=volumes_service,
        )
        assert a is b

    def test_same_volume_collapses_via_service(self, client):
        # The dispatcher entry point for ``Volume`` is constrained by
        # the legacy double-__init__ shape of resource-typed targets;
        # the canonical "two callers, one resource" identity test for
        # volumes runs through ``Volumes.volume(...)`` instead — same
        # singleton, just via the explicit factory.
        a = Volumes(client=client).volume(
            catalog_name="main", schema_name="sales", volume_name="raw",
        )
        b = Volumes(client=client).volume(
            catalog_name="main", schema_name="sales", volume_name="raw",
        )
        assert a is b

    def test_url_form_and_posix_form_collapse(self, volumes_service):
        # The dispatcher normalizes ``/Volumes/cat/sch/vol/x`` to
        # ``dbfs+volume:///cat/sch/vol/x`` before keying the singleton
        # cache, so both spellings should land on the same instance.
        a = DatabricksPath(
            "/Volumes/main/sales/raw/x", service=volumes_service,
        )
        b = DatabricksPath(
            "dbfs+volume:///main/sales/raw/x", service=volumes_service,
        )
        assert a is b
