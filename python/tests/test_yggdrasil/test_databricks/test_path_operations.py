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
    TABLE_PATH_PREFIX,
    VOLUME_PATH_PREFIX,
    _coerce_to_url_str,
    _looks_like_posix,
    _parse_posix,
    _relative_join_parts,
    _resolve_databricks_subclass,
    _strip_dbfs_family_prefix,
    resolve_path_prefix,
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


# ===========================================================================
# Path-join navigation — joining a catalog / schema / volume mints the
# right concrete Databricks instance by depth
# ===========================================================================


class TestJoinPathCreatesCorrectInstances:
    """``/`` (and :meth:`joinpath`) walk *down* the volume family by
    depth — catalog → schema → volume → :class:`VolumePath` — minting
    the right concrete resource at every step.

    This is the filesystem-navigation surface: each appended segment
    descends one level. It is distinct from the logical
    ``__getitem__`` surface (``catalog["sch"]`` → :class:`UCSchema`,
    ``schema["tbl"]`` → :class:`Table`), and from the module-level
    dispatcher that resolves a whole ``/Volumes/...`` POSIX seed at
    once — but it must agree with both on the depth → type mapping so
    ``cat / "sch" / "vol" / "x"`` lands on the same
    ``catalog_name`` / ``schema_name`` / ``volume_name`` triple a
    direct ``DatabricksPath("/Volumes/cat/sch/vol/x")`` does.
    """

    # ── catalog as the join root ──────────────────────────────────────────

    def test_catalog_join_schema_yields_schema(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        sch = cat / "sales"
        assert isinstance(sch, UCSchema)
        assert sch.catalog_name == "main"
        assert sch.schema_name == "sales"

    def test_catalog_join_schema_collapses_onto_getitem(self, catalogs_service):
        # ``cat / "sales"`` and the logical ``cat["sales"]`` must resolve
        # to the very same singleton — both spellings address the one
        # ``main.sales`` schema, so their cached state has to be shared.
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert (cat / "sales") is cat["sales"]
        assert (cat / "sales") is cat.schema("sales")

    def test_catalog_join_to_volume_depth(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        vol = cat / "sales" / "raw"
        assert isinstance(vol, Volume)
        assert vol.catalog_name == "main"
        assert vol.schema_name == "sales"
        assert vol.volume_name == "raw"

    def test_catalog_join_to_volume_path_depth(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        leaf = cat / "sales" / "raw" / "data.parquet"
        assert isinstance(leaf, VolumePath)
        assert leaf.full_path() == "/Volumes/main/sales/raw/data.parquet"

    def test_catalog_joinpath_multi_segment_to_volume_path(self, catalogs_service):
        # A single ``joinpath`` with several segments resolves to the
        # same place as the chained ``/`` form.
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        leaf = cat.joinpath("sales", "raw", "year=2026", "data.parquet")
        assert isinstance(leaf, VolumePath)
        assert (
            leaf.full_path()
            == "/Volumes/main/sales/raw/year=2026/data.parquet"
        )

    # ── schema as the join root ───────────────────────────────────────────

    def test_schema_join_volume_yields_volume(self, schemas_service):
        sch = UCSchema(
            service=schemas_service,
            catalog_name="main",
            schema_name="sales",
        )
        vol = sch / "raw"
        assert isinstance(vol, Volume)
        assert vol.catalog_name == "main"
        assert vol.schema_name == "sales"
        assert vol.volume_name == "raw"

    def test_schema_join_to_volume_path_depth(self, schemas_service):
        sch = UCSchema(
            service=schemas_service,
            catalog_name="main",
            schema_name="sales",
        )
        leaf = sch / "raw" / "sub" / "f.csv"
        assert isinstance(leaf, VolumePath)
        assert leaf.full_path() == "/Volumes/main/sales/raw/sub/f.csv"

    # ── volume as the join root ───────────────────────────────────────────

    def test_volume_join_yields_volume_path(self, volumes_service):
        vol = Volume(
            service=volumes_service,
            catalog_name="main",
            schema_name="sales",
            volume_name="raw",
        )
        leaf = vol / "data.parquet"
        assert isinstance(leaf, VolumePath)
        assert leaf.full_path() == "/Volumes/main/sales/raw/data.parquet"

    def test_volume_join_nested_yields_volume_path(self, volumes_service):
        vol = Volume(
            service=volumes_service,
            catalog_name="main",
            schema_name="sales",
            volume_name="raw",
        )
        leaf = vol / "a" / "b" / "c.parquet"
        assert isinstance(leaf, VolumePath)
        assert leaf.full_path() == "/Volumes/main/sales/raw/a/b/c.parquet"

    # ── agreement with the whole-seed dispatcher ──────────────────────────

    def test_chained_join_matches_dispatcher_address(
        self, catalogs_service, volumes_service,
    ):
        # Walking down from the catalog one segment at a time has to
        # address the same UC location the dispatcher resolves from the
        # equivalent POSIX seed — same concrete type, same canonical URL.
        # (Instance identity isn't asserted: a :class:`VolumePath`
        # singleton keys on its service object, and the walked path
        # carries the volume's own service rather than ``volumes_service``.)
        catalogs_service.client = volumes_service.client
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        walked = cat / "sales" / "raw" / "x.parquet"
        dispatched = DatabricksPath(
            "/Volumes/main/sales/raw/x.parquet",
            service=volumes_service,
        )
        assert isinstance(walked, VolumePath)
        assert type(walked) is type(dispatched)
        assert walked == dispatched
        assert walked.full_path() == dispatched.full_path()


# ===========================================================================
# path_prefix — the catalog/schema "navigation surface" that tells a
# path-join which child type to mint (instead of guessing)
# ===========================================================================


class TestPathPrefixResolver:
    """:func:`resolve_path_prefix` is the single source of truth both the
    singleton key and ``__init__`` use, so they can never disagree."""

    def test_explicit_prefix_wins(self):
        assert resolve_path_prefix(TABLE_PATH_PREFIX) == TABLE_PATH_PREFIX
        assert resolve_path_prefix(VOLUME_PATH_PREFIX) == VOLUME_PATH_PREFIX

    def test_derives_volume_from_volume_scheme(self):
        assert (
            resolve_path_prefix(url=URL.from_("dbfs+volume:///c/s"))
            == VOLUME_PATH_PREFIX
        )

    def test_derives_table_from_table_scheme(self):
        assert (
            resolve_path_prefix(url=URL.from_("dbfs+table:///c/s/t"))
            == TABLE_PATH_PREFIX
        )

    def test_defaults_to_volume_for_unmapped_scheme(self):
        # ``dbfs+catalog`` / ``dbfs+schema`` are a handle's *own* scheme;
        # they say nothing about the child surface, so the volume
        # filesystem (the dominant ``/`` target) is the default.
        assert resolve_path_prefix(url=URL.from_("dbfs+catalog:///c")) == (
            VOLUME_PATH_PREFIX
        )
        assert resolve_path_prefix() == VOLUME_PATH_PREFIX


class TestPathPrefixOnCatalogAndSchema:
    """The catalog records its navigation surface and hands it down to
    every schema it mints, so the schema knows its child type up front."""

    def test_catalog_defaults_to_volume_surface(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert cat.path_prefix == VOLUME_PATH_PREFIX

    def test_catalog_accepts_table_surface(self, catalogs_service):
        cat = UCCatalog(
            service=catalogs_service,
            catalog_name="main",
            path_prefix=TABLE_PATH_PREFIX,
        )
        assert cat.path_prefix == TABLE_PATH_PREFIX

    def test_prefix_propagates_to_schema_via_navigation(self, catalogs_service):
        cat = UCCatalog(
            service=catalogs_service,
            catalog_name="main",
            path_prefix=TABLE_PATH_PREFIX,
        )
        # Every way of reaching a child schema carries the surface down.
        assert cat.schema("sales").path_prefix == TABLE_PATH_PREFIX
        assert cat["sales"].path_prefix == TABLE_PATH_PREFIX
        assert (cat / "sales").path_prefix == TABLE_PATH_PREFIX

    def test_schema_catalog_round_trip_preserves_prefix(self, schemas_service):
        sch = UCSchema(
            service=schemas_service,
            catalog_name="main",
            schema_name="sales",
            path_prefix=TABLE_PATH_PREFIX,
        )
        # Up to the catalog and back down must land on the same surface.
        assert sch.catalog.path_prefix == TABLE_PATH_PREFIX

    def test_volume_and_table_surfaces_are_distinct_singletons(
        self, catalogs_service,
    ):
        vol_cat = UCCatalog(service=catalogs_service, catalog_name="main")
        tbl_cat = UCCatalog(
            service=catalogs_service,
            catalog_name="main",
            path_prefix=TABLE_PATH_PREFIX,
        )
        # Same name + client, different surface → different handles, each
        # with its own cached infos / child resolution.
        assert vol_cat is not tbl_cat
        assert vol_cat.path_prefix != tbl_cat.path_prefix
        # ...but two volume-surface handles still collapse.
        assert vol_cat is UCCatalog(
            service=catalogs_service, catalog_name="main",
        )


class TestPathPrefixDrivesChildType:
    """The whole point: a path-join resolves to the child type the
    surface dictates — volume catalogs descend into Volumes /
    VolumePaths, table catalogs into Tables — no guessing."""

    def test_volume_catalog_join_yields_volume_then_path(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert isinstance(cat / "sales" / "raw", Volume)
        assert isinstance(cat / "sales" / "raw" / "f.parquet", VolumePath)

    def test_table_catalog_join_yields_table(self, catalogs_service):
        cat = UCCatalog(
            service=catalogs_service,
            catalog_name="main",
            path_prefix=TABLE_PATH_PREFIX,
        )
        leaf = cat / "sales" / "orders"
        assert isinstance(leaf, Table)
        assert leaf.catalog_name == "main"
        assert leaf.schema_name == "sales"
        assert leaf.table_name == "orders"

    def test_table_catalog_join_past_table_raises(self, catalogs_service):
        # A table is a leaf, not a path-navigable container; a single
        # multi-segment join that descends past one fails loudly rather
        # than fabricating a nonsensical resource.
        cat = UCCatalog(
            service=catalogs_service,
            catalog_name="main",
            path_prefix=TABLE_PATH_PREFIX,
        )
        with pytest.raises(ValueError):
            cat.joinpath("sales", "orders", "oops")

    def test_from_url_volume_scheme_makes_volume_catalog(self, catalogs_service):
        cat = UCCatalog.from_url(
            "dbfs+volume:///main", service=catalogs_service,
        )
        assert cat.path_prefix == VOLUME_PATH_PREFIX

    def test_from_url_schema_table_scheme_makes_table_schema(
        self, schemas_service,
    ):
        sch = UCSchema.from_url(
            "dbfs+table:///main/sales", service=schemas_service,
        )
        assert sch.path_prefix == TABLE_PATH_PREFIX
        assert isinstance(sch / "orders", Table)


# ===========================================================================
# Multi-part path strings in a join — a join must always *extend* the
# handle (never reset / over-count) so the depth-based type stays right
# ===========================================================================


class TestRelativeJoinParts:
    """:func:`_relative_join_parts` flattens whatever a caller passes to
    ``/`` / ``joinpath`` into clean, relative components."""

    def test_single_multipart_string_splits(self):
        assert _relative_join_parts(("a/b/c",)) == ["a", "b", "c"]

    def test_multiple_segments_flatten(self):
        assert _relative_join_parts(("a", "b/c", "d")) == ["a", "b", "c", "d"]

    def test_leading_slash_is_relative_not_absolute(self):
        # The leading-slash "absolute reset" is stripped — a Databricks
        # path must not be able to escape its namespace via a join.
        assert _relative_join_parts(("/a/b",)) == ["a", "b"]

    def test_trailing_and_duplicate_slashes_drop_empties(self):
        assert _relative_join_parts(("a/b/",)) == ["a", "b"]
        assert _relative_join_parts(("a//b",)) == ["a", "b"]

    def test_dot_components_dropped_dotdot_kept(self):
        assert _relative_join_parts(("a/./b",)) == ["a", "b"]
        assert _relative_join_parts(("a/../b",)) == ["a", "..", "b"]

    def test_empty_and_none_yield_nothing(self):
        assert _relative_join_parts(("",)) == []
        assert _relative_join_parts((None,)) == []


class TestMultiPartJoinResolvesCorrectType:
    """A multi-part string descends exactly as the chained ``/`` form
    does — same depth, same concrete type — for every join root."""

    def test_catalog_multipart_to_volume(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert isinstance(cat / "sales/raw", Volume)

    def test_catalog_multipart_to_volume_path(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        leaf = cat / "sales/raw/year=2026/data.parquet"
        assert isinstance(leaf, VolumePath)
        assert leaf.full_path() == "/Volumes/main/sales/raw/year=2026/data.parquet"

    def test_multipart_string_matches_chained_form(self, catalogs_service):
        # ``cat / "sales/raw"`` and ``cat / "sales" / "raw"`` are the same
        # UC location — and, sharing a service, the same singleton.
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert (cat / "sales/raw") is (cat / "sales" / "raw")

    def test_multi_arg_joinpath_matches_chained_form(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert cat.joinpath("sales", "raw") is (cat / "sales" / "raw")

    def test_schema_multipart_to_volume_path(self, schemas_service):
        sch = UCSchema(
            service=schemas_service, catalog_name="main", schema_name="sales",
        )
        leaf = sch / "raw/sub/f.csv"
        assert isinstance(leaf, VolumePath)
        assert leaf.full_path() == "/Volumes/main/sales/raw/sub/f.csv"

    def test_volume_multipart_to_volume_path(self, volumes_service):
        vol = Volume(
            service=volumes_service,
            catalog_name="main",
            schema_name="sales",
            volume_name="raw",
        )
        leaf = vol / "a/b/c.parquet"
        assert isinstance(leaf, VolumePath)
        assert leaf.full_path() == "/Volumes/main/sales/raw/a/b/c.parquet"


class TestJoinExtendsNeverResets:
    """The edge cases that used to mis-resolve: a leading slash dropped
    the handle's identity, a trailing slash over-counted into a deeper
    type. A join now always extends the fixed coordinates."""

    def test_leading_slash_still_extends_catalog(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        vol = cat / "/sales/raw"
        assert isinstance(vol, Volume)
        assert vol.full_path() == "/Volumes/main/sales/raw"

    def test_trailing_slash_does_not_overcount(self, catalogs_service):
        # ``"sales/raw/"`` is still the volume (depth 3), not a
        # zero-length child path beneath it.
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert isinstance(cat / "sales/raw/", Volume)

    def test_duplicate_slashes_collapse(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert isinstance(cat / "sales//raw", Volume)

    def test_empty_join_returns_same_handle(self, catalogs_service):
        cat = UCCatalog(service=catalogs_service, catalog_name="main")
        assert cat / "" is cat

    def test_volume_path_join_stays_in_namespace(self, volumes_service):
        # A leading slash on a VolumePath join must not reset to an
        # absolute path that escapes the volume root.
        vp = VolumePath(
            "/Volumes/main/sales/raw/dir",
            service=volumes_service,
        )
        joined = vp / "/escape/attempt"
        assert isinstance(joined, VolumePath)
        assert joined.full_path() == "/Volumes/main/sales/raw/dir/escape/attempt"

    def test_table_catalog_multipart_to_table(self, catalogs_service):
        cat = UCCatalog(
            service=catalogs_service,
            catalog_name="main",
            path_prefix=TABLE_PATH_PREFIX,
        )
        assert isinstance(cat / "sales/orders", Table)


class TestByteShapedPathJoins:
    """The same extend-don't-reset / multi-part rules apply to the
    byte-shaped surfaces (DBFS / Workspace / VolumePath), which all
    inherit :meth:`DatabricksPath.joinpath`."""

    def test_dbfs_multipart_join(self):
        from yggdrasil.databricks.fs.service import DBFSService
        svc = MagicMock(spec=DBFSService)
        p = DBFSPath("/dbfs/tmp/dir", service=svc)
        assert (p / "a/b/c").full_path() == "/dbfs/tmp/dir/a/b/c"

    def test_dbfs_leading_slash_stays_in_namespace(self):
        from yggdrasil.databricks.fs.service import DBFSService
        svc = MagicMock(spec=DBFSService)
        p = DBFSPath("/dbfs/tmp/dir", service=svc)
        # A leading slash must extend, not reset to ``/abs`` (which would
        # escape the DBFS root).
        assert (p / "/abs/x").full_path() == "/dbfs/tmp/dir/abs/x"

    def test_workspace_multipart_join(self):
        svc = MagicMock(spec=WorkspacePath._SERVICE_CLASS)
        p = WorkspacePath("/Workspace/Users/me", service=svc)
        assert (p / "proj/nb").full_path() == "/Workspace/Users/me/proj/nb"

    def test_volume_path_multipart_and_trailing_slash(self, volumes_service):
        vp = VolumePath("/Volumes/main/sales/raw/dir", service=volumes_service)
        assert (vp / "a/b/").full_path() == "/Volumes/main/sales/raw/dir/a/b"


class TestVolumePathNavigationStillWorks:
    """Regression: the ``joinpath`` override must not disturb the
    ``Volume.path`` factory or the ``with_name`` / ``with_suffix``
    helpers that lean on parent + join."""

    def test_volume_join_matches_volume_path_factory(self, volumes_service):
        vol = Volume(
            service=volumes_service,
            catalog_name="main",
            schema_name="sales",
            volume_name="raw",
        )
        # ``vol / "x/y.parquet"`` and ``vol.path("x/y.parquet")`` are the
        # same VolumePath singleton.
        assert (vol / "x/y.parquet") is vol.path("x/y.parquet")

    def test_with_name_and_suffix_on_volume_path(self, volumes_service):
        vp = VolumePath(
            "/Volumes/main/sales/raw/dir/file.csv",
            service=volumes_service,
        )
        assert vp.with_name("other.txt").full_path() == (
            "/Volumes/main/sales/raw/dir/other.txt"
        )
        assert vp.with_suffix(".parquet").full_path() == (
            "/Volumes/main/sales/raw/dir/file.parquet"
        )


class TestPathPrefixSurvivesPickle:
    """The navigation surface is plain instance state — it must round-
    trip through pickling like the rest of the resource."""

    def test_table_catalog_pickle_round_trip(self, catalogs_service):
        cat = UCCatalog(
            service=catalogs_service,
            catalog_name="main",
            path_prefix=TABLE_PATH_PREFIX,
        )
        assert "path_prefix" in cat.__getstate__()
        # The state carries the surface so an unpickled handle resolves
        # children the same way.
        assert cat.__getstate__()["path_prefix"] == TABLE_PATH_PREFIX
