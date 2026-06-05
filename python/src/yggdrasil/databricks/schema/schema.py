"""
Per-schema resource: lifecycle, table navigation, and tag helpers.

The :class:`Schema` dataclass wraps a single Unity Catalog schema and exposes
instance-level methods only.  Collection operations live in
:mod:`~yggdrasil.databricks.catalog.catalogs`.

Hierarchy navigation
--------------------
::

    schema["table_name"]  # → Table
    schema.table("orders")  # → Table
    schema.tables()         # → Iterator[Table]
    schema.catalog          # → Catalog (navigate up)

Tag handling
------------
Tag reads / writes / deletes route through ``client.entity_tags`` (entity
type ``"schemas"``).  The host-scoped cache in that service is
authoritative, so this class no longer carries its own tag cache. The
legacy ``set_tags_ddl`` helper is retained for dry-run / logging only —
``set_tags`` and ``unset_tags`` go through the REST API.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar, Iterable, Iterator, Mapping, Optional, TYPE_CHECKING

from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.catalog import (
    PermissionsChange,
    Privilege,
    PrivilegeAssignment,
    SchemaInfo,
    SecurableType,
)
from yggdrasil.concurrent.threading import Job
from yggdrasil.databricks.path import (
    DatabricksPath,
    TABLE_PATH_PREFIX,
    resolve_path_prefix,
)
from yggdrasil.databricks.sql.sql_utils import DEFAULT_TAG_COLLATION, databricks_tag_literal
from yggdrasil.dataclasses import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.enums import MediaTypes, MimeType, MimeTypes, Scheme
from yggdrasil.enums.mode import Mode, ModeLike
from yggdrasil.io.holder import IO
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.path import Path
from yggdrasil.url import URL

if TYPE_CHECKING:
    from yggdrasil.databricks.schema.schemas import Schemas
    from yggdrasil.databricks.catalog.catalog import UCCatalog
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.table.table import Table

__all__ = ["UCSchema"]

logger = logging.getLogger(__name__)


def _normalize_privileges(
    privileges: "str | Privilege | Iterable[str | Privilege] | None",
) -> Iterator[Privilege]:
    """Yield :class:`Privilege` enums for any caller-facing privilege spec.

    Accepts a single privilege or an iterable; strings are matched
    case-insensitively with ``-`` / spaces folded to ``_``
    (``"external use schema"`` → :attr:`Privilege.EXTERNAL_USE_SCHEMA`).
    Duplicates are deduped while preserving caller order. ``None`` and
    empty / whitespace-only items are skipped.

    Raises :class:`ValueError` on an unrecognized privilege name —
    the error message includes the list of valid privileges so a typo
    surfaces immediately.
    """
    if privileges is None:
        return
    if isinstance(privileges, (str, Privilege)):
        items: Iterable[Any] = (privileges,)
    else:
        items = privileges

    seen: set[Privilege] = set()
    for item in items:
        if item is None:
            continue
        if isinstance(item, Privilege):
            normalized = item
        else:
            token = str(item).strip()
            if not token:
                continue
            key = token.upper().replace("-", "_").replace(" ", "_")
            key = "_".join(p for p in key.split("_") if p)
            try:
                normalized = Privilege(key)
            except ValueError as exc:
                valid = ", ".join(p.value for p in Privilege)
                raise ValueError(
                    f"Unknown Unity Catalog privilege {token!r}. "
                    f"Pass a Privilege enum or one of: {valid}."
                ) from exc
        if normalized in seen:
            continue
        seen.add(normalized)
        yield normalized


class UCSchema(DatabricksPath):
    """A single Unity Catalog schema — lifecycle, table navigation, tags.

    Identity is ``(client, catalog_name, schema_name, path_prefix)``:
    two callers asking for the same schema *on the same navigation
    surface* under the same client collapse onto one instance via the
    :class:`Singleton` cache, so the cached :class:`SchemaInfo` and
    tag state are shared. :attr:`path_prefix` (inherited from the
    parent catalog) fixes what a ``/`` path-join descends into —
    ``/Volumes/`` → :class:`Volume` (then :class:`VolumePath`),
    ``/Tables/`` → :class:`Table` — so the child type is known up
    front rather than guessed.

    URL-addressable through :class:`DatabricksPath` under
    :attr:`Scheme.DATABRICKS_SCHEMA` (``dbfs+schema://``); the
    Path / Holder byte primitives raise — a schema is a logical
    UC resource, not a positional byte buffer. Mirrors the same
    ``(DatabricksPath, Singleton)`` shape that :class:`Catalog`
    uses.
    """

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_SCHEMA

    NAMESPACE_PREFIX: ClassVar[str] = "/Schemas/"
    _INSTANCES: ClassVar = Singleton._INSTANCES.__class__(default_ttl=None)
    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: "Schemas | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        *,
        path_prefix: str | None = None,
        url: URL | None = None,
        **_kwargs: Any,
    ) -> Any:
        # Key on the bound :class:`DatabricksClient` *instance*, not
        # on the host string — two clients with the same host but
        # different credentials are distinct identities and must own
        # distinct ``Schema`` instances. Mirrors :class:`Catalog`'s
        # convention.
        client = None
        try:
            client = service.client if service is not None else None
        except Exception:
            client = None
        # Resolve the catalog/schema names against the service
        # defaults the same way ``__init__`` will, so two calls that
        # differ only in "passed explicitly vs. inherited from the
        # service" land on the same singleton.
        if catalog_name is None and service is not None:
            catalog_name = getattr(service, "catalog_name", None)
        if schema_name is None and service is not None:
            schema_name = getattr(service, "schema_name", None)
        # ``path_prefix`` keeps a volume-surface and a table-surface view
        # of the same schema distinct — each resolves its children to a
        # different type. Resolved exactly as ``__init__`` will.
        return (
            cls,
            client,
            catalog_name,
            schema_name,
            resolve_path_prefix(path_prefix, url),
        )

    def __new__(
        cls,
        service: "Schemas | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        *,
        singleton_ttl: "int | None" = ...,
        **kwargs: Any,
    ):
        # Mirror :class:`Catalog`'s opt-in cache contract: per-call
        # ``singleton_ttl`` overrides ``_SINGLETON_TTL``; ``...`` on
        # both sides means "don't register" and every call allocates
        # a fresh instance. Cache lookup runs BEFORE the
        # :class:`DatabricksPath` construction chain so a hit skips
        # :class:`Holder` /:class:`Path` allocation entirely; the
        # ``object.__new__`` short-circuit keeps the MRO's
        # :class:`Singleton.__new__` from re-keying with empty args.
        if singleton_ttl is ...:
            singleton_ttl = cls._SINGLETON_TTL

        def _allocate() -> "UCSchema":
            return object.__new__(cls)

        if singleton_ttl is ...:
            return _allocate()

        key = cls._singleton_key(
            service,
            catalog_name=catalog_name,
            schema_name=schema_name,
            path_prefix=kwargs.get("path_prefix"),
            url=kwargs.get("url"),
        )
        ttl_arg = (
            float(singleton_ttl)
            if isinstance(singleton_ttl, int) and not isinstance(singleton_ttl, bool)
            else singleton_ttl
        )

        def _build() -> "UCSchema":
            inst = _allocate()
            try:
                object.__setattr__(inst, "_singleton_key_", key)
            except AttributeError:
                pass
            return inst

        return cls._INSTANCES.get_or_set(key, _build, ttl=ttl_arg)

    def __init__(
        self,
        data: Any = None,
        service: "Schemas | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        *,
        infos_ttl: float | None = None,
        infos: SchemaInfo | None = None,
        infos_fetched_at: float | None = None,
        url: URL | None = None,
        path_prefix: str | None = None,
        singleton_ttl: "int | None" = ...,
    ):
        # ``singleton_ttl`` is consumed by ``__new__``; accept it here
        # too so Python's auto-call after ``__new__`` doesn't trip on
        # an unexpected kwarg. ``data`` is only populated by the
        # :class:`DatabricksPath` dispatcher's post-``__new__`` auto-``__init__``
        # pass (the original positional path string); :meth:`from_url` already
        # built this schema, so it's discarded and the ``_initialized`` guard
        # no-ops the redundant pass.
        del singleton_ttl, data
        # Singleton-cached re-entry: a second ``Schema(service=…,
        # catalog_name=…, schema_name=…)`` call returns the live
        # instance via ``__new__``; skip the second pass so the
        # cached ``_infos`` / fetch timestamp don't get reset under
        # the caller.
        if getattr(self, "_initialized", False):
            return

        # Resolve the child-navigation surface from the *incoming* url
        # before it's rebuilt below into this schema's own
        # ``dbfs+schema://`` form. Lock-step with ``_singleton_key``.
        resolved_prefix = resolve_path_prefix(path_prefix, url)

        if service is None:
            from .schemas import Schemas
            service = Schemas.current()

        catalog_name = catalog_name or service.catalog_name
        schema_name = schema_name or service.schema_name

        if url is None:
            host = ""
            try:
                base_host = service.client.base_url.host if service is not None else ""
                host = base_host or ""
            except Exception:
                host = ""
            path_parts = [p for p in (catalog_name, schema_name) if p]
            url = URL(
                scheme=type(self).scheme.value,
                host=host,
                path="/" + "/".join(path_parts) if path_parts else "/",
            )

        super().__init__(url=url, service=service)
        self.service = service
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.path_prefix = resolved_prefix
        self._infos_ttl = infos_ttl
        self._infos = infos
        self._infos_fetched_at = infos_fetched_at
        # Tri-state cache of "does the current user hold EXTERNAL USE SCHEMA on
        # this schema?" — the UC prerequisite for direct cloud-storage access to
        # external securables under it. ``None`` until first checked, then
        # ``True`` / ``False``. Cached here because the schema is a singleton,
        # so one ``grants.get_effective`` answers for the whole process.
        self._can_use_external = None
        self._initialized = True

    # ── Path / Holder primitives — Schema is logical, not byte-shaped ─────────

    @property
    def is_remote_path(self) -> bool:
        return False

    @property
    def size(self) -> int:
        return 0

    @property
    def parent(self) -> "IO | None":
        return self.catalog

    @property
    def parents(self) -> "Iterator[IO]":
        yield self.catalog

    def full_path(self) -> str:
        return f"{self.NAMESPACE_PREFIX}{self.catalog_name}/{self.schema_name}"

    def _from_url(self, url: URL) -> "DatabricksPath":
        # ``url.parts`` is 0-indexed (leading ``/`` stripped), so this
        # schema's own URL (``/<cat>/<sch>``) is two parts. Path-join
        # navigation follows the volume-family depth model — catalog (1)
        # → schema (2) → volume (3) → :class:`VolumePath` (4+); the
        # logical ``schema["tbl"]`` → :class:`Table` surface stays on
        # ``__getitem__``.
        # Drop empty components (trailing / duplicate slashes) so the
        # depth count reflects real segments however the URL was built.
        parts = [p for p in url.parts if p]
        n = len(parts)

        if n <= 1:
            # ``/<catalog>`` — walked back up to the parent catalog.
            return self.catalog
        if n == 2:
            # ``/<catalog>/<schema>`` — this schema itself.
            return self
        # Depth ≥ 3 names a child of the schema. Its type is fixed by
        # this schema's navigation surface (:attr:`path_prefix`) — not
        # guessed: a table catalog descends into a :class:`Table`, a
        # volume catalog into a :class:`Volume` (then a VolumePath).
        if self.path_prefix == TABLE_PATH_PREFIX:
            if n == 3:
                from yggdrasil.databricks.table.tables import Tables

                return Tables(client=self.client).table(
                    catalog_name=self.catalog_name,
                    schema_name=self.schema_name,
                    table_name=parts[2],
                )
            raise ValueError(
                f"URL {url} descends {n} segments under a table catalog; "
                f"tables are leaves, not path-navigable containers. Use a "
                f"volume catalog (path_prefix='/Volumes/') for file paths."
            )
        # Volume surface — the join only ever extends this schema's own
        # coordinates, so anchor on them and let the volume own the
        # volume → VolumePath leg of the walk.
        from yggdrasil.databricks.volume.volumes import Volumes

        volume = Volumes(client=self.client).volume(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            volume_name=parts[2],
        )
        return volume if n == 3 else volume._from_url(url)

    def _stat_uncached(self) -> IOStats:
        infos = self.read_infos(default=None)
        kind = IOKind.MISSING if infos is None else IOKind.DIRECTORY

        return IOStats(
            kind=kind,
            media_type=MediaTypes.DATABRICKS_UNITY_CATALOG_SCHEMA,
        )

    def _read_mv(self, n: int, pos: int) -> memoryview:
        raise NotImplementedError(
            f"{type(self).__name__} is a logical Unity Catalog resource, "
            f"not a positional byte buffer. Navigate via "
            f"``schema['<table>']`` or ``schema.tables()`` instead."
        )

    def _write_mv(self, data: memoryview, pos: int) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} is a logical Unity Catalog resource. "
            f"Use ``create()`` / ``update()`` to mutate metadata."
        )

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["Path"]:
        for volume in self.service.volumes.list(catalog_name=self.catalog_name, schema_name=self.schema_name):
            if recursive:
                yield from volume.ls(recursive=recursive, singleton_ttl=singleton_ttl)
            else:
                yield volume

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        del parents
        self.get_or_create() if exist_ok else self.create(missing_ok=False)

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        self.delete(wait=wait, raise_error=not missing_ok)

    def _remove_dir(
        self,
        recursive: bool,
        missing_ok: bool,
        wait: WaitingConfig,
    ) -> None:
        self.delete(force=recursive, wait=wait, raise_error=not missing_ok)

    @classmethod
    def default_media_type(cls) -> MimeType:
        return MimeTypes.DATABRICKS_UNITY_CATALOG_SCHEMA

    @classmethod
    def from_url(cls, url: "URL | str", **kwargs: Any) -> "UCSchema":
        """Build a :class:`Schema` from a ``dbfs+volume:///cat/sch`` URL.

        Used by the :class:`DatabricksPath` dispatcher when a caller
        passes a POSIX volume path that resolves to schema depth
        (``DatabricksPath("/Volumes/main/sales")`` →
        ``Schema("main", "sales")``).
        """
        from yggdrasil.databricks.client import DatabricksClient
        from .schemas import Schemas

        u = URL.from_(url)
        parts = [p for p in (u.path or "/").lstrip("/").split("/") if p]
        if len(parts) < 2:
            raise ValueError(
                f"Cannot derive schema name from URL {u!r} — expected "
                f"two path segments (e.g. ``dbfs+volume:///main/sales``)."
            )
        catalog_name, schema_name = parts[0], parts[1]
        service = kwargs.pop("service", None)
        if service is None:
            client = (
                DatabricksClient(host=f"https://{u.host}/")
                if u.host else DatabricksClient.current()
            )
            service = Schemas(client=client)
        # Capture the child-navigation surface from the source scheme
        # before ``__init__`` rebuilds the URL into ``dbfs+schema://``.
        kwargs.setdefault("path_prefix", resolve_path_prefix(url=u))
        return cls(
            service=service,
            catalog_name=catalog_name,
            schema_name=schema_name,
            **kwargs,
        )

    # ── DatabricksResource compatibility ──────────────────────────────────────

    @property
    def client(self) -> "DatabricksClient":
        if self.service is not None:
            return self.service.client
        return super().client

    # ── identity ──────────────────────────────────────────────────────────────

    def full_name(self, safe: str | bool | None = None) -> str:
        """Return the two-part schema name, optionally backtick-quoted."""
        if safe:
            q = safe if isinstance(safe, str) else "`"
            return f"{q}{self.catalog_name}{q}.{q}{self.schema_name}{q}"
        return f"{self.catalog_name}.{self.schema_name}"

    def __str__(self) -> str:
        return self.full_name()

    # ── dict-like navigation ──────────────────────────────────────────────────

    def __getitem__(self, name: str) -> "Table":
        """``schema["table_name"]`` → :class:`Table`."""
        return self.table(name)

    def __setitem__(self, name: str, new_name: str) -> None:
        """``schema["old_table"] = "new_table"`` renames a child table."""
        self.table(name).rename(new_name)

    def __iter__(self) -> Iterator["Table"]:
        """Iterate over every table in this schema."""
        return self.tables()

    # ── URL ───────────────────────────────────────────────────────────────────

    @property
    def explore_url(self) -> URL:
        """Workspace UI URL pointing at this schema's Catalog Explorer page."""
        return self.client.base_url.with_path(
            f"/explore/data/{self.catalog_name}/{self.schema_name}"
        )

    # ── cache management ──────────────────────────────────────────────────────

    def _reset_cache(self, invalidate_cache: bool = False) -> None:
        """Evict the cached :class:`SchemaInfo`.

        ``invalidate_cache=True`` also drops this schema's tag list from
        ``client.entity_tags`` — used on structural changes (delete / rename)
        where the ``entity_name`` itself becomes stale.
        """
        if invalidate_cache:
            try:
                self.client.entity_tags.invalidate_cached_tags(
                    "schemas", self.full_name(),
                )
            except Exception:  # cache invalidation is best-effort
                pass
        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_infos_fetched_at", None)
        # A delete / rename / explicit clear can change the grant picture too.
        object.__setattr__(self, "_can_use_external", None)

    def clear(self) -> "UCSchema":
        """Public alias for :meth:`_reset_cache`; returns ``self``."""
        self._reset_cache()
        return self

    # ── infos / existence ─────────────────────────────────────────────────────

    @property
    def infos(self) -> SchemaInfo:
        return self.read_infos()

    def read_infos(self, default: Any = ...):
        now = time.time()

        if self._infos is not None:
            age = now - (self._infos_fetched_at or 0.0)
            if self._infos_ttl is None or age < self._infos_ttl:
                return self._infos
            logger.debug(
                "Cache expired for schema %r (age=%.0fs, ttl=%.0fs) — refreshing",
                self, age, self._infos_ttl,
            )

        logger.debug("Fetching schema info for %r from remote", self)
        try:
            infos = self.client.workspace_client().schemas.get(full_name=self.full_name())
        except Exception:
            if default is ...:
                raise

            logger.warning(f"Schema {self.full_name(safe=True)} not found", exc_info=True)
            return default

        logger.info("Fetched schema info for %r from remote", self)
        object.__setattr__(self, "_infos", infos)
        object.__setattr__(self, "_infos_fetched_at", now)
        return infos

    @property
    def schema_id(self):
        infos = self.read_infos(default=None)
        return infos.schema_id if infos is not None else None

    def exists(self) -> bool:
        """``True`` if this schema is reachable via the Unity Catalog API."""
        try:
            _ = self.infos
            return True
        except Exception:
            return False

    @property
    def comment(self) -> Optional[str]:
        return self.infos.comment

    @property
    def owner(self) -> Optional[str]:
        return self.infos.owner

    @property
    def storage_location(self) -> Optional[str]:
        return self.infos.storage_location

    @property
    def storage_path(self):
        l = self.storage_location

        if not l:
            return None

        return Path.from_(l)

    @property
    def storage_root(self) -> Optional[str]:
        return self.infos.storage_root

    # ── navigation ────────────────────────────────────────────────────────────

    @property
    def catalog(self) -> "UCCatalog":
        """Navigate up to the parent :class:`UCCatalog`.

        The parent inherits this schema's :attr:`path_prefix` so a
        round-trip up-then-down (``schema.catalog / schema_name``) lands
        back on the same surface — and the same singleton.
        """
        from yggdrasil.databricks.catalog.catalog import UCCatalog as _Catalog
        return _Catalog(
            service=self.service,
            catalog_name=self.catalog_name,
            path_prefix=self.path_prefix,
        )

    def table(self, name: str) -> "Table":
        """Return a :class:`Table` within this schema.

        Args:
            name: Table name (unqualified).
        """
        return self.client.tables.table(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=name,
        )

    def tables(self, name: str | None = None) -> Iterator["Table"]:
        """Iterate over tables in this schema, optionally filtered by name."""
        return self.client.tables.list_tables(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            name=name,
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def create(
        self,
        *,
        comment: str | None = None,
        properties: Optional[Mapping[str, str]] = None,
        storage_root: str | None = None,
        missing_ok: bool = True,
    ) -> "UCSchema":
        """Create this schema in Unity Catalog.

        Args:
            comment:      Human-readable description.
            properties:   Extra key/value properties.
            storage_root: External storage root URI.
            missing_ok: Silently succeed if the schema already exists.
        """
        # Idempotent: a successful read means it already exists — never
        # auto-create from a read, only here.
        if self.read_infos(default=None) is None:
            uc = self.client.workspace_client().schemas
            logger.debug(
                "Creating schema %r (storage_root=%s, missing_ok=%s)",
                self, storage_root, missing_ok,
            )
            kwargs = dict(
                catalog_name=self.catalog_name,
                name=self.schema_name,
                comment=comment,
                properties=properties,
                storage_root=storage_root,
            )
            try:
                info = uc.create(**kwargs)
                object.__setattr__(self, "_infos", info)
                object.__setattr__(self, "_infos_fetched_at", time.time())
            except Exception as exc:
                low = str(exc).lower()
                if missing_ok and "already exists" in low:
                    logger.debug(
                        "Schema %r already exists — soft-resetting cache", self,
                    )
                    self._reset_cache()
                elif "not exist" in low or "not found" in low:
                    # Parent catalog missing — create it and retry once.
                    logger.info("Schema %r create failed (%s); ensuring parent catalog", self, exc)
                    self.catalog.get_or_create()
                    info = uc.create(**kwargs)
                    object.__setattr__(self, "_infos", info)
                    object.__setattr__(self, "_infos_fetched_at", time.time())
                else:
                    raise
        # Keep the path stat cache in lock-step with the now-current info so a
        # follow-up exists() / is_dir() / stat() doesn't observe a stale MISSING.
        self._persist_stat_cache(self._stat_uncached())
        return self

    def get_or_create(
        self,
        *,
        comment: str | None = None,
        properties: Optional[Mapping[str, str]] = None,
        storage_root: str | None = None,
    ) -> "UCSchema":
        """Create this schema (and any missing parent catalog) if it doesn't
        exist, then return ``self``. :meth:`create` is itself idempotent and
        ensures the parent on a not-found, so this is just a named alias."""
        return self.create(
            comment=comment,
            properties=properties,
            storage_root=storage_root,
            missing_ok=True,
        )

    def delete(
        self,
        *,
        force: bool = False,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "UCSchema":
        """Delete this schema from Unity Catalog.

        Args:
            force:       Cascade-delete all child tables.
            wait:        Block until the API call returns.
            raise_error: Re-raise :exc:`DatabricksError` on failure.
        """
        uc = self.client.workspace_client().schemas
        logger.debug(
            "Deleting schema %r (force=%s, wait=%s)", self, force, bool(wait),
        )
        if wait:
            try:
                uc.delete(full_name=self.full_name(), force=force)
            except DatabricksError:
                if raise_error:
                    raise
        else:
            Job.make(uc.delete, self.full_name()).fire_and_forget()

        # Structural change — drop both _infos and the entity-tag cache.
        self._reset_cache(invalidate_cache=True)
        return self

    # ── clone ───────────────────────────────────────────────────────────────────

    def clone(
        self,
        target: "str | UCSchema | None" = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        deep: bool = True,
        mode: ModeLike = Mode.IGNORE,
        include_views: bool = True,
        name: str | None = None,
        max_workers: int | None = None,
    ) -> dict[str, str]:
        """Clone every child of this schema into *target*, in parallel.

        The target schema (and any missing parent catalog) is created first,
        then each child table / view is cloned concurrently. ``mode`` is the
        single existence policy, applied uniformly to every sub-clone:

        - ``IGNORE`` (default) — ``CREATE … IF NOT EXISTS``: a child already
          present is **skipped** (left untouched), a missing one is created.
        - ``OVERWRITE`` / ``TRUNCATE`` — ``CREATE OR REPLACE``: overwrite
          same-named targets.
        - ``ERROR_IF_EXISTS`` — plain ``CREATE``: record a failure per clash.

        A target that exists but **changed kind** (table ⇄ view) can't be
        cross-replaced by Delta, so it is dropped first and recreated as the
        source's current kind (except under ``ERROR_IF_EXISTS``, which lets the
        clash surface as a failure). One child's failure is recorded and never
        aborts the rest of the batch.

        Args:
            target:        Destination schema — a :class:`UCSchema`, a
                           ``"catalog.schema"`` / ``"schema"`` dotted name, or
                           ``None`` when *catalog_name* / *schema_name* are
                           passed explicitly. A bare name reuses this schema's
                           catalog.
            catalog_name:  Target catalog override (defaults to this schema's).
            schema_name:   Target schema override.
            deep:          DEEP clone (independent copy) vs SHALLOW (metadata
                           only, shares the source's files).
            mode:          Existence policy (``Mode`` or mode-like string)
                           forwarded to every sub-clone — see above. Defaults to
                           ``IGNORE`` (skip what's already there).
            include_views: Also clone view-shaped children (re-emitting their
                           definition); ``False`` clones only tables.
            name:          Optional child-name filter (exact or glob) — clone a
                           subset.
            max_workers:   Thread-pool size for the fan-out (defaults to the
                           child count, capped at 16).

        Returns:
            ``{table_name: status}`` where status is ``"created"``,
            ``"skipped"`` (already present), or ``"failed: <error>"``.
        """
        import concurrent.futures as cf

        # One existence policy drives the whole fan-out, forwarded to every
        # sub-clone so the batch is uniform.
        sub_mode = Mode.from_(mode)
        if sub_mode not in (
            Mode.OVERWRITE, Mode.TRUNCATE, Mode.IGNORE, Mode.ERROR_IF_EXISTS,
        ):
            raise ValueError(
                f"clone mode must be OVERWRITE/TRUNCATE, IGNORE, or "
                f"ERROR_IF_EXISTS — got {sub_mode.name}."
            )
        skip_existing = sub_mode is Mode.IGNORE
        # ERROR_IF_EXISTS wants a clash to fail; every other mode means
        # "(re)create over it", which is what licenses dropping a kind-drifted
        # target before recreating it.
        recreate_on_drift = sub_mode is not Mode.ERROR_IF_EXISTS

        # Resolve the destination catalog / schema from whichever form the
        # caller passed (UCSchema, dotted string, or explicit kwargs).
        if isinstance(target, UCSchema):
            target_catalog, target_schema = target.catalog_name, target.schema_name
        else:
            parsed_catalog = parsed_schema = None
            if target:
                parts = [p.strip().strip("`") for p in str(target).split(".") if p.strip()]
                if len(parts) == 1:
                    parsed_schema = parts[0]
                elif len(parts) == 2:
                    parsed_catalog, parsed_schema = parts
                else:
                    raise ValueError(
                        f"clone target {target!r} must be a 'schema' or "
                        f"'catalog.schema' name."
                    )
            target_catalog = catalog_name or parsed_catalog or self.catalog_name
            target_schema = schema_name or parsed_schema
        if not target_schema:
            raise ValueError(
                "clone needs a target schema — pass target='catalog.schema' / "
                "'schema', a UCSchema, or schema_name=."
            )
        if target_catalog == self.catalog_name and target_schema == self.schema_name:
            raise ValueError(
                f"Cannot clone {self.full_name()} onto itself — choose a "
                f"different target catalog/schema."
            )

        tgt = (
            target
            if isinstance(target, UCSchema)
            else type(self)(
                service=self.service,
                catalog_name=target_catalog,
                schema_name=target_schema,
                path_prefix=self.path_prefix,
            )
        )
        tgt.get_or_create(comment=f"clone of {self.full_name()}")

        # ``list_tables`` pre-stores each child's SchemaInfo, so ``is_view`` is
        # free here — no extra round-trip to filter views out.
        children = [
            child for child in self.tables(name=name)
            if include_views or not child.is_view
        ]
        if not children:
            logger.info("clone %s → %s: no children to copy", self, tgt)
            return {}

        def _clone_one(src: "Table") -> tuple[str, str]:
            dst = tgt.table(src.table_name)
            try:
                if dst.exists():
                    # ``exists()`` populated dst's infos, so ``is_view`` is free.
                    kind_changed = bool(src.is_view) != bool(dst.is_view)
                    if not kind_changed and skip_existing:
                        return src.table_name, "skipped"
                    if kind_changed and recreate_on_drift:
                        # Neither direction can be cross-replaced — a view can't
                        # ``CREATE OR REPLACE TABLE`` and a table can't
                        # ``CREATE OR REPLACE VIEW`` — so drop the stale target
                        # (table→view *and* view→table) and recreate it as the
                        # source's current kind. The UC tables API drops views
                        # too, so one ``delete`` covers both.
                        logger.info(
                            "clone %s → %s: target changed kind (view=%s→%s) — "
                            "dropping before recreate",
                            src.full_name(), dst.full_name(),
                            dst.is_view, src.is_view,
                        )
                        dst.delete(missing_ok=True)
                src.clone(target=dst, deep=deep, mode=sub_mode)
                return src.table_name, "created"
            except Exception as exc:  # noqa: BLE001 — collect, don't abort the batch
                logger.warning(
                    "clone %s → %s failed: %s", src.full_name(), dst.full_name(), exc,
                )
                return src.table_name, f"failed: {exc}"

        workers = max_workers or min(len(children), 16)
        results: dict[str, str] = {}
        with cf.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ygg-schema-clone",
        ) as pool:
            for table_name, status in pool.map(_clone_one, children):
                results[table_name] = status

        created = sum(s == "created" for s in results.values())
        skipped = sum(s == "skipped" for s in results.values())
        failed = sum(s.startswith("failed") for s in results.values())
        logger.info(
            "cloned %s → %s: %d created, %d skipped, %d failed",
            self.full_name(), tgt.full_name(), created, skipped, failed,
        )
        return results

    # ── tags ──────────────────────────────────────────────────────────────────

    @property
    def tags(self) -> tuple[Any, ...]:
        """Schema-level entity-tag assignments — served from ``client.entity_tags``."""
        return tuple(
            self.client.entity_tags.entity_tags(
                "schemas", self.full_name(), default=()
            ) or ()
        )

    def set_tags_ddl(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
    ) -> str:
        """Build an ``ALTER SCHEMA … SET TAGS`` DDL statement.

        Retained for dry-run / logging contexts; :meth:`set_tags` no longer
        executes this DDL — it goes through the ``entity_tag_assignments``
        REST API instead.
        """
        pairs: list[str] = []
        for k, v in (tags or {}).items():
            key = str(k).strip() if k is not None else ""
            val = str(v).strip() if v is not None else ""
            if key and val:
                pairs.append(
                    f"{databricks_tag_literal(key, collation=tag_collation)} = "
                    f"{databricks_tag_literal(val, collation=tag_collation)}"
                )
        if not pairs:
            raise ValueError(f"Cannot set empty tags on {self!r}")
        return (
            f"ALTER SCHEMA {self.full_name(safe=True)} "
            f"SET TAGS ({', '.join(pairs)})"
        )

    def set_tags(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
        mode: ModeLike | None = None,
    ) -> "UCSchema":
        """Apply schema-level tags via the UC ``entity_tag_assignments`` API.

        ``tag_collation`` is accepted for API compatibility and ignored —
        collations only matter for the legacy DDL literal form.

        ``mode`` selects the write strategy (``"upsert"`` default, ``"overwrite"``
        for strict replace, ``"append"`` / ``"ignore"`` / ``"error_if_exists"``).
        """
        del tag_collation
        if not tags:
            return self

        self.client.entity_tags.update_entity_tags(
            tags=tags,
            entity_type="schemas",
            entity_name=self.full_name(),
            mode=mode,
        )
        return self

    def unset_tags(
        self,
        tag_keys: Iterable[str],
        *,
        if_exists: bool = True,
    ) -> "UCSchema":
        """Delete schema-level tag assignments by key."""
        self.client.entity_tags.delete_entity_tags(
            entity_type="schemas",
            entity_name=self.full_name(),
            tag_keys=tag_keys,
            if_exists=if_exists,
        )
        return self

    # ── grants ────────────────────────────────────────────────────────────────

    def _grants_securable_type(self) -> SecurableType:
        return SecurableType.SCHEMA

    def _grants_full_name(self) -> str:
        return self.full_name()

    # ── permissions (CRUD) ────────────────────────────────────────────────────

    def permissions(
        self,
        *,
        principal: str | None = None,
    ) -> tuple[PrivilegeAssignment, ...]:
        """Direct grants on this schema (no inherited privileges).

        Calls the Unity Catalog ``grants.get`` endpoint.

        Args:
            principal: Optional filter — return only grants for this
                user / group / service principal.

        Returns:
            Tuple of :class:`PrivilegeAssignment` (one per principal
            with at least one direct grant).
        """
        kwargs: dict[str, Any] = {}
        if principal is not None:
            kwargs["principal"] = principal
        response = self.client.workspace_client().grants.get(
            securable_type=SecurableType.SCHEMA.value,
            full_name=self.full_name(),
            **kwargs,
        )
        return tuple(response.privilege_assignments or ())

    def effective_permissions(
        self,
        *,
        principal: str | None = None,
    ) -> tuple[Any, ...]:
        """Effective grants on this schema, including privileges inherited
        from the parent catalog / metastore.

        Calls the Unity Catalog ``grants.get_effective`` endpoint.
        """
        kwargs: dict[str, Any] = {}
        if principal is not None:
            kwargs["principal"] = principal
        response = self.client.workspace_client().grants.get_effective(
            securable_type=SecurableType.SCHEMA.value,
            full_name=self.full_name(),
            **kwargs,
        )
        return tuple(response.privilege_assignments or ())

    def can_use_external(self, *, refresh: bool = False) -> bool:
        """True when the current user holds ``EXTERNAL USE SCHEMA`` on this
        schema (directly or inherited from the catalog / metastore).

        That privilege is Unity Catalog's prerequisite for touching an external
        securable's backing cloud storage directly — so callers gate a direct
        storage-path read / write on it and otherwise fall back to the Files
        API. The result is cached on this (singleton) schema, so a single
        ``grants.get_effective`` decides for the whole process; ``refresh``
        forces a re-check. Any lookup failure (no current-user, denied grants
        read, …) resolves to ``False`` without raising — the safe default is
        "go through the Files API".
        """
        cached = self._can_use_external
        if cached is not None and not refresh:
            return cached

        granted = False
        try:
            current = self.client.iam.users.current_user
            principal = (
                getattr(current, "email", None)
                or getattr(current, "username", None)
                or getattr(current, "name", None)
            )
            if principal:
                for assignment in self.effective_permissions(principal=principal):
                    for p in (assignment.privileges or ()):
                        # ``grants.get_effective`` returns ``EffectivePrivilege``
                        # wrappers (the enum lives on ``.privilege``); a plain
                        # ``grants.get`` hands back ``Privilege`` enums directly.
                        # Handle both, else an inherited grant never matches.
                        priv = getattr(p, "privilege", p)
                        if priv is Privilege.EXTERNAL_USE_SCHEMA or (
                            getattr(priv, "value", str(priv))
                            == Privilege.EXTERNAL_USE_SCHEMA.value
                        ):
                            granted = True
                            break
                    if granted:
                        break
        except Exception as exc:  # no current-user / denied grants read / …
            logger.debug(
                "EXTERNAL USE SCHEMA check failed for %r (%s) — assuming no",
                self, exc,
            )
            granted = False

        object.__setattr__(self, "_can_use_external", granted)
        return granted

    def mark_external_unusable(self) -> None:
        """Record that direct external-storage access didn't actually work for
        this schema (e.g. the grant is present but the bucket policy denied the
        I/O) so callers stop re-trying it and route through the Files API."""
        object.__setattr__(self, "_can_use_external", False)

    def grant(
        self,
        principal: str,
        privileges: "str | Privilege | Iterable[str | Privilege]",
    ) -> "UCSchema":
        """Add one or more privileges for *principal* on this schema.

        Privileges may be passed as :class:`Privilege` enums or as
        strings (case-insensitive, ``-`` / spaces accepted in place of
        ``_``).  Example::

            schema.grant("alice@example.com", "EXTERNAL USE SCHEMA")
            schema.grant("data-engs", [Privilege.USE_SCHEMA, "SELECT"])
        """
        return self.update_permissions(
            changes=[PermissionsChange(
                principal=principal,
                add=list(_normalize_privileges(privileges)),
            )]
        )

    def revoke(
        self,
        principal: str,
        privileges: "str | Privilege | Iterable[str | Privilege]",
    ) -> "UCSchema":
        """Remove one or more privileges for *principal* on this schema."""
        return self.update_permissions(
            changes=[PermissionsChange(
                principal=principal,
                remove=list(_normalize_privileges(privileges)),
            )]
        )

    def set_permissions(
        self,
        principal: str,
        privileges: "str | Privilege | Iterable[str | Privilege]",
    ) -> "UCSchema":
        """Replace *principal*'s direct grants on this schema with
        exactly *privileges*.

        Computes the diff against the current direct grants and emits a
        single ``grants.update`` call that adds the missing privileges
        and removes the extras.  Inherited grants are not touched (they
        belong to the parent securable).
        """
        desired = set(_normalize_privileges(privileges))
        current: set[Privilege] = set()
        for assignment in self.permissions(principal=principal):
            for p in (assignment.privileges or ()):
                current.add(p if isinstance(p, Privilege) else Privilege(p))

        add = desired - current
        remove = current - desired
        if not add and not remove:
            return self

        return self.update_permissions(
            changes=[PermissionsChange(
                principal=principal,
                add=sorted(add, key=lambda p: p.value) or None,
                remove=sorted(remove, key=lambda p: p.value) or None,
            )]
        )

    def update_permissions(
        self,
        changes: "Iterable[PermissionsChange | Mapping[str, Any]]",
    ) -> "UCSchema":
        """Apply a batch of ``PermissionsChange`` to this schema.

        Accepts :class:`PermissionsChange` instances or plain mappings
        (``{"principal": ..., "add": [...], "remove": [...]}``).  Empty
        / no-op changes are filtered out before the API call.
        """
        normalized: list[PermissionsChange] = []
        for change in changes or ():
            if isinstance(change, PermissionsChange):
                pc = change
            elif isinstance(change, Mapping):
                pc = PermissionsChange(
                    principal=change.get("principal"),
                    add=list(_normalize_privileges(change.get("add"))) or None,
                    remove=list(_normalize_privileges(change.get("remove"))) or None,
                )
            else:
                raise TypeError(
                    f"Schema.update_permissions: each change must be a "
                    f"PermissionsChange or mapping, got {type(change).__name__}: "
                    f"{change!r}."
                )
            if not pc.principal:
                raise ValueError(
                    f"Schema.update_permissions: change is missing 'principal': {pc!r}."
                )
            if not pc.add and not pc.remove:
                continue
            normalized.append(pc)

        if not normalized:
            return self

        self.client.workspace_client().grants.update(
            securable_type=SecurableType.SCHEMA.value,
            full_name=self.full_name(),
            changes=normalized,
        )
        return self

    # ── update ────────────────────────────────────────────────────────────────

    def update(
        self,
        *,
        comment: str | None = None,
        owner: str | None = None,
        properties: Optional[Mapping[str, str]] = None,
    ) -> "UCSchema":
        """Update schema metadata in-place and refresh the local cache."""
        kwargs: dict[str, Any] = {}
        if comment is not None:
            kwargs["comment"] = comment
        if owner is not None:
            kwargs["owner"] = owner
        if properties is not None:
            kwargs["properties"] = properties

        logger.debug(
            "Updating schema %r (fields=%s)", self, sorted(kwargs.keys()),
        )
        info = self.client.workspace_client().schemas.update(
            full_name=self.full_name(), **kwargs
        )
        object.__setattr__(self, "_infos", info)
        object.__setattr__(self, "_infos_fetched_at", time.time())
        return self

    # ── rename ────────────────────────────────────────────────────────────────

    def rename(self, new_name: str) -> "UCSchema":
        """Rename this schema in-place (``ALTER SCHEMA … RENAME TO …``).

        The catalog parent is unchanged; *new_name* is the unqualified schema name.
        """
        new_name = (new_name or "").strip().strip("`")
        if not new_name:
            raise ValueError("Cannot rename schema to an empty name")
        if new_name == self.schema_name:
            logger.debug(
                "Skipping rename of schema %r — new name matches current", self,
            )
            return self

        logger.debug(
            "Renaming schema %r → %s.%s", self, self.catalog_name, new_name,
        )

        # Drop the old entity-tag cache key before the rename — the
        # ``entity_name`` is the two-part full name, and after the rename
        # it's dead.
        try:
            self.client.entity_tags.invalidate_cached_tags(
                "schemas", self.full_name(),
            )
        except Exception:
            pass

        info = self.client.workspace_client().schemas.update(
            full_name=self.full_name(), new_name=new_name,
        )
        self.schema_name = new_name
        object.__setattr__(self, "_infos", info)
        object.__setattr__(self, "_infos_fetched_at", time.time())
        return self


# Backwards-compat alias so existing ``from … import Schema`` keeps working.
Schema = UCSchema