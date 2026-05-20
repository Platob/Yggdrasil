"""
Per-catalog resource: lifecycle, navigation, and tag helpers.

The :class:`Catalog` dataclass wraps a single Unity Catalog catalog and
exposes instance-level methods only.  Collection operations live in
:mod:`~yggdrasil.databricks.catalog.catalogs`.

Hierarchy navigation
--------------------
::

    catalog["schema_name"]           # → Schema
    catalog["schema_name"]["table"]  # → Table
    catalog.schema("sales")          # → Schema
    catalog.schemas()                # → Iterator[Schema]
    catalog.table("sales.orders")    # → Table

Tag handling
------------
Tag reads / writes / deletes route through ``client.entity_tags`` (entity
type ``"catalogs"``).  The host-scoped cache in that service is
authoritative, so this class no longer carries its own tag cache. The
legacy ``set_tags_ddl`` helper is retained for dry-run / logging only —
``set_tags`` and ``unset_tags`` go through the REST API.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar, Iterable, Iterator, Mapping, Optional, TYPE_CHECKING

from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import CatalogInfo, SecurableType
from yggdrasil.concurrent.threading import Job
from yggdrasil.data.enums import MediaTypes, MimeType, MimeTypes, Scheme
from yggdrasil.dataclasses import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.databricks.path import DatabricksPath
from yggdrasil.io import URL
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.io.path import Path
from yggdrasil.data.enums.mode import Mode, ModeLike

from yggdrasil.databricks.sql.sql_utils import DEFAULT_TAG_COLLATION, databricks_tag_literal, quote_ident

if TYPE_CHECKING:
    from yggdrasil.databricks.catalog.catalogs import Catalogs
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.schema.schema import Schema
    from yggdrasil.databricks.table.table import Table

__all__ = ["Catalog"]

logger = logging.getLogger(__name__)


class Catalog(DatabricksPath, Singleton):
    """A single Unity Catalog catalog — lifecycle, schema navigation, tags.

    Identity is ``(client, catalog_name)``: two callers asking for
    the same catalog under the same client collapse onto one
    instance via the :class:`Singleton` cache, so the cached
    :class:`CatalogInfo` and tag state are shared.

    URL-addressable through :class:`DatabricksPath` under
    :attr:`Scheme.DATABRICKS_CATALOG` (``dbfs+catalog://``); the
    Path / Holder byte primitives raise — a catalog is a logical
    UC resource, not a positional byte buffer.
    """

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_CATALOG

    # Per-class singleton cache so this surface stays separate from
    # the rest of the project's :class:`Singleton` users.
    _INSTANCES: ClassVar = Singleton._INSTANCES.__class__(default_ttl=None)
    # Cache every catalog under the singleton convention; the cached
    # ``CatalogInfo`` and tag state are worth keeping for the
    # process lifetime so navigation through ``catalogs[name]`` /
    # ``schemas`` / ``tables`` doesn't keep refetching.
    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: "Catalogs | None" = None,
        *,
        catalog_name: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        # Key on the bound :class:`DatabricksClient` *instance*, not on
        # the host string — two clients with the same host but
        # different credentials (different PAT, different OAuth
        # secrets, …) are distinct identities and must own distinct
        # ``Catalog`` instances. ``DatabricksClient`` is itself a
        # :class:`Singleton`, so reusing the same client gives a
        # stable, hashable identity.
        client = None
        try:
            client = service.client if service is not None else None
        except Exception:
            client = None
        return (cls, client, catalog_name)

    def __new__(
        cls,
        service: "Catalogs | None" = None,
        *,
        catalog_name: str | None = None,
        singleton_ttl: "int | None" = ...,
        **kwargs: Any,
    ):
        # Mirror :class:`Singleton.__new__`'s opt-in cache contract:
        # a per-call ``singleton_ttl`` overrides the class default
        # (:attr:`_SINGLETON_TTL`); ``...`` on both sides means
        # "don't register" and every call allocates a fresh
        # instance. Cache lookup runs BEFORE the
        # :class:`DatabricksPath` construction chain so a hit skips
        # :class:`Holder` /:class:`Path` allocation entirely.
        if singleton_ttl is ...:
            singleton_ttl = cls._SINGLETON_TTL

        # Allocate via ``object.__new__`` directly: ``Catalog.__init__``
        # builds the canonical ``dbfs+catalog://`` URL itself, so the
        # :class:`DatabricksPath` /:class:`Holder` ``__new__`` chain
        # has nothing useful to add — and chaining through it would
        # also re-enter :class:`Singleton.__new__` from the MRO,
        # collapsing every allocation onto the empty-key singleton.
        def _allocate() -> "Catalog":
            return object.__new__(cls)

        if singleton_ttl is ...:
            return _allocate()

        key = cls._singleton_key(service, catalog_name=catalog_name)
        with cls._INSTANCES_LOCK:
            existing = cls._INSTANCES.get(key)
            if existing is not None:
                return existing
            instance = _allocate()
            try:
                object.__setattr__(instance, "_singleton_key_", key)
            except AttributeError:
                pass
            ttl_arg = (
                float(singleton_ttl)
                if isinstance(singleton_ttl, int) and not isinstance(singleton_ttl, bool)
                else singleton_ttl
            )
            cls._INSTANCES.set(key, instance, ttl=ttl_arg)
            return instance

    def __init__(
        self,
        service: "Catalogs | None" = None,
        *,
        catalog_name: str | None = None,
        infos_ttl: float | None = None,
        infos: CatalogInfo | None = None,
        infos_fetched_at: float | None = None,
        url: URL | None = None,
        singleton_ttl: "int | None" = ...,
    ):
        # ``singleton_ttl`` is consumed by ``__new__``; accept it here
        # too so Python's auto-call after ``__new__`` doesn't trip on
        # an unexpected kwarg.
        del singleton_ttl
        # Singleton-cached re-entry: a second ``Catalog(service=…,
        # catalog_name=…)`` call on the same key returns the live
        # instance via ``__new__``; skip the second pass so the cached
        # ``_infos`` / fetch timestamp don't get reset under the caller.
        if getattr(self, "_initialized", False):
            return

        if service is None:
            from .catalogs import Catalogs
            service = Catalogs.current()

        if url is None:
            host = ""
            try:
                base_host = service.client.base_url.host if service is not None else ""
                host = base_host or ""
            except Exception:
                host = ""
            url = URL(
                scheme=type(self).scheme.value,
                host=host,
                path=f"/{catalog_name}" if catalog_name else "/",
            )

        super().__init__(url=url, service=service)
        self.service = service
        self.catalog_name = catalog_name
        self._infos_ttl = infos_ttl or 1800.0
        self._infos = infos
        self._infos_fetched_at = infos_fetched_at
        self._initialized = True

    # ── Path / Holder primitives — Catalog is logical, not byte-shaped ────────

    @property
    def is_remote_path(self) -> bool:
        # The catalog identity lives in Unity Catalog, not at a
        # backend file URL — mirror :class:`Table`.
        return False

    @property
    def size(self) -> int:
        return 0

    def full_path(self) -> str:
        return self.full_name()

    def _stat(self) -> IOStats:
        return self._stat_uncached()

    def _stat_uncached(self) -> IOStats:
        return IOStats(
            size=0,
            mtime=0.0,
            kind=IOKind.DIRECTORY if self.exists else IOKind.MISSING,
            media_type=MediaTypes.DATABRICKS_UNITY_CATALOG_CATALOG,
        )

    def _read_mv(self, n: int, pos: int) -> memoryview:
        raise NotImplementedError(
            f"{type(self).__name__} is a logical Unity Catalog resource, "
            f"not a positional byte buffer. Navigate via "
            f"``catalog['<schema>']`` or ``catalog.schemas()`` instead."
        )

    def _write_mv(self, data: memoryview, pos: int) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} is a logical Unity Catalog resource. "
            f"Use ``create()`` / ``update()`` to mutate metadata."
        )

    def _bread(self, n: int, pos: int, mode: Mode) -> BytesIO:
        raise NotImplementedError(
            f"{type(self).__name__} is a logical Unity Catalog resource."
        )

    def _bwrite(self, data: BytesIO, pos: int, mode: Mode) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} is a logical Unity Catalog resource."
        )

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["Path"]:
        del recursive, singleton_ttl
        return iter(())

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        del parents
        self.ensure_created() if exist_ok else self.create(if_not_exists=False)

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
        return MimeTypes.DATABRICKS_UNITY_CATALOG_CATALOG

    @classmethod
    def from_url(cls, url: "URL | str", **kwargs: Any) -> "Catalog":
        """Build a :class:`Catalog` from a ``dbfs+volume:///cat`` /
        ``dbfs+catalog:///cat`` URL.

        Used by the :class:`DatabricksPath` dispatcher when a caller
        passes a POSIX volume path that resolves to catalog depth
        (``DatabricksPath("/Volumes/main")`` → ``Catalog("main")``).
        Pulls the catalog name from the first path segment and binds
        the underlying :class:`DatabricksClient` to whatever the URL's
        host resolves to (or :meth:`DatabricksClient.current` when
        the URL carries no host).
        """
        from yggdrasil.databricks.client import DatabricksClient
        from .catalogs import Catalogs

        u = URL.from_(url)
        parts = [p for p in (u.path or "/").lstrip("/").split("/") if p]
        if not parts:
            raise ValueError(
                f"Cannot derive catalog name from URL {u!r} — expected at "
                f"least one path segment (e.g. ``dbfs+volume:///main``)."
            )
        catalog_name = parts[0]
        client = kwargs.pop("client", None)
        if client is None:
            client = (
                DatabricksClient(host=f"https://{u.host}/")
                if u.host else DatabricksClient.current()
            )
        service = kwargs.pop("service", None) or Catalogs(client=client)
        return cls(service=service, catalog_name=catalog_name, **kwargs)

    # ── DatabricksResource compatibility ──────────────────────────────────────

    @property
    def client(self) -> "DatabricksClient":
        # Prefer the service's client (the resource's authoritative
        # binding); fall back to the path-level client when the
        # service was explicitly cleared.
        if self.service is not None:
            return self.service.client
        return super().client

    # ── identity ──────────────────────────────────────────────────────────────

    def full_name(self, safe: bool = None) -> str:
        """Return the catalog name (single-part identifier)."""
        return quote_ident(self.catalog_name) if safe else self.catalog_name

    def __str__(self) -> str:
        return self.catalog_name or ""

    # ── dict-like navigation ──────────────────────────────────────────────────

    def __getitem__(self, name: str) -> "Schema":
        """``catalog["schema_name"]`` → :class:`Schema`."""
        return self.schema(name)

    def __setitem__(self, name: str, new_name: str) -> None:
        """``catalog["old_schema"] = "new_schema"`` renames a child schema."""
        self.schema(name).rename(new_name)

    def __iter__(self) -> Iterator["Schema"]:
        """Iterate over every schema in this catalog."""
        return self.schemas()

    # ── URL ───────────────────────────────────────────────────────────────────

    @property
    def explore_url(self) -> URL:
        """Workspace UI URL pointing at this catalog's Catalog Explorer page."""
        return self.client.base_url.with_path(f"/explore/data/{self.catalog_name}")

    # ── cache management ──────────────────────────────────────────────────────

    def _reset_cache(self, invalidate_cache: bool = False) -> None:
        """Evict the cached :class:`CatalogInfo`.

        ``invalidate_cache=True`` also drops this catalog's tag list from
        ``client.entity_tags`` — used on structural changes (delete / rename)
        where the ``entity_name`` itself becomes stale.
        """
        if invalidate_cache:
            try:
                self.client.entity_tags.invalidate_cached_tags(
                    "catalogs", self.catalog_name,
                )
            except Exception:  # cache invalidation is best-effort
                pass
        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_infos_fetched_at", None)

    def clear(self) -> "Catalog":
        """Public alias for :meth:`_reset_cache`; returns ``self``."""
        self._reset_cache()
        return self

    # ── infos / existence ─────────────────────────────────────────────────────

    @property
    def infos(self) -> CatalogInfo:
        """CatalogInfo — local cache first (TTL-guarded), then remote on miss."""
        now = time.time()

        if self._infos is not None:
            age = now - (self._infos_fetched_at or 0.0)
            if self._infos_ttl is None or age < self._infos_ttl:
                return self._infos
            logger.debug(
                "Cache expired for catalog %r (age=%.0fs, ttl=%.0fs) — refreshing",
                self, age, self._infos_ttl,
            )

        logger.debug("Fetching catalog info for %r from remote", self)
        infos = self.client.workspace_client().catalogs.get(self.catalog_name)
        logger.info("Fetched catalog info for %r from remote", self)
        object.__setattr__(self, "_infos", infos)
        object.__setattr__(self, "_infos_fetched_at", now)
        return self._infos

    @property
    def exists(self) -> bool:
        """``True`` if this catalog is reachable via the Unity Catalog API."""
        try:
            _ = self.infos
            return True
        except NotFound:
            return False

    @property
    def comment(self) -> Optional[str]:
        return self.infos.comment

    @property
    def owner(self) -> Optional[str]:
        return self.infos.owner

    @property
    def storage_root(self) -> Optional[str]:
        return self.infos.storage_root

    # ── navigation ────────────────────────────────────────────────────────────

    def schema(self, name: str) -> "Schema":
        """Return a :class:`Schema` bound to this catalog.

        Args:
            name: Schema name (unqualified).
        """
        from yggdrasil.databricks.schema.schema import Schema as _Schema
        return _Schema(
            service=self.service,
            catalog_name=self.catalog_name,
            schema_name=name,
        )

    def schemas(self) -> Iterator["Schema"]:
        """Iterate over every schema in this catalog (single API call)."""
        from yggdrasil.databricks.schema.schema import Schema as _Schema
        logger.debug("Listing schemas in catalog %r", self)
        for info in self.client.workspace_client().schemas.list(catalog_name=self.catalog_name):
            s = _Schema(
                service=self.service,
                catalog_name=self.catalog_name,
                schema_name=info.name,
            )
            object.__setattr__(s, "_infos", info)
            object.__setattr__(s, "_infos_fetched_at", time.time())
            yield s

    def table(
        self,
        location: str | None = None,
        *,
        schema_name: str | None = None,
        table_name: str | None = None,
    ) -> "Table":
        """Return a :class:`Table` within this catalog.

        Args:
            location:    Two- or three-part dotted name (``"schema.table"`` or
                         ``"catalog.schema.table"``).
            schema_name: Schema override.
            table_name:  Table override.
        """
        return self.client.tables.table(
            location=location,
            catalog_name=self.catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )

    def tables(
        self,
        schema_name: str | None = None,
    ) -> Iterator["Table"]:
        """Iterate over all tables in the given schema (or all schemas)."""
        if schema_name:
            yield from self.client.tables.list_tables(
                catalog_name=self.catalog_name,
                schema_name=schema_name,
            )
        else:
            for s in self.schemas():
                yield from s.tables()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def create(
        self,
        *,
        comment: str | None = None,
        properties: Optional[Mapping[str, str]] = None,
        storage_root: str | None = None,
        if_not_exists: bool = True,
    ) -> "Catalog":
        """Create this catalog in Unity Catalog.

        Args:
            comment:      Human-readable description.
            properties:   Extra key/value properties.
            storage_root: External storage root URI (for external catalogs).
            if_not_exists: Silently succeed if the catalog already exists.
        """
        uc = self.client.workspace_client().catalogs
        logger.debug(
            "Creating catalog %r (storage_root=%s, if_not_exists=%s)",
            self, storage_root, if_not_exists,
        )
        try:
            info = uc.create(
                name=self.catalog_name,
                comment=comment,
                properties=properties,
                storage_root=storage_root,
            )
            object.__setattr__(self, "_infos", info)
            object.__setattr__(self, "_infos_fetched_at", time.time())
        except DatabricksError as exc:
            if if_not_exists and "already exists" in str(exc).lower():
                logger.debug(
                    "Catalog %r already exists — soft-resetting cache", self,
                )
                self._reset_cache()
            else:
                raise
        return self

    def ensure_created(
        self,
        *,
        comment: str | None = None,
        properties: Optional[Mapping[str, str]] = None,
        storage_root: str | None = None,
    ) -> "Catalog":
        """Create this catalog if it does not already exist, then return ``self``."""
        if not self.exists:
            self.create(
                comment=comment,
                properties=properties,
                storage_root=storage_root,
                if_not_exists=True,
            )
        return self

    def delete(
        self,
        *,
        force: bool = False,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Catalog":
        """Delete this catalog from Unity Catalog.

        Args:
            force:       Cascade-delete all child schemas and tables.
            wait:        Block until the API call returns.
            raise_error: Re-raise :exc:`DatabricksError` on failure.
        """
        uc = self.client.workspace_client().catalogs
        logger.debug(
            "Deleting catalog %r (force=%s, wait=%s)", self, force, bool(wait),
        )
        if wait:
            try:
                uc.delete(name=self.catalog_name, force=force)
            except DatabricksError:
                if raise_error:
                    raise
        else:
            Job.make(uc.delete, self.catalog_name).fire_and_forget()

        # Structural change — drop both _infos and the entity-tag cache.
        self._reset_cache(invalidate_cache=True)
        return self

    # ── tags ──────────────────────────────────────────────────────────────────

    @property
    def tags(self) -> tuple[Any, ...]:
        """Catalog-level entity-tag assignments — served from ``client.entity_tags``."""
        return tuple(
            self.client.entity_tags.entity_tags(
                "catalogs", self.catalog_name, default=()
            ) or ()
        )

    def set_tags_ddl(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
    ) -> str:
        """Build an ``ALTER CATALOG … SET TAGS`` DDL statement.

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
        return f"ALTER CATALOG `{self.catalog_name}` SET TAGS ({', '.join(pairs)})"

    def set_tags(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
        mode: ModeLike | None = None,
    ) -> "Catalog":
        """Apply catalog-level tags via the UC ``entity_tag_assignments`` API.

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
            entity_type="catalogs",
            entity_name=self.catalog_name,
            mode=mode,
        )
        return self

    def unset_tags(
        self,
        tag_keys: Iterable[str],
        *,
        if_exists: bool = True,
    ) -> "Catalog":
        """Delete catalog-level tag assignments by key."""
        self.client.entity_tags.delete_entity_tags(
            entity_type="catalogs",
            entity_name=self.catalog_name,
            tag_keys=tag_keys,
            if_exists=if_exists,
        )
        return self

    # ── grants ────────────────────────────────────────────────────────────────

    def _grants_securable_type(self) -> SecurableType:
        return SecurableType.CATALOG

    def _grants_full_name(self) -> str:
        return self.catalog_name

    # ── update ────────────────────────────────────────────────────────────────

    def update(
        self,
        *,
        comment: str | None = None,
        owner: str | None = None,
        properties: Optional[Mapping[str, str]] = None,
    ) -> "Catalog":
        """Update catalog metadata in-place and refresh the local cache."""
        kwargs: dict[str, Any] = {}
        if comment is not None:
            kwargs["comment"] = comment
        if owner is not None:
            kwargs["owner"] = owner
        if properties is not None:
            kwargs["properties"] = properties

        logger.debug(
            "Updating catalog %r (fields=%s)", self, sorted(kwargs.keys()),
        )
        info = self.client.workspace_client().catalogs.update(
            name=self.catalog_name, **kwargs
        )
        object.__setattr__(self, "_infos", info)
        object.__setattr__(self, "_infos_fetched_at", time.time())
        return self

    # ── rename ────────────────────────────────────────────────────────────────

    def rename(self, new_name: str) -> "Catalog":
        """Rename this catalog in-place (``ALTER CATALOG … RENAME TO …``)."""
        new_name = (new_name or "").strip().strip("`")
        if not new_name:
            raise ValueError("Cannot rename catalog to an empty name")
        if new_name == self.catalog_name:
            logger.debug(
                "Skipping rename of catalog %r — new name matches current", self,
            )
            return self

        logger.debug("Renaming catalog %r → %r", self, new_name)

        # Drop the old entity-tag cache key before the rename — the
        # ``entity_name`` is the key, and after the rename it's dead.
        try:
            self.client.entity_tags.invalidate_cached_tags(
                "catalogs", self.catalog_name,
            )
        except Exception:
            pass

        info = self.client.workspace_client().catalogs.update(
            name=self.catalog_name, new_name=new_name,
        )
        self.catalog_name = new_name
        object.__setattr__(self, "_infos", info)
        object.__setattr__(self, "_infos_fetched_at", time.time())
        return self