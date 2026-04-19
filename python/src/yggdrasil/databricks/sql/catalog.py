"""
Per-catalog resource: lifecycle, navigation, and tag helpers.

The :class:`Catalog` dataclass wraps a single Unity Catalog catalog and
exposes instance-level methods only.  Collection operations live in
:mod:`~yggdrasil.databricks.sql.catalogs`.

Hierarchy navigation
--------------------
::

    catalog["schema_name"]           # → Schema
    catalog["schema_name"]["table"]  # → Table
    catalog.schema("sales")          # → Schema
    catalog.schemas()                # → Iterator[Schema]
    catalog.table("sales.orders")    # → Table
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional, TYPE_CHECKING

from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import CatalogInfo, SecurableType

from yggdrasil.concurrent.threading import Job
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io import URL
from .grants import GrantsMixin
from .sql_utils import DEFAULT_TAG_COLLATION, databricks_tag_literal, quote_ident

if TYPE_CHECKING:
    from .schema import Schema
    from .table import Table

__all__ = ["Catalog"]

logger = logging.getLogger(__name__)


@dataclass
class Catalog(GrantsMixin):
    """A single Unity Catalog catalog — lifecycle, schema navigation, tags."""
    catalog_name: str | None = None

    # TTL for the _infos cache (seconds).  None disables expiry.
    _infos_ttl: float | None = field(default=1800.0, repr=False, compare=False, hash=False)

    _infos: Optional[CatalogInfo] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _infos_fetched_at: float | None = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )

    # ── identity ──────────────────────────────────────────────────────────────

    def full_name(self, safe: bool = None) -> str:
        """Return the catalog name (single-part identifier)."""
        return quote_ident(self.catalog_name) if safe else self.catalog_name

    def __repr__(self) -> str:
        return f"Catalog<{self.url.to_string()!r}>"

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
    def url(self) -> URL:
        return self.client.base_url.with_path(f"/explore/data/{self.catalog_name}")

    # ── cache management ──────────────────────────────────────────────────────

    def _reset_cache(self) -> None:
        """Evict the cached :class:`CatalogInfo`."""
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
                logger.debug(
                    "Cache hit [Catalog._infos] catalog=%s age=%.0fs",
                    self.catalog_name, age,
                )
                return self._infos
            logger.debug(
                "Cache expired [Catalog._infos] catalog=%s age=%.0fs ttl=%.0fs — refreshing",
                self.catalog_name, age, self._infos_ttl,
            )

        infos = self.client.workspace_client().catalogs.get(self.catalog_name)
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
        from .schema import Schema as _Schema
        return _Schema(
            service=self.service,
            catalog_name=self.catalog_name,
            schema_name=name,
        )

    def schemas(self) -> Iterator["Schema"]:
        """Iterate over every schema in this catalog (single API call)."""
        from .schema import Schema as _Schema
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
        if wait:
            try:
                uc.delete(name=self.catalog_name, force=force)
            except DatabricksError:
                if raise_error:
                    raise
        else:
            Job.make(uc.delete, self.catalog_name).fire_and_forget()

        self._reset_cache()
        return self

    # ── tags ──────────────────────────────────────────────────────────────────

    def set_tags_ddl(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
    ) -> str:
        """Build an ``ALTER CATALOG … SET TAGS`` DDL statement."""
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
    ) -> "Catalog":
        """Execute ``ALTER CATALOG … SET TAGS`` for the given mapping."""
        if tags:
            self.sql.execute(self.set_tags_ddl(tags, tag_collation=tag_collation))
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
            return self

        info = self.client.workspace_client().catalogs.update(
            name=self.catalog_name, new_name=new_name,
        )
        self.catalog_name = new_name
        object.__setattr__(self, "_infos", info)
        object.__setattr__(self, "_infos_fetched_at", time.time())
        return self

