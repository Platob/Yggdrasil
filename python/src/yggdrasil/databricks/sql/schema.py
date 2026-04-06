"""
Per-schema resource: lifecycle, table navigation, and tag helpers.

The :class:`Schema` dataclass wraps a single Unity Catalog schema and exposes
instance-level methods only.  Collection operations live in
:mod:`~yggdrasil.databricks.sql.catalogs`.

Hierarchy navigation
--------------------
::

    schema["table_name"]  # → Table
    schema.table("orders")  # → Table
    schema.tables()         # → Iterator[Table]
    schema.catalog          # → Catalog (navigate up)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional, TYPE_CHECKING

from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import SchemaInfo

from yggdrasil.concurrent.threading import Job
from yggdrasil.databricks.client import DatabricksResource, DatabricksService
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io import URL

from .types import escape_sql_string

if TYPE_CHECKING:
    from .catalog import Catalog
    from .table import Table

__all__ = ["Schema"]

logger = logging.getLogger(__name__)


@dataclass
class Schema(DatabricksResource):
    """A single Unity Catalog schema — lifecycle, table navigation, tags."""

    service: DatabricksService

    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None

    # TTL for the _infos cache (seconds).  None disables expiry.
    _infos_ttl: Optional[float] = field(default=1800.0, repr=False, compare=False, hash=False)

    _infos: Optional[SchemaInfo] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _infos_fetched_at: Optional[float] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )

    # ── identity ──────────────────────────────────────────────────────────────

    def full_name(self, safe: str | bool | None = None) -> str:
        """Return the two-part schema name, optionally backtick-quoted."""
        if safe:
            q = safe if isinstance(safe, str) else "`"
            return f"{q}{self.catalog_name}{q}.{q}{self.schema_name}{q}"
        return f"{self.catalog_name}.{self.schema_name}"

    def __repr__(self) -> str:
        return f"Schema<{self.url.to_string()!r}>"

    def __str__(self) -> str:
        return self.full_name()

    # ── dict-like navigation ──────────────────────────────────────────────────

    def __getitem__(self, name: str) -> "Table":
        """``schema["table_name"]`` → :class:`Table`."""
        return self.table(name)

    # ── URL ───────────────────────────────────────────────────────────────────

    @property
    def url(self) -> URL:
        return self.client.base_url.with_path(
            f"/explore/data/{self.catalog_name}/{self.schema_name}"
        )

    # ── cache management ──────────────────────────────────────────────────────

    def _reset_cache(self) -> None:
        """Evict the cached :class:`SchemaInfo`."""
        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_infos_fetched_at", None)

    def clear(self) -> "Schema":
        """Public alias for :meth:`_reset_cache`; returns ``self``."""
        self._reset_cache()
        return self

    # ── infos / existence ─────────────────────────────────────────────────────

    @property
    def infos(self) -> SchemaInfo:
        """SchemaInfo — local cache first (TTL-guarded), then remote on miss."""
        now = time.time()

        if self._infos is not None:
            age = now - (self._infos_fetched_at or 0.0)
            if self._infos_ttl is None or age < self._infos_ttl:
                logger.debug(
                    "Cache hit [Schema._infos] schema=%s age=%.0fs",
                    self.full_name(), age,
                )
                return self._infos
            logger.debug(
                "Cache expired [Schema._infos] schema=%s age=%.0fs ttl=%.0fs — refreshing",
                self.full_name(), age, self._infos_ttl,
            )

        infos = self.client.workspace_client().schemas.get(full_name=self.full_name())
        object.__setattr__(self, "_infos", infos)
        object.__setattr__(self, "_infos_fetched_at", now)
        return self._infos

    @property
    def exists(self) -> bool:
        """``True`` if this schema is reachable via the Unity Catalog API."""
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
    def storage_location(self) -> Optional[str]:
        return self.infos.storage_location

    @property
    def storage_root(self) -> Optional[str]:
        return self.infos.storage_root

    # ── navigation ────────────────────────────────────────────────────────────

    @property
    def catalog(self) -> "Catalog":
        """Navigate up to the parent :class:`Catalog`."""
        from .catalog import Catalog as _Catalog
        return _Catalog(service=self.service, catalog_name=self.catalog_name)

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

    def tables(self) -> Iterator["Table"]:
        """Iterate over all tables in this schema."""
        return self.client.tables.list_tables(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def create(
        self,
        *,
        comment: Optional[str] = None,
        properties: Optional[Mapping[str, str]] = None,
        storage_root: Optional[str] = None,
        if_not_exists: bool = True,
    ) -> "Schema":
        """Create this schema in Unity Catalog.

        Args:
            comment:      Human-readable description.
            properties:   Extra key/value properties.
            storage_root: External storage root URI.
            if_not_exists: Silently succeed if the schema already exists.
        """
        uc = self.client.workspace_client().schemas
        try:
            info = uc.create(
                catalog_name=self.catalog_name,
                name=self.schema_name,
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
        comment: Optional[str] = None,
        properties: Optional[Mapping[str, str]] = None,
        storage_root: Optional[str] = None,
    ) -> "Schema":
        """Create this schema if it does not already exist, then return ``self``."""
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
    ) -> "Schema":
        """Delete this schema from Unity Catalog.

        Args:
            force:       Cascade-delete all child tables.
            wait:        Block until the API call returns.
            raise_error: Re-raise :exc:`DatabricksError` on failure.
        """
        uc = self.client.workspace_client().schemas
        if wait:
            try:
                uc.delete(full_name=self.full_name(), force=force)
            except DatabricksError:
                if raise_error:
                    raise
        else:
            Job.make(uc.delete, self.full_name()).fire_and_forget()

        self._reset_cache()
        return self

    # ── tags ──────────────────────────────────────────────────────────────────

    def set_tags_ddl(self, tags: Mapping[str, str] | None) -> str:
        """Build an ``ALTER SCHEMA … SET TAGS`` DDL statement."""
        pairs: list[str] = []
        for k, v in (tags or {}).items():
            key = str(k).strip() if k is not None else ""
            val = str(v).strip() if v is not None else ""
            if key and val:
                pairs.append(
                    f"'{escape_sql_string(key)}' = '{escape_sql_string(val)}'"
                )
        if not pairs:
            raise ValueError(f"Cannot set empty tags on {self!r}")
        return (
            f"ALTER SCHEMA {self.full_name(safe=True)} "
            f"SET TAGS ({', '.join(pairs)})"
        )

    def set_tags(self, tags: Mapping[str, str] | None) -> "Schema":
        """Execute ``ALTER SCHEMA … SET TAGS`` for the given mapping."""
        if tags:
            self.sql.execute(self.set_tags_ddl(tags))
        return self

    # ── update ────────────────────────────────────────────────────────────────

    def update(
        self,
        *,
        comment: Optional[str] = None,
        owner: Optional[str] = None,
        properties: Optional[Mapping[str, str]] = None,
    ) -> "Schema":
        """Update schema metadata in-place and refresh the local cache."""
        kwargs: dict[str, Any] = {}
        if comment is not None:
            kwargs["comment"] = comment
        if owner is not None:
            kwargs["owner"] = owner
        if properties is not None:
            kwargs["properties"] = properties

        info = self.client.workspace_client().schemas.update(
            full_name=self.full_name(), **kwargs
        )
        object.__setattr__(self, "_infos", info)
        object.__setattr__(self, "_infos_fetched_at", time.time())
        return self

