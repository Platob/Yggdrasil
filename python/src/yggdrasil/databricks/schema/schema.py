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
from typing import Any, Iterable, Iterator, Mapping, Optional, TYPE_CHECKING

from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import (
    PermissionsChange,
    Privilege,
    PrivilegeAssignment,
    SchemaInfo,
    SecurableType,
)
from yggdrasil.concurrent.threading import Job
from yggdrasil.databricks.client import DatabricksResource
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io import URL
from yggdrasil.data.enums.mode import ModeLike

from yggdrasil.databricks.sql.sql_utils import DEFAULT_TAG_COLLATION, databricks_tag_literal

if TYPE_CHECKING:
    from yggdrasil.databricks.schema.schemas import Schemas
    from yggdrasil.databricks.catalog.catalog import Catalog
    from yggdrasil.databricks.table.table import Table

__all__ = ["Schema"]

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


class Schema(DatabricksResource):
    def __init__(
        self,
        service: "Schemas | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        *,
        infos_ttl: float | None = None,
        infos: SchemaInfo | None = None,
        infos_fetched_at: float | None = None,
    ):
        if service is None:
            from .schemas import Schemas
            service = Schemas.current()

        super().__init__(service=service)
        self.catalog_name = catalog_name or service.catalog_name
        self.schema_name = schema_name or service.schema_name
        self._infos_ttl = infos_ttl
        self._infos = infos
        self._infos_fetched_at = infos_fetched_at

    # ── identity ──────────────────────────────────────────────────────────────

    def full_name(self, safe: str | bool | None = None) -> str:
        """Return the two-part schema name, optionally backtick-quoted."""
        if safe:
            q = safe if isinstance(safe, str) else "`"
            return f"{q}{self.catalog_name}{q}.{q}{self.schema_name}{q}"
        return f"{self.catalog_name}.{self.schema_name}"

    def __repr__(self) -> str:
        return f"Schema({self.url.to_string()!r})"

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
    def url(self) -> URL:
        return self.client.base_url.with_path(
            f"/explore/data/{self.catalog_name}/{self.schema_name}"
        )

    @property
    def explore_url(self) -> URL:
        """Workspace UI URL pointing at this schema's Catalog Explorer page."""
        return self.url

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
        from yggdrasil.databricks.catalog.catalog import Catalog as _Catalog
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
        comment: str | None = None,
        properties: Optional[Mapping[str, str]] = None,
        storage_root: str | None = None,
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

        # Structural change — drop both _infos and the entity-tag cache.
        self._reset_cache(invalidate_cache=True)
        return self

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
    ) -> "Schema":
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
    ) -> "Schema":
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

    def grant(
        self,
        principal: str,
        privileges: "str | Privilege | Iterable[str | Privilege]",
    ) -> "Schema":
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
    ) -> "Schema":
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
    ) -> "Schema":
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
    ) -> "Schema":
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

    # ── rename ────────────────────────────────────────────────────────────────

    def rename(self, new_name: str) -> "Schema":
        """Rename this schema in-place (``ALTER SCHEMA … RENAME TO …``).

        The catalog parent is unchanged; *new_name* is the unqualified schema name.
        """
        new_name = (new_name or "").strip().strip("`")
        if not new_name:
            raise ValueError("Cannot rename schema to an empty name")
        if new_name == self.schema_name:
            return self

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