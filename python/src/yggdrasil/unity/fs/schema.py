"""Filesystem-backed :class:`UnitySchema`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from yggdrasil.io.path import Path
from yggdrasil.unity.fs import registry
from yggdrasil.unity.fs.table import FSTable
from yggdrasil.unity.fs.view import FSView
from yggdrasil.unity.info import SchemaInfo
from yggdrasil.unity.schema import UnitySchema

if TYPE_CHECKING:
    from yggdrasil.unity.fs.catalog import FSCatalog
    from yggdrasil.unity.fs.engine import FSEngine


__all__ = ["FSSchema"]


logger = logging.getLogger(__name__)


class FSSchema(UnitySchema):
    """Schema backed by a directory under :attr:`FSCatalog.path`."""

    def __init__(self, *, catalog: "FSCatalog", name: str) -> None:
        self._catalog = catalog
        self._name = name

    # ── identity ───────────────────────────────────────────────────────

    @property
    def engine(self) -> "FSEngine":
        return self._catalog.engine

    @property
    def catalog(self) -> "FSCatalog":
        return self._catalog

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        """Directory this schema lives in (``<catalog>/<name>``)."""
        return self._catalog.schema_path(self._name)

    # ── info ───────────────────────────────────────────────────────────

    def _read_info(self) -> SchemaInfo:
        return registry.read_schema_info(self.path)

    # ── lifecycle ──────────────────────────────────────────────────────

    def create(
        self,
        *,
        comment: str | None = None,
        owner: str | None = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> "FSSchema":
        if self.exists:
            if if_not_exists:
                logger.debug("Schema %r already exists — skipping create", self)
                return self
            raise FileExistsError(
                f"Schema {self.full_name!r} already exists at "
                f"{self.path.full_path()!r}."
            )
        if not self._catalog.exists:
            raise FileNotFoundError(
                f"Catalog {self._catalog.full_name!r} does not exist. "
                "Create it first via engine.create_catalog(name)."
            )
        logger.debug(
            "Creating schema %r (owner=%r, properties=%r)",
            self, owner, dict(properties or {}),
        )
        self.path.mkdir(parents=True, exist_ok=True)
        info = SchemaInfo(
            catalog_name=self._catalog.name,
            name=self._name,
            comment=comment,
            owner=owner,
            properties=dict(properties or {}),
        )
        registry.write_schema_info(self.path, info)
        self._store_info(info)
        logger.info("Created schema %r", self)
        return self

    def delete(
        self,
        *,
        recursive: bool = False,
        missing_ok: bool = True,
    ) -> "FSSchema":
        if not self.exists:
            if missing_ok:
                logger.debug("Schema %r does not exist — skipping delete", self)
                return self
            raise FileNotFoundError(
                f"Schema {self.full_name!r} does not exist."
            )
        if not recursive:
            children = [
                entry for entry in self.path.iterdir()
                if entry.name != registry.META_DIR and not entry.name.startswith(".")
            ]
            if children:
                names = sorted(c.name for c in children)
                raise OSError(
                    f"Schema {self.full_name!r} is not empty "
                    f"(tables/views={names!r}). Pass recursive=True to cascade."
                )
        logger.debug("Deleting schema %r (recursive=%s)", self, recursive)
        self.path.remove(recursive=True, missing_ok=missing_ok)
        self._invalidate_info()
        logger.info("Deleted schema %r", self)
        return self

    # ── table / view navigation ────────────────────────────────────────

    def table(self, name: str) -> FSTable:
        if not name or "/" in name or "." in name:
            raise ValueError(
                f"Table name must be a non-empty single segment without "
                f"'/' or '.'; got {name!r}."
            )
        return FSTable(schema=self, name=name)

    def tables(self) -> Iterator[FSTable]:
        if not self.path.exists():
            return
        logger.debug("Listing tables in schema %r", self)
        for entry in self.path.iterdir():
            if entry.name == registry.META_DIR or entry.name.startswith("."):
                continue
            if not (entry / registry.META_DIR / registry.TABLE_FILE).exists():
                continue
            yield FSTable(schema=self, name=entry.name)

    def view(self, name: str) -> FSView:
        if not name or "/" in name or "." in name:
            raise ValueError(
                f"View name must be a non-empty single segment without "
                f"'/' or '.'; got {name!r}."
            )
        return FSView(schema=self, name=name)

    def views(self) -> Iterator[FSView]:
        if not self.path.exists():
            return
        logger.debug("Listing views in schema %r", self)
        for entry in self.path.iterdir():
            if entry.name == registry.META_DIR or entry.name.startswith("."):
                continue
            if not (entry / registry.META_DIR / registry.VIEW_FILE).exists():
                continue
            yield FSView(schema=self, name=entry.name)

    # ── creators ───────────────────────────────────────────────────────

    def create_table(
        self,
        name: str,
        schema: Any,
        *,
        format: Any = ...,
        partition_by: "tuple[str, ...] | list[str] | None" = None,
        comment: str | None = None,
        owner: str | None = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> FSTable:
        table = self.table(name)
        table.create(
            schema=schema,
            format=format,
            partition_by=partition_by,
            comment=comment,
            owner=owner,
            properties=properties,
            if_not_exists=if_not_exists,
        )
        return table

    def create_view(
        self,
        name: str,
        source: Any,
        *,
        definition: str | None = None,
        comment: str | None = None,
        owner: str | None = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> FSView:
        view = self.view(name)
        view.create(
            source=source,
            definition=definition,
            comment=comment,
            owner=owner,
            properties=properties,
            if_not_exists=if_not_exists,
        )
        return view

    # ── helpers ────────────────────────────────────────────────────────

    def child_path(self, name: str) -> Path:
        """Directory of *name* under this schema (table or view root)."""
        return self.path / name
