"""Filesystem-backed :class:`UnityCatalog`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Mapping

from yggdrasil.io.path import Path
from yggdrasil.unity.catalog import UnityCatalog
from yggdrasil.unity.fs import registry
from yggdrasil.unity.fs.schema import FSSchema
from yggdrasil.unity.info import CatalogInfo

if TYPE_CHECKING:
    from yggdrasil.unity.fs.engine import FSEngine


__all__ = ["FSCatalog"]


logger = logging.getLogger(__name__)


class FSCatalog(UnityCatalog):
    """Catalog backed by a directory under :attr:`FSEngine.base`."""

    def __init__(self, *, engine: "FSEngine", name: str) -> None:
        self._engine = engine
        self._name = name

    # ── identity ───────────────────────────────────────────────────────

    @property
    def engine(self) -> "FSEngine":
        return self._engine

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        """Directory this catalog lives in (``<base>/<name>``)."""
        return self._engine.catalog_path(self._name)

    # ── info ───────────────────────────────────────────────────────────

    def _read_info(self) -> CatalogInfo:
        return registry.read_catalog_info(self.path)

    # ── lifecycle ──────────────────────────────────────────────────────

    def create(
        self,
        *,
        comment: str | None = None,
        owner: str | None = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> "FSCatalog":
        if self.exists:
            if if_not_exists:
                logger.debug("Catalog %r already exists — skipping create", self)
                return self
            raise FileExistsError(
                f"Catalog {self.full_name!r} already exists at "
                f"{self.path.full_path()!r}."
            )
        logger.debug(
            "Creating catalog %r (owner=%r, properties=%r)",
            self, owner, dict(properties or {}),
        )
        self.path.mkdir(parents=True, exist_ok=True)
        info = CatalogInfo(
            name=self._name,
            comment=comment,
            owner=owner,
            properties=dict(properties or {}),
        )
        registry.write_catalog_info(self.path, info)
        self._store_info(info)
        logger.info("Created catalog %r", self)
        return self

    def delete(
        self,
        *,
        recursive: bool = False,
        missing_ok: bool = True,
    ) -> "FSCatalog":
        if not self.exists:
            if missing_ok:
                logger.debug("Catalog %r does not exist — skipping delete", self)
                return self
            raise FileNotFoundError(
                f"Catalog {self.full_name!r} does not exist."
            )
        if not recursive:
            children = [
                entry for entry in self.path.iterdir()
                if entry.name != registry.META_DIR and not entry.name.startswith(".")
            ]
            if children:
                names = sorted(c.name for c in children)
                raise OSError(
                    f"Catalog {self.full_name!r} is not empty (schemas={names!r}). "
                    "Pass recursive=True to cascade the delete."
                )
        logger.debug("Deleting catalog %r (recursive=%s)", self, recursive)
        self.path.remove(recursive=True, missing_ok=missing_ok)
        self._invalidate_info()
        logger.info("Deleted catalog %r", self)
        return self

    # ── schema navigation ──────────────────────────────────────────────

    def schema(self, name: str) -> FSSchema:
        if not name or "/" in name or "." in name:
            raise ValueError(
                f"Schema name must be a non-empty single segment without "
                f"'/' or '.'; got {name!r}."
            )
        return FSSchema(catalog=self, name=name)

    def schemas(self) -> Iterator[FSSchema]:
        if not self.path.exists():
            return
        logger.debug("Listing schemas in catalog %r", self)
        for entry in self.path.iterdir():
            if entry.name == registry.META_DIR or entry.name.startswith("."):
                continue
            if not (entry / registry.META_DIR / registry.SCHEMA_FILE).exists():
                continue
            yield FSSchema(catalog=self, name=entry.name)

    def create_schema(
        self,
        name: str,
        *,
        comment: str | None = None,
        owner: str | None = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> FSSchema:
        schema = self.schema(name)
        schema.create(
            comment=comment,
            owner=owner,
            properties=properties,
            if_not_exists=if_not_exists,
        )
        return schema

    # ── helpers ────────────────────────────────────────────────────────

    def schema_path(self, name: str) -> Path:
        """Directory of *name* under this catalog. Not guaranteed to exist."""
        return self.path / name
