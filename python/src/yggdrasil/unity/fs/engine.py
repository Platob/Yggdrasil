"""Filesystem-backed :class:`UnityEngine` over a :class:`Path` base.

``FSEngine(base)`` takes any :class:`Path` (a :class:`LocalPath`, a
remote path, anything that satisfies the :class:`Path` contract) and
treats it as the root of a Unity-Catalog-style namespace. Catalogs are
direct children; their metadata lives under ``<catalog>/_yggdrasil/``.

The engine is intentionally lightweight: navigation methods return
fresh handles, listings walk :meth:`Path.iterdir` and filter by
metadata presence. There is no in-memory catalog index — the on-disk
layout IS the index.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Mapping

from yggdrasil.io.path import Path
from yggdrasil.unity.engine import UnityEngine
from yggdrasil.unity.fs.catalog import FSCatalog
from yggdrasil.unity.fs.registry import CATALOG_FILE, META_DIR


__all__ = ["FSEngine"]


logger = logging.getLogger(__name__)


class FSEngine(UnityEngine):
    """Unity-Catalog-style facade over a filesystem :class:`Path` root.

    Catalogs are direct subdirectories of :attr:`base` carrying a
    ``_yggdrasil/catalog.json`` sidecar; ``catalogs()`` lists exactly
    those entries, so a foreign directory under the same root is
    invisible.
    """

    def __init__(self, base: "Path | str | Any") -> None:
        self.base: Path = base if isinstance(base, Path) else Path.from_(base)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base={self.base!r})"

    # ── navigation ─────────────────────────────────────────────────────

    def catalog(self, name: str) -> FSCatalog:
        if not name or "/" in name or "." in name:
            raise ValueError(
                f"Catalog name must be a non-empty single segment without "
                f"'/' or '.'; got {name!r}."
            )
        return FSCatalog(engine=self, name=name)

    def catalogs(self) -> Iterator[FSCatalog]:
        if not self.base.exists():
            return
        logger.debug("Listing catalogs under %r", self.base)
        for entry in self.base.iterdir():
            if entry.name.startswith("."):
                continue
            if not (entry / META_DIR / CATALOG_FILE).exists():
                continue
            yield FSCatalog(engine=self, name=entry.name)

    def create_catalog(
        self,
        name: str,
        *,
        comment: str | None = None,
        owner: str | None = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> FSCatalog:
        catalog = self.catalog(name)
        catalog.create(
            comment=comment,
            owner=owner,
            properties=properties,
            if_not_exists=if_not_exists,
        )
        return catalog

    # ── helpers exposed for tests / power-users ────────────────────────

    def catalog_path(self, name: str) -> Path:
        """Root directory of *name* under :attr:`base`. Not guaranteed to exist."""
        return self.base / name
