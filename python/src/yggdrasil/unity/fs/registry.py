"""JSON sidecar registry for the filesystem Unity backend.

Every resource keeps its info dataclass in a hidden ``_yggdrasil/``
directory under its node. The names are stable so the on-disk layout
itself doubles as the catalog index — a directory IS a catalog when
``_yggdrasil/catalog.json`` exists, a schema when
``_yggdrasil/schema.json`` exists, and so on.

Layout (relative to the engine's base :class:`Path`)::

    <base>/<catalog>/_yggdrasil/catalog.json
    <base>/<catalog>/<schema>/_yggdrasil/schema.json
    <base>/<catalog>/<schema>/<table>/_yggdrasil/table.json
    <base>/<catalog>/<schema>/<table>/data/*.{parquet,arrow,…}
    <base>/<catalog>/<schema>/<view>/_yggdrasil/view.json

Reads are one ``read_bytes`` + JSON decode; writes parents-mkdir then
``write_bytes``. Nothing else lives in the registry — schema sniffing,
listing, partition discovery all walk the on-disk layout directly.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from yggdrasil.io.path import Path
from yggdrasil.pickle import json as json_module
from yggdrasil.unity.info import CatalogInfo, SchemaInfo, TableInfo, ViewInfo


__all__ = [
    "META_DIR",
    "CATALOG_FILE",
    "SCHEMA_FILE",
    "TABLE_FILE",
    "VIEW_FILE",
    "DATA_DIR",
    "write_catalog_info",
    "read_catalog_info",
    "write_schema_info",
    "read_schema_info",
    "write_table_info",
    "read_table_info",
    "write_view_info",
    "read_view_info",
    "delete_metadata",
]


logger = logging.getLogger(__name__)


#: Hidden metadata directory name under each resource node. Leading
#: dot keeps :meth:`FolderIO.iter_children` from picking it up as a
#: data file when the table directory is iterated.
META_DIR = "_yggdrasil"
CATALOG_FILE = "catalog.json"
SCHEMA_FILE = "schema.json"
TABLE_FILE = "table.json"
VIEW_FILE = "view.json"

#: Subdirectory under each table where data files live. Keeps the
#: metadata folder and the data files visually separate so a human
#: poking at the layout knows at a glance where the rows are.
DATA_DIR = "data"


# ── low-level read / write ──────────────────────────────────────────────


def _meta_path(node: Path, filename: str) -> Path:
    return node / META_DIR / filename


def _load(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file {path.full_path()!r} does not exist. "
            "The resource has not been created or the layout is corrupt."
        )
    payload = json_module.loads(path.read_bytes())
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Metadata file {path.full_path()!r} contains {type(payload).__name__}; "
            "expected a JSON object."
        )
    return payload


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_module.dumps(payload, to_bytes=True))


# ── catalog ─────────────────────────────────────────────────────────────


def write_catalog_info(node: Path, info: CatalogInfo) -> None:
    """Persist *info* under ``<node>/_yggdrasil/catalog.json``."""
    logger.debug("Writing catalog info to %r", node)
    _dump(_meta_path(node, CATALOG_FILE), info.to_dict())


def read_catalog_info(node: Path) -> CatalogInfo:
    """Read ``<node>/_yggdrasil/catalog.json`` into a :class:`CatalogInfo`."""
    return CatalogInfo.from_dict(_load(_meta_path(node, CATALOG_FILE)))


# ── schema ──────────────────────────────────────────────────────────────


def write_schema_info(node: Path, info: SchemaInfo) -> None:
    logger.debug("Writing schema info to %r", node)
    _dump(_meta_path(node, SCHEMA_FILE), info.to_dict())


def read_schema_info(node: Path) -> SchemaInfo:
    return SchemaInfo.from_dict(_load(_meta_path(node, SCHEMA_FILE)))


# ── table ───────────────────────────────────────────────────────────────


def write_table_info(node: Path, info: TableInfo) -> None:
    logger.debug("Writing table info to %r", node)
    _dump(_meta_path(node, TABLE_FILE), info.to_dict())


def read_table_info(node: Path) -> TableInfo:
    return TableInfo.from_dict(_load(_meta_path(node, TABLE_FILE)))


# ── view ────────────────────────────────────────────────────────────────


def write_view_info(node: Path, info: ViewInfo) -> None:
    logger.debug("Writing view info to %r", node)
    _dump(_meta_path(node, VIEW_FILE), info.to_dict())


def read_view_info(node: Path) -> ViewInfo:
    return ViewInfo.from_dict(_load(_meta_path(node, VIEW_FILE)))


# ── deletion ────────────────────────────────────────────────────────────


def delete_metadata(node: Path, *, missing_ok: bool = True) -> None:
    """Remove the ``_yggdrasil/`` sidecar under *node*.

    Used by ``delete`` paths so the resource stops resolving while
    leaving any data files alone — the caller (or a higher-level
    ``delete`` with ``recursive=True``) decides whether to wipe data.
    """
    meta = node / META_DIR
    if not meta.exists():
        if missing_ok:
            return
        raise FileNotFoundError(
            f"Metadata directory {meta.full_path()!r} does not exist."
        )
    meta.remove(recursive=True, missing_ok=missing_ok)
