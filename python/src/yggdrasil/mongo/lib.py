"""Optional-dependency guard for the MongoDB backend.

Three drivers cooperate, in preference order:

* :mod:`pymongo` (required) — the canonical MongoDB driver. Powers
  every metadata, lifecycle, and command path; also the row-fallback
  read/write surface (cursor → ``list[dict]`` → :class:`pyarrow.Table`).
* :mod:`bson` (ships with pymongo) — exposes :class:`bson.ObjectId`,
  :class:`bson.Decimal128`, :class:`bson.Binary`, the BSON codec, and
  :class:`bson.codec_options.CodecOptions` used by both drivers.
* :mod:`pymongoarrow` (preferred, optional) — Arrow-native fast path
  for :class:`yggdrasil.mongo.collection.MongoCollection` reads
  (``find_arrow_all`` / ``aggregate_arrow_all``) and writes
  (``write``). Slots schema inference straight onto a
  :class:`pyarrow.Table` without going through Python row dicts.

Each accessor caches its module on first call and raises a helpful
"install ``pip install ygg[mongo]``" error otherwise — matches the
pattern used by :mod:`yggdrasil.postgres.lib` and the rest of the
optional-engine guards.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "pymongo",
    "pymongo_module",
    "bson",
    "bson_module",
    "pymongoarrow",
    "pymongoarrow_module",
    "pymongoarrow_api_module",
    "pymongoarrow_schema_module",
    "pymongoarrow_writer_module",
    "has_pymongo",
    "has_pymongoarrow",
]


_pymongo: Any = None
_pymongo_attempted: bool = False
_bson: Any = None
_bson_attempted: bool = False
_pma: Any = None
_pma_attempted: bool = False
_pma_api: Any = None
_pma_schema: Any = None
_pma_writer: Any = None


_INSTALL_HINT = (
    "pymongo is required for yggdrasil.mongo; install it with "
    "`pip install ygg[mongo]` or `pip install pymongo>=4.5`."
)
_PMA_HINT = (
    "pymongoarrow is required for the Arrow-native MongoDB path; "
    "install it with `pip install ygg[mongo]` or "
    "`pip install pymongoarrow>=1.3`."
)


def pymongo_module() -> Any:
    """Return the imported :mod:`pymongo` module."""
    global _pymongo, _pymongo_attempted
    if _pymongo is not None:
        return _pymongo
    if _pymongo_attempted:
        raise ImportError(_INSTALL_HINT)
    _pymongo_attempted = True
    try:
        import pymongo as _mod
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    _pymongo = _mod
    return _mod


def bson_module() -> Any:
    """Return the imported :mod:`bson` module (ships with pymongo)."""
    global _bson, _bson_attempted
    if _bson is not None:
        return _bson
    if _bson_attempted:
        raise ImportError(_INSTALL_HINT)
    _bson_attempted = True
    try:
        import bson as _mod
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    _bson = _mod
    return _mod


def pymongoarrow_module() -> Any:
    """Return the imported :mod:`pymongoarrow` module.

    pymongoarrow is the Arrow-native fast path for reads and writes —
    when present, MongoDB → Arrow round-trips never materialise a
    Python row in between. Falls back to a pymongo cursor + Arrow lift
    when the package isn't installed.
    """
    global _pma, _pma_attempted, _pma_api, _pma_schema, _pma_writer
    if _pma is not None:
        return _pma
    if _pma_attempted:
        raise ImportError(_PMA_HINT)
    _pma_attempted = True
    try:
        import pymongoarrow as _mod
        from pymongoarrow import api as _api
        from pymongoarrow import schema as _schema
        try:
            from pymongoarrow import writer as _writer
        except ImportError:
            _writer = None
    except ImportError as exc:
        raise ImportError(_PMA_HINT) from exc
    _pma = _mod
    _pma_api = _api
    _pma_schema = _schema
    _pma_writer = _writer
    return _mod


def pymongoarrow_api_module() -> Any:
    """``pymongoarrow.api`` — exposes ``find_arrow_all`` / ``aggregate_arrow_all`` / ``write``."""
    if _pma_api is not None:
        return _pma_api
    pymongoarrow_module()
    return _pma_api


def pymongoarrow_schema_module() -> Any:
    """``pymongoarrow.schema`` — exposes :class:`Schema` (the pma one)."""
    if _pma_schema is not None:
        return _pma_schema
    pymongoarrow_module()
    return _pma_schema


def pymongoarrow_writer_module() -> Any:
    """``pymongoarrow.writer`` — only present in newer pymongoarrow.

    ``None`` when the installed pymongoarrow is too old for the writer
    surface; callers should fall back to ``pymongoarrow.api.write`` or
    the pymongo bulk path.
    """
    if _pma_writer is not None:
        return _pma_writer
    if _pma_attempted and _pma is not None:
        return None
    pymongoarrow_module()
    return _pma_writer


def has_pymongo() -> bool:
    """Probe-only — never raises."""
    global _pymongo, _pymongo_attempted
    if _pymongo is not None:
        return True
    if _pymongo_attempted:
        return False
    try:
        pymongo_module()
    except ImportError:
        return False
    return True


def has_pymongoarrow() -> bool:
    """Probe-only — never raises."""
    global _pma, _pma_attempted
    if _pma is not None:
        return True
    if _pma_attempted:
        return False
    try:
        pymongoarrow_module()
    except ImportError:
        return False
    return True


# Convenience singletons matching the rest-of-package pattern
# (``from yggdrasil.polars.lib import polars``). Resolution is
# deferred to first attribute access so importing this module
# alone never fails on a missing optional package.


class _LazyModule:
    __slots__ = ("_loader", "_module")

    def __init__(self, loader):
        self._loader = loader
        self._module = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = self._loader()
        return getattr(self._module, name)

    def __bool__(self) -> bool:
        return self._module is not None


pymongo = _LazyModule(pymongo_module)
bson = _LazyModule(bson_module)
pymongoarrow = _LazyModule(pymongoarrow_module)
