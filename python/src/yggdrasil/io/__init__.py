from __future__ import annotations

# URL and IOStats are eager — they don't pull anything that re-enters
# ``yggdrasil.data``. Holder/Memory/MemoryStream are deferred via PEP
# 562 ``__getattr__`` because :class:`Holder` inherits
# :class:`yggdrasil.io.tabular.base.Tabular`, which imports
# :class:`yggdrasil.data.schema.Schema`. Schema in turn pulls
# ``data_field`` — and when ``yggdrasil.data`` is the entry point of the
# import (e.g. via ``data/types/base.py`` → ``data.enums.mime_type`` →
# ``yggdrasil.io``), ``data_field`` is still mid-flight, so eagerly
# loading Holder here closes a circular import.
#
# IO is lazy for the same reason: pulling it would trigger the
# buffer/tabular chain.
from .io_stats import IOStats


_LAZY_DATA_NAMES = {"Holder", "Memory", "MemoryStream"}
_LAZY_IO_NAMES = {"IO"}

# The concrete format leaves used to hide behind ``io.primitive`` /
# ``io.nested`` umbrella packages; the grouping layer is gone — every
# leaf module sits directly under ``yggdrasil.io``. Their public classes
# are re-exported here lazily (same circular-import guard as above:
# importing a leaf pulls Tabular → data, so it must stay deferred) so
# ``from yggdrasil.io import ArrowIPCFile`` keeps working.
_LAZY_LEAF_NAMES = {
    "ArrowIPCFile": "arrow_ipc_file",
    "ParquetFile": "parquet_file",
    "CSVFile": "csv_file",
    "JSONFile": "json_file",
    "NDJSONFile": "ndjson_file",
    "XLSXFile": "xlsx_file",
    "PickleFile": "pickle_file",
    "ZipFile": "zip_file",
    "ZipOptions": "zip_file",
    "ZipEntryFile": "zip_file",
    "DeltaFolder": "delta",
    "DeltaOptions": "delta",
}


def __getattr__(name: str):
    if name == "URL":
        from yggdrasil.url import URL as value

        globals()["URL"] = value
        return value
    if name in _LAZY_DATA_NAMES:
        if name == "Holder":
            from .holder import Holder as value
        elif name == "Memory":
            from yggdrasil.path.memory import Memory as value
        else:
            from yggdrasil.path.memory_stream import MemoryStream as value
        globals()[name] = value
        return value
    if name in _LAZY_IO_NAMES:
        from .base import IO as value
        globals()[name] = value
        return value
    if name in _LAZY_LEAF_NAMES:
        from importlib import import_module

        value = getattr(import_module(f"{__name__}.{_LAZY_LEAF_NAMES[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | _LAZY_DATA_NAMES
        | _LAZY_IO_NAMES
        | set(_LAZY_LEAF_NAMES)
        | {"URL", "IOStats"}
    )
