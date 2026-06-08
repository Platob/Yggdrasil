"""Databricks SQL helpers and engine wrappers."""

from .exceptions import SQLError
from .sql_utils import *


def __getattr__(name):
    # ``SQLEngine`` is loaded lazily to avoid a circular import: engine.py
    # imports the Unity Catalog subpackages (catalog/schema/table/...), which
    # themselves pull sql_utils from this package's namespace.
    if name == "SQLEngine":
        from .engine import SQLEngine as _SQLEngine
        return _SQLEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
