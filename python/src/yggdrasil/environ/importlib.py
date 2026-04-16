from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType
from typing import Any

__all__ = [
    "cached_import",
    "cached_from_import",
]


@lru_cache(maxsize=None)
def cached_import(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


@lru_cache(maxsize=None)
def cached_from_import(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)