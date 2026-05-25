"""FastAPI service exceptions — thin re-exports from :mod:`yggdrasil.exceptions.api`."""
from __future__ import annotations

from yggdrasil.exceptions.api import (
    APIError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    register_api_exception_handlers as register_exception_handlers,
)

__all__ = [
    "APIError",
    "NotFoundError",
    "ConflictError",
    "ForbiddenError",
    "register_exception_handlers",
]
