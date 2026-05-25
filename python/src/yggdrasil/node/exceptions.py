"""Bot API exceptions — thin re-exports from :mod:`yggdrasil.exceptions.api`."""
from __future__ import annotations

from yggdrasil.exceptions.api import (
    APIError as BotError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    TimeoutError,
    register_api_exception_handlers as register_exception_handlers,
)

__all__ = [
    "BotError",
    "NotFoundError",
    "ConflictError",
    "ForbiddenError",
    "TimeoutError",
    "register_exception_handlers",
]
