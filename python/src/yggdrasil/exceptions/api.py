"""API server exceptions — raised when yggdrasil *is* the server.

Unlike the HTTP client hierarchy (:mod:`.http`), these carry only a
``detail`` string and ``status_code`` int — no ``Response`` object —
because they originate inside our own request handlers.

Both the bot and fastapi apps should raise these instead of defining
local duplicates.  A single :func:`register_exception_handlers` wires
them into any FastAPI application.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import YGGException

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = [
    "APIError",
    "BadRequestError",
    "UnauthorizedError",
    "ForbiddenError",
    "NotFoundError",
    "ConflictError",
    "TimeoutError",
    "UnprocessableError",
    "register_api_exception_handlers",
]


class APIError(YGGException):
    """Base for every API-handler exception.  Carries HTTP semantics."""

    status_code: int = 400

    def __init__(self, detail: str = "", *, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class BadRequestError(APIError):
    """400 Bad Request."""
    status_code = 400


class UnauthorizedError(APIError):
    """401 Unauthorized."""
    status_code = 401


class ForbiddenError(APIError):
    """403 Forbidden."""
    status_code = 403


class NotFoundError(APIError):
    """404 Not Found."""
    status_code = 404


class MethodNotAllowedError(APIError):
    """405 Method Not Allowed."""
    status_code = 405


class ConflictError(APIError):
    """409 Conflict."""
    status_code = 409


class TimeoutError(APIError):
    """408 Request Timeout."""
    status_code = 408


class UnprocessableError(APIError):
    """422 Unprocessable Entity."""
    status_code = 422


class TooManyRequestsError(APIError):
    """429 Too Many Requests."""
    status_code = 429


def register_api_exception_handlers(app: "FastAPI") -> None:
    """Wire :class:`APIError` into a FastAPI app's exception handling."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.exception_handler(APIError)
    async def _handle(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status": exc.status_code},
        )
