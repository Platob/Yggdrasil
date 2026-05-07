"""Lightweight HTTP-shaped exceptions for the API.

Routers raise :class:`APIError` (or one of the named subclasses) when
they want to fail with a specific status code; the registered handler
turns them into JSON. We deliberately keep the surface tiny — FastAPI
already has :class:`HTTPException` and the stdlib has plenty of error
classes; this just gives us a place to attach helpful context (what
was passed, what's available, what to try next) without sprinkling
raw status codes across routers.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class APIError(Exception):
    """An HTTP-shaped error with a status code and message.

    The message goes straight into the JSON body's ``detail`` field;
    keep it sharp and useful — see :doc:`/AGENTS` on error tone.
    """

    status_code: int = 400

    def __init__(self, detail: str, status_code: "int | None" = None) -> None:
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class NotFound(APIError):
    status_code = 404


class Conflict(APIError):
    status_code = 409


class Forbidden(APIError):
    status_code = 403


def register_exception_handlers(app: FastAPI) -> None:
    """Wire :class:`APIError` and friends to JSON responses on *app*."""

    @app.exception_handler(APIError)
    async def _handle_api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
