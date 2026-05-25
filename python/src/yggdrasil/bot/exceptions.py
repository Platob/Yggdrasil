from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class BotError(Exception):
    def __init__(self, detail: str, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class NotFoundError(BotError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=404)


class ConflictError(BotError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=409)


class ForbiddenError(BotError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=403)


class TimeoutError(BotError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=408)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BotError)
    async def handle_bot_error(request: Request, exc: BotError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
