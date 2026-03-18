from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class APIError(Exception):
    def __init__(self, detail: str, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class NotFoundError(APIError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=404)


class ConflictError(APIError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=409)


class ForbiddenError(APIError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=403)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
