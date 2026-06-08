"""``/api/call`` — invoke a registered remote function.

Request body is yggdrasil pickle (``application/x-yggdrasil-pickle``) or
JSON, shaped ``{"func": str, "args": [...], "kwargs": {...}}``. The return
value is serialized with :func:`serialize_result`, so tabular results come
back as an Arrow IPC stream and everything else as pickle.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request, Response

from yggdrasil.exceptions.api import BadRequestError
from yggdrasil.node.remote import call_registered
from yggdrasil.node.transport import (
    CONTENT_TYPE_PICKLE,
    deserialize_pickle,
    serialize_result,
)

router = APIRouter()

_REQUEST_COUNT = 0


def request_count() -> int:
    return _REQUEST_COUNT


def _bump() -> None:
    global _REQUEST_COUNT
    _REQUEST_COUNT += 1


@router.post("/api/call")
async def call_function(request: Request) -> Response:
    _bump()
    body = await request.body()
    content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()

    if content_type == CONTENT_TYPE_PICKLE:
        payload = deserialize_pickle(body)
    else:
        # JSON (or unlabeled — assume JSON).
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            raise BadRequestError(
                f"Could not parse request body as JSON: {exc}. "
                f"Send {CONTENT_TYPE_PICKLE!r} or valid JSON with "
                f'shape {{"func": str, "args": [...], "kwargs": {{...}}}}.'
            ) from exc

    if not isinstance(payload, dict):
        raise BadRequestError(
            f"Request body must decode to a dict, got {type(payload).__name__}. "
            f'Expected {{"func": str, "args": [...], "kwargs": {{...}}}}.'
        )

    func = payload.get("func")
    if not func or not isinstance(func, str):
        raise BadRequestError(
            f"Missing or invalid 'func' in request body (got {func!r}). "
            f'Expected a function name string, e.g. {{"func": "ns:fn", ...}}.'
        )

    args = tuple(payload.get("args") or ())
    kwargs = payload.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise BadRequestError(
            f"'kwargs' must be an object, got {type(kwargs).__name__}."
        )

    result: Any = call_registered(func, args, kwargs)
    data, ct = serialize_result(result)
    return Response(content=data, media_type=ct)
