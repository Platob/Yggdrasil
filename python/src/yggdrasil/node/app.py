"""create_app — the full node: REST surface + ``/api/call`` remote dispatch.

Builds :func:`create_api` and adds the ``POST /api/call`` endpoint that runs a
registered ``@remote`` function and serializes its result in the transport
format that fits (Arrow for tabular, pickle otherwise).
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import Response

from yggdrasil.node.api.app import create_api
from yggdrasil.node.remote import _REGISTRY
from yggdrasil.node.transport import deserialize_pickle, serialize_result


def create_app(settings=None) -> FastAPI:
    from yggdrasil.node.config import Settings

    settings = settings or Settings()
    app = create_api(settings)

    @app.post("/api/call")
    async def call_endpoint(request: Request):
        payload = deserialize_pickle(await request.body())
        fn = _REGISTRY.get(payload["func"])
        if fn is None:
            return Response(content=b"not found", status_code=404)
        result = fn(*payload.get("args", ()), **payload.get("kwargs", {}))
        data, content_type = serialize_result(result)
        return Response(content=data, media_type=content_type)

    return app
