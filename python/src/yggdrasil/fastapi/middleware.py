"""Gzip request-body decompression utilities for FastAPI routes."""
from __future__ import annotations

import gzip

from fastapi import Request
from fastapi.routing import APIRoute


class GzipRequest(Request):
    """Transparently decompresses gzip-encoded request bodies."""

    async def body(self) -> bytes:
        if not hasattr(self, "_cached_body"):
            raw = await super().body()
            encoding = self.headers.get("Content-Encoding", "")
            if "gzip" in encoding.lower():
                raw = gzip.decompress(raw)
            self._cached_body = raw  # type: ignore[attr-defined]
        return self._cached_body  # type: ignore[attr-defined]


class GzipRoute(APIRoute):
    """APIRoute subclass that wraps incoming requests with :class:`GzipRequest`."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            request = GzipRequest(request.scope, request.receive)
            return await original(request)

        return handler

