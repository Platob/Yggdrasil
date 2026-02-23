"""
Reverse proxy for the frontend upstream.

Every request that is NOT matched by /api/v1/* is forwarded to
``Settings.frontend_upstream``.  Response headers and status codes are
preserved so that the caller sees the upstream response verbatim.

Typical use-cases
-----------------
- Development: forward ``/*`` to Vite / CRA dev server (``http://localhost:5173``)
- Production:  forward ``/*`` to a dedicated frontend service / CDN origin

The proxy is mounted at the end of the FastAPI router so that all /api/v1
routes are evaluated first and this handler only fires on unmatched paths.
"""
from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.config import Settings, get_settings

router = APIRouter()

# A single shared async client – created once, reused for all requests.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(follow_redirects=True, timeout=30.0)
    return _client


# ---------------------------------------------------------------------------
# Hop-by-hop headers that must NOT be forwarded to/from the upstream.
# ---------------------------------------------------------------------------
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)


def _forward_headers(headers: httpx.Headers | dict) -> dict:
    return {k: v for k, v in dict(headers).items() if k.lower() not in _HOP_BY_HOP}


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    include_in_schema=False,
)
async def frontend_proxy(
    path: str,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Forward the request to the configured frontend upstream."""
    upstream_base = settings.frontend_upstream.rstrip("/")
    strip = settings.frontend_strip_prefix.rstrip("/")
    forward_path = path if not strip else path.removeprefix(strip.lstrip("/"))

    url = f"{upstream_base}/{forward_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()

    client = _get_client()
    try:
        upstream_response = await client.request(
            method=request.method,
            url=url,
            headers=_forward_headers(request.headers),
            content=body,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Frontend upstream unreachable: {exc}",
        ) from exc

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_forward_headers(upstream_response.headers),
        media_type=upstream_response.headers.get("content-type"),
    )
