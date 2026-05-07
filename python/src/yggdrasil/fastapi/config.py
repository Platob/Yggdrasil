"""Settings for the Yggdrasil FastAPI service.

The service is intentionally minimal: it wraps a
:class:`yggdrasil.io.tabular.TabularEngine` (the canonical
``catalog.schema.name`` registry) and exposes Arrow-first endpoints
on top. Every knob here is overridable through ``YGG_API_*`` env
vars so a deploy can override host/port/limits without rewriting
code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: "str | None", default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: "str | None", default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Expected integer for env var, got {value!r}. "
            "Pass a base-10 integer or unset the variable to use the default."
        ) from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for the FastAPI app.

    All fields have sensible defaults; pass overrides explicitly or
    rely on :func:`get_settings` to read from the environment.
    """

    app_name: str = "yggdrasil-api"
    app_version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    allow_remote: bool = False
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    api_prefix: str = ""
    # Arrow IPC stream is the default response type — the most
    # efficient transfer when both ends speak Arrow. ``application/octet-stream``
    # is what we tag a stream-format payload with; the file format
    # uses ``application/vnd.apache.arrow.file``.
    default_media_type: str = "application/vnd.apache.arrow.stream"
    # Streaming batch size (rows). Caps a single record-batch in the
    # response so massive tables don't blow out the writer.
    stream_batch_rows: int = 65_536
    # Cap for the request body on registration / SQL endpoints.
    max_request_bytes: int = 256 * 1024 * 1024  # 256 MiB

    @property
    def local_clients(self) -> "set[str]":
        return {"127.0.0.1", "::1", "localhost"}


def get_settings() -> Settings:
    """Build :class:`Settings` from ``YGG_API_*`` environment variables."""
    return Settings(
        app_name=os.getenv("YGG_API_APP_NAME", "yggdrasil-api"),
        app_version=os.getenv("YGG_API_APP_VERSION", "1.0.0"),
        host=os.getenv("YGG_API_HOST", "127.0.0.1"),
        port=_as_int(os.getenv("YGG_API_PORT"), 8000),
        allow_remote=_as_bool(os.getenv("YGG_API_ALLOW_REMOTE"), False),
        docs_url=os.getenv("YGG_API_DOCS_URL", "/docs"),
        redoc_url=os.getenv("YGG_API_REDOC_URL", "/redoc"),
        openapi_url=os.getenv("YGG_API_OPENAPI_URL", "/openapi.json"),
        api_prefix=os.getenv("YGG_API_PREFIX", ""),
        default_media_type=os.getenv(
            "YGG_API_DEFAULT_MEDIA_TYPE", "application/vnd.apache.arrow.stream"
        ),
        stream_batch_rows=_as_int(os.getenv("YGG_API_STREAM_BATCH_ROWS"), 65_536),
        max_request_bytes=_as_int(
            os.getenv("YGG_API_MAX_REQUEST_BYTES"), 256 * 1024 * 1024
        ),
    )
