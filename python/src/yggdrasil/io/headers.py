# yggdrasil.io.headers
from __future__ import annotations

import platform
import re
import socket
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Mapping, MutableMapping, Optional, Union

from yggdrasil.version import __version_info__, __version__
from .buffer import BytesIO
from .enums import Codec, MediaType
from ..environ import UserInfo

__all__ = [
    "HeaderValue",
    "PromotedHeaders",
    "normalize_headers",
]

HeaderValue = Union[str, bytes]


PYVERSION = str(platform.python_version())
DEFAULT_USER_AGENT: str = (
    f"yggdrasil/{__version_info__} "
    f"os/{platform.system().lower()} "
    f"py/{PYVERSION}"
)

try:
    DEFAULT_HOSTNAME = socket.gethostname()
except Exception:
    DEFAULT_HOSTNAME = "localhost"

SENSITIVE_HEADER_KEYS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "x-auth-token",
    "x-csrf-token",
    "x-xsrf-token",
    "x-amz-security-token",
    "x-amz-access-token",
}

CANONICAL_HEADER_NAMES = {
    "accept": "Accept",
    "accept-encoding": "Accept-Encoding",
    "accept-language": "Accept-Language",
    "authorization": "Authorization",
    "content-length": "Content-Length",
    "content-type": "Content-Type",
    "content-encoding": "Content-Encoding",
    "cookie": "Cookie",
    "etag": "ETag",
    "host": "Host",
    "last-modified": "Last-Modified",
    "location": "Location",
    "proxy-authorization": "Proxy-Authorization",
    "set-cookie": "Set-Cookie",
    "transfer-encoding": "Transfer-Encoding",
    "user-agent": "User-Agent",
    "x-amz-access-token": "X-Amz-Access-Token",
    "x-amz-security-token": "X-Amz-Security-Token",
    "x-api-key": "X-API-Key",
    "x-auth-token": "X-Auth-Token",
    "x-correlation-id": "X-Correlation-ID",
    "x-csrf-token": "X-CSRF-Token",
    "x-request-id": "X-Request-ID",
    "x-xsrf-token": "X-XSRF-Token",
}

BEARER_RE = re.compile(r"^\s*Bearer\s+(.+)\s*$", re.IGNORECASE)
BASIC_RE = re.compile(r"^\s*Basic\s+(.+)\s*$", re.IGNORECASE)
JWT_LIKE_RE = re.compile(r"[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+")


def _to_text(value: HeaderValue) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_header_name(name: HeaderValue) -> tuple[str, str]:
    text = _to_text(name).strip()
    lower = text.lower()
    return CANONICAL_HEADER_NAMES.get(lower, text), lower


def _parse_int_header(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return int(text)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class PromotedHeaders:
    """
    Common HTTP headers extracted into dedicated typed fields.

    `remaining` contains all non-promoted headers after normalization.
    """

    host: Optional[str] = None
    user_agent: Optional[str] = None
    accept: Optional[str] = None
    accept_encoding: Optional[str] = None
    accept_language: Optional[str] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    content_encoding: Optional[str] = None
    transfer_encoding: Optional[str] = None
    x_request_id: Optional[str] = None
    x_correlation_id: Optional[str] = None
    remaining: dict[str, str] = field(default_factory=dict)

    HEADER_TO_ATTR: ClassVar[dict[str, str]] = {
        "host": "host",
        "user-agent": "user_agent",
        "accept": "accept",
        "accept-encoding": "accept_encoding",
        "accept-language": "accept_language",
        "content-type": "content_type",
        "content-length": "content_length",
        "content-encoding": "content_encoding",
        "transfer-encoding": "transfer_encoding",
        "x-request-id": "x_request_id",
        "x-correlation-id": "x_correlation_id",
    }

    @classmethod
    def extract(
        cls,
        headers: Mapping[HeaderValue, HeaderValue],
        *,
        normalize: bool = True,
    ) -> "PromotedHeaders":
        """
        Extract common headers into typed attributes.

        Matching is case-insensitive. When `normalize=True`, recognized header
        names are canonicalized in `remaining`.
        """
        normalized: dict[str, str] = {}
        lower_to_actual: dict[str, str] = {}

        for raw_name, raw_value in headers.items():
            if normalize:
                actual_name, lower_name = _normalize_header_name(raw_name)
            else:
                actual_name = _to_text(raw_name).strip()
                lower_name = actual_name.lower()

            normalized[actual_name] = _to_text(raw_value)
            lower_to_actual[lower_name] = actual_name

        kwargs: dict[str, object] = {}
        promoted_lowers = set(cls.HEADER_TO_ATTR)

        for header_lower, attr_name in cls.HEADER_TO_ATTR.items():
            actual_name = lower_to_actual.get(header_lower)
            value = normalized.get(actual_name) if actual_name is not None else None

            if attr_name == "content_length":
                kwargs[attr_name] = _parse_int_header(value)
            else:
                kwargs[attr_name] = value

        kwargs["remaining"] = {
            name: value
            for name, value in normalized.items()
            if name.lower() not in promoted_lowers
        }

        return cls(**kwargs)

    @property
    def values(self) -> dict[str, object]:
        return {
            "host": self.host,
            "user_agent": self.user_agent,
            "accept": self.accept,
            "accept_encoding": self.accept_encoding,
            "accept_language": self.accept_language,
            "content_type": self.content_type,
            "content_length": self.content_length,
            "content_encoding": self.content_encoding,
            "transfer_encoding": self.transfer_encoding,
            "x_request_id": self.x_request_id,
            "x_correlation_id": self.x_correlation_id,
        }


def _sanitize_sensitive_value(
    value: str,
    *,
    mode: Literal["remove", "redact", "hash"],
    anonymize: bool,
) -> Optional[str]:
    if not anonymize:
        return value
    if mode == "remove":
        return None
    return "<redacted>"


def _sanitize_authorization_value(
    value: str,
    *,
    mode: Literal["remove", "redact", "hash"],
    anonymize: bool,
) -> Optional[str]:
    if not anonymize:
        return value

    if BEARER_RE.match(value):
        return None if mode == "remove" else "Bearer <redacted>"

    if BASIC_RE.match(value):
        return None if mode == "remove" else "Basic <redacted>"

    return _sanitize_sensitive_value(value, mode=mode, anonymize=anonymize)


def _looks_like_token(value: str) -> bool:
    return len(value) >= 40 and JWT_LIKE_RE.search(value) is not None


def normalize_headers(
    headers: Mapping[HeaderValue, HeaderValue],
    *,
    is_request: bool,
    mode: Literal["remove", "redact", "hash"] = "remove",
    anonymize: bool = False,
    body: Optional[BytesIO] = None,
) -> MutableMapping[str, str]:
    """
    Normalize header names and optionally sanitize sensitive values.

    Behavior:
    - canonicalizes recognized header names
    - optionally removes or redacts sensitive values
    - backfills Content-Type / Transfer-Encoding / Content-Length from `body`
      when missing
    """
    out: MutableMapping[str, str] = {}

    has_content_type = False
    has_content_length = False
    has_content_encoding = False
    has_user_agent = False

    for raw_name, raw_value in headers.items():
        name, name_lower = _normalize_header_name(raw_name)
        value = _to_text(raw_value)

        if name_lower == "content-type":
            has_content_type = True
        elif name_lower == "content-length":
            has_content_length = True
        elif name_lower == "content-encoding":
            has_content_encoding = True
        elif name_lower == "user-agent":
            has_user_agent = True

        if name_lower == "authorization":
            sanitized = _sanitize_authorization_value(
                value,
                mode=mode,
                anonymize=anonymize,
            )
        elif name_lower in SENSITIVE_HEADER_KEYS:
            sanitized = _sanitize_sensitive_value(
                value,
                mode=mode,
                anonymize=anonymize,
            )
        elif _looks_like_token(value):
            sanitized = _sanitize_sensitive_value(
                value,
                mode=mode,
                anonymize=anonymize,
            )
        else:
            sanitized = value

        if sanitized is not None:
            out[name] = sanitized

    if body is not None:
        media_type: Optional[MediaType] = None

        if not has_content_type:
            media_type = media_type or body.media_type
            out["Content-Type"] = media_type.full_mime_type(concat_codec=False).value

            if not has_content_encoding:
                codec: Optional[Codec] = media_type.codec
                if codec is not None:
                    out["Content-Encoding"] = codec.name

        if not has_content_length:
            out["Content-Length"] = str(body.size)

    if is_request:
        current = UserInfo.current()

        pv = current.product_version or "0.0.0"

        if current.product:
            out["X-Product"] = current.product
            out["X-Product-Version"] = pv

        if not has_user_agent:
            if current.product:
                out["User-Agent"] = f"{current.product}/{pv} {DEFAULT_USER_AGENT}"
            else:
                out["User-Agent"] = DEFAULT_USER_AGENT

        if current.git_url:
            out["X-Git-Url"] = current.git_url.to_string()

        out["Host"] = out.get("Host", DEFAULT_HOSTNAME)

        out["X-Ygg-Version"] = __version__
        out["X-Py-Version"] = PYVERSION

    return out