"""Case-insensitive, multi-value HTTP header dict — :class:`HTTPHeaderDict`.

The transport surface (raw socket writes, retry-after parsing, the
urllib3-shim error path in :mod:`yggdrasil.exceptions.http`) speaks the
urllib3-shaped :class:`HTTPHeaderDict`: same name, same multi-value
semantics, lowercase-keyed storage with first-seen original casing
preserved on iteration. The high-level
:class:`yggdrasil.io.headers.Headers` is a different abstraction
(normalised, anonymisation-aware, hash-stable) and isn't a drop-in
replacement at the transport layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import collections.abc
from typing import Any, ClassVar, Iterator, Mapping, MutableMapping, Optional, Tuple, Union
import hashlib
import re
import platform
import os
import socket
from yggdrasil.version import __version_info__


__all__ = ["HTTPHeaderDict"]


class HTTPHeaderDict(collections.abc.MutableMapping):
    """Case-insensitive, multi-value header dict mirroring ``urllib3``'s.

    Stores values per lowercase key but preserves the first-seen original
    casing when iterating. Multi-value headers (Set-Cookie, …) are joined
    with ``, `` on read, matching urllib3's collapsing behavior.
    """

    def __init__(self, headers: Any = None, **kwargs: str) -> None:
        # _store maps lowercase key -> (original_case, [values])
        self._store: dict[str, Tuple[str, list[str]]] = {}
        if headers is not None:
            self.extend(headers)
        if kwargs:
            self.extend(kwargs)

    # MutableMapping protocol -------------------------------------------------
    def __setitem__(self, key: str, value: str) -> None:
        self._store[key.lower()] = (key, [value])

    def __getitem__(self, key: str) -> str:
        _, values = self._store[key.lower()]
        return ", ".join(values)

    def __delitem__(self, key: str) -> None:
        del self._store[key.lower()]

    def __iter__(self) -> Iterator[str]:
        return (original for original, _ in self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.lower() in self._store

    # Multi-value helpers -----------------------------------------------------
    def add(self, key: str, value: str) -> None:
        slot = self._store.get(key.lower())
        if slot is None:
            self._store[key.lower()] = (key, [value])
        else:
            slot[1].append(value)

    def extend(self, other: Any) -> None:
        if isinstance(other, HTTPHeaderDict):
            for original, values in other._store.values():
                for v in values:
                    self.add(original, v)
            return
        if hasattr(other, "items"):
            other = other.items()
        for k, v in other:
            self.add(k, v)

    def getlist(self, key: str) -> list[str]:
        slot = self._store.get(key.lower())
        return list(slot[1]) if slot is not None else []

    def __repr__(self) -> str:
        return f"HTTPHeaderDict({dict(self.items())!r})"
# yggdrasil.io.headers
__all__ = [
    "Headers",
    "HeaderValue",
    "PromotedHeaders",
    "normalize_headers",
    "DEFAULT_HOSTNAME",
    "get_default_user_agent",
    "get_default_headers",
]

HeaderValue = Union[str, bytes]


PYVERSION = str(platform.python_version())
DEFAULT_USER_AGENT: str = ""

try:
    DEFAULT_HOSTNAME = socket.gethostname()
except Exception:
    DEFAULT_HOSTNAME = "localhost"

DEFAULT_HEADERS = {}

def get_default_headers() -> dict[str, str]:
    global DEFAULT_HEADERS

    if not DEFAULT_HEADERS:
        DEFAULT_HEADERS = {
            "X-Ygg-Version": __version__,
            "X-Py-Version": PYVERSION,
            "X-Host": DEFAULT_HOSTNAME,
        }

        from yggdrasil.environ.userinfo import UserInfo

        current = UserInfo.current()
        pv = current.product_version or "0.0.0"

        if current.product:
            DEFAULT_HEADERS["X-Product"] = current.product
            DEFAULT_HEADERS["X-Product-Version"] = pv

        if current.git_url:
            DEFAULT_HEADERS["X-Git-Url"] = current.git_url.to_string()

    return DEFAULT_HEADERS


def get_default_user_agent() -> str:
    global DEFAULT_USER_AGENT

    if not DEFAULT_USER_AGENT:
        from yggdrasil.environ.userinfo import UserInfo
        current = UserInfo.current()

        DEFAULT_USER_AGENT = (
            f"yggdrasil/{__version_info__} "
            f"os/{platform.system().lower()} "
            f"py/{PYVERSION}"
        )

        if current.product:
            DEFAULT_USER_AGENT = f"{current.product}/{current.product_version or '0.0.0'} {DEFAULT_USER_AGENT}"

    return DEFAULT_USER_AGENT


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
    "content-disposition": "Content-Disposition",
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


@dataclass
class PromotedHeaders:
    """
    Common HTTP headers extracted into dedicated typed fields.

    `remaining` contains all non-promoted headers after normalization.
    """

    host: str | None = None
    user_agent: str | None = None
    accept: str | None = None
    accept_encoding: str | None = None
    accept_language: str | None = None
    content_type: str | None = None
    content_length: int = 0
    content_encoding: str | None = None
    transfer_encoding: str | None = None
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
    }

    @classmethod
    def extract(
        cls,
        headers: Mapping[HeaderValue, HeaderValue],
        *,
        normalize: bool = True,
        host: str | None = None,
    ) -> "PromotedHeaders":
        """
        Extract common headers into typed attributes.

        Matching is case-insensitive. When `normalize=True`, recognized header
        names are canonicalized in `remaining`.
        """
        normalized: dict[str, str] = {}
        lower_to_actual: dict[str, str] = {}

        if host:
            normalized["Host"] = host
            lower_to_actual["host"] = "Host"

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

        built = cls(**kwargs)

        if not built.content_length:
            object.__setattr__(built, "content_length", 0)

        return built

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
        }


def _sanitize_sensitive_value(
    value: str,
    *,
    mode: Literal["remove", "redact"],
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
    mode: Literal["remove", "redact"],
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
    headers: "Headers | Mapping[HeaderValue, HeaderValue] | None",
    *,
    is_request: bool,
    add_missing: bool = True,
    mode: Literal["remove", "redact"] = "remove",
    anonymize: bool = False,
    body: Optional[Holder] = None,
) -> "Headers":
    """Backwards-compatible thin wrapper around :meth:`Headers.normalized`.

    The whole normalization vocabulary (canonical names, sensitive
    detection, body-derived Content-* backfill, request defaults)
    lives on :class:`Headers` so a single audit covers the whole
    behaviour. This free function stays so existing callers
    (``normalize_headers(some_dict, is_request=True)``) keep working
    without rewriting every site.
    """
    return Headers.from_(headers).normalized(
        is_request=is_request,
        add_missing=add_missing,
        mode=mode,
        anonymize=anonymize,
        body=body,
    )


# Sentinel used by :class:`Headers` to tell "key absent" from "key
# present with the same value" without paying for an extra ``in``
# probe. ``object()`` would work; ``...`` is already the codebase's
# blessed missing-arg singleton (see CLAUDE.md → "Use `...` as the
# unset / missing sentinel").
_MISSING: Any = ...


class HTTPHeaders(MutableMapping[str, str]):
    """Mutable string-string mapping with versioned change tracking.

    Acts as a drop-in replacement for the plain ``dict[str, str]``
    that :class:`Session`, :class:`PreparedRequest`, and
    :class:`Response` were using. Two reasons to specialize it:

    - **Version counter.** Every mutation that actually changes the
      contents bumps :attr:`version`. ``(id(headers), version)`` is a
      tight cache fingerprint — ``id`` covers wholesale dict swaps,
      ``version`` covers in-place mutations the ``id`` can't see (the
      common case: ``headers["Authorization"] = new_token`` on the
      same dict object).
    - **Cached digests.** :attr:`byte_length` and :attr:`xxh3_64`
      memoize against the version, so a request that hashes the same
      headers ten times pays the walk once. Same-value writes
      (``headers["Accept"] = "*/*"`` when it already is) short-circuit
      without bumping the version, so re-applying defaults is free.

    Keys and values are coerced to ``str`` on the way in. Lookups stay
    case-sensitive — the canonical-name normalization sits in
    :func:`normalize_headers`, separate from this container, so
    callers that already speak in canonical names don't pay for a
    second pass.
    """

    __slots__ = (
        "_data",
        "_version",
        "_byte_length",
        "_byte_length_version",
        "_xxh3_64",
        "_xxh3_64_version",
        "_canonical_bytes",
        "_canonical_bytes_version",
        "_anonymized_cache",
        "_anonymized_cache_version",
    )

    def __init__(
        self,
        data: "Headers | Mapping[Any, Any] | Iterable[tuple[Any, Any]] | None" = None,
    ) -> None:
        # ``_version`` starts at 0; the cache slots use ``-1`` so the
        # first read is always a miss even when no mutation has fired.
        self._version: int = 0
        self._byte_length: int = 0
        self._byte_length_version: int = -1
        self._xxh3_64: int = 0
        self._xxh3_64_version: int = -1
        self._canonical_bytes: bytes = b""
        self._canonical_bytes_version: int = -1
        # Lazy: most Headers instances never have :meth:`anonymized`
        # called on them, so don't pay the per-construct dict alloc.
        self._anonymized_cache: dict[str, "Headers"] | None = None
        self._anonymized_cache_version: int = -1
        if data is None:
            self._data = {}
            return
        if isinstance(data, Headers):
            # Already string-keyed string-valued — copy in one shot.
            self._data = dict(data._data)
            return
        if isinstance(data, dict):
            # Hot path: most callers pass a ``dict[str, str]`` literal.
            # Trust the shape and copy directly; fall back to per-item
            # ``str()`` coercion only when something diverges.
            for k, v in data.items():
                if type(k) is not str or type(v) is not str:
                    self._data = {str(k): str(v) for k, v in data.items()}
                    return
            self._data = dict(data)
            return
        if isinstance(data, Mapping):
            self._data = {str(k): str(v) for k, v in data.items()}
            return
        self._data = {str(k): str(v) for k, v in data}

    @classmethod
    def from_(
        cls,
        arg: "Headers | Mapping[Any, Any] | Iterable[tuple[Any, Any]] | None" = None,
    ) -> "Headers":
        """Coerce *arg* to :class:`Headers` — passing an existing
        instance through unchanged so callers can ``Headers.from_(x)``
        regardless of what ``x`` is."""
        if isinstance(arg, cls):
            return arg
        return cls(arg)

    # ------------------------------------------------------------------
    # Mapping / MutableMapping protocol
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        sk = str(key)
        sv = str(value)
        existing = self._data.get(sk, _MISSING)
        if existing is not _MISSING and existing == sv:
            return
        self._data[sk] = sv
        self._version += 1

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self._version += 1

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Headers):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def __repr__(self) -> str:
        return f"Headers({self._data!r})"

    def __bool__(self) -> bool:
        return bool(self._data)

    # MutableMapping mixins (``pop``, ``popitem``, ``setdefault``,
    # ``__contains__``, ``keys``, ``items``, ``values``, ``get``,
    # ``__eq__``) all route through the four primitives above —
    # ``setdefault`` reaches ``__setitem__``, ``pop`` reaches
    # ``__delitem__``, etc. — so version bumps stay consistent without
    # us writing each method out by hand.

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Bulk update — same shape as :meth:`dict.update`.

        We override the mixin so the version bumps once per actual
        change rather than once per touched key, and so the no-op
        early return matches :meth:`__setitem__`.
        """
        if not args and not kwargs:
            return
        if len(args) > 1:
            raise TypeError(
                f"update expected at most 1 positional argument, got {len(args)}"
            )
        merged: dict[str, str] = {}
        if args:
            other = args[0]
            if isinstance(other, Headers):
                merged.update(other._data)
            elif isinstance(other, Mapping):
                for k, v in other.items():
                    merged[str(k)] = str(v)
            else:
                for k, v in other:
                    merged[str(k)] = str(v)
        for k, v in kwargs.items():
            merged[str(k)] = str(v)
        for k, v in merged.items():
            existing = self._data.get(k, _MISSING)
            if existing is not _MISSING and existing == v:
                continue
            self._data[k] = v
            self._version += 1

    def clear(self) -> None:
        if not self._data:
            return
        self._data.clear()
        self._version += 1

    def copy(self) -> "Headers":
        """Shallow copy as a fresh :class:`Headers` (version reset).

        Bypasses ``__init__``'s per-item ``str()`` coercion — we already
        know the source's keys / values are strings and the rest of the
        cache slots start fresh.
        """
        new = Headers.__new__(Headers)
        new._data = dict(self._data)
        new._version = 0
        new._byte_length = 0
        new._byte_length_version = -1
        new._xxh3_64 = 0
        new._xxh3_64_version = -1
        new._canonical_bytes = b""
        new._canonical_bytes_version = -1
        new._anonymized_cache = None
        new._anonymized_cache_version = -1
        return new

    def to_dict(self) -> dict[str, str]:
        """Snapshot as a plain ``dict`` — handy for code that needs
        to mutate independently of the live container or hand the
        contents to an API that only speaks ``dict``."""
        return dict(self._data)

    # ------------------------------------------------------------------
    # Normalization / anonymization — centralized header vocabulary
    # ------------------------------------------------------------------

    @staticmethod
    def canonical_name(name: HeaderValue) -> str:
        """Canonical casing for *name* (e.g. ``content-type`` →
        ``Content-Type``). Falls back to the caller's stripped form
        when the header isn't in the registry."""
        return _normalize_header_name(name)[0]

    def anonymized(self, mode: Literal["remove", "redact"] = "remove") -> "Headers":
        """Return a copy with sensitive values dropped/redacted.

        Authorization scheme (``Bearer``, ``Basic``, …) is preserved
        in ``redact`` mode; long token-shaped values are caught via
        :func:`_looks_like_token` so unrecognized credential headers
        still get sanitized. Names are canonicalized on the way out
        so repeated normalize calls are idempotent.

        Memoized against :attr:`version` — the canonical anonymized
        form is hot on every request's ``public_hash`` /
        ``public_url_hash`` computation, and per response's
        ``public_hash``. A version bump on the source headers
        invalidates the cache transparently.
        """
        cache = self._anonymized_cache
        if self._anonymized_cache_version == self._version and cache is not None:
            cached = cache.get(mode)
            if cached is not None:
                return cached

        out = Headers()
        for raw_name, raw_value in self._data.items():
            name, name_lower = _normalize_header_name(raw_name)
            value = _to_text(raw_value)
            if name_lower == "authorization":
                sanitized = _sanitize_authorization_value(value, mode=mode, anonymize=True)
            elif name_lower in SENSITIVE_HEADER_KEYS or _looks_like_token(value):
                sanitized = _sanitize_sensitive_value(value, mode=mode, anonymize=True)
            else:
                sanitized = value
            if sanitized is not None:
                out[name] = sanitized
        if cache is None or self._anonymized_cache_version != self._version:
            cache = {}
            self._anonymized_cache = cache
            self._anonymized_cache_version = self._version
        cache[mode] = out
        return out

    def normalized(
        self,
        *,
        is_request: bool,
        add_missing: bool = True,
        mode: Literal["remove", "redact"] = "remove",
        anonymize: bool = False,
        body: "Optional[Holder]" = None,
    ) -> "Headers":
        """Return a fresh :class:`Headers` with names canonicalized,
        sensitive values optionally sanitized, and (when
        ``add_missing``) request defaults / body-derived ``Content-*``
        slots filled in.

        This is the canonical normalize surface — :func:`normalize_headers`
        is a thin wrapper that calls into this method so every
        canonicalization site goes through one piece of code.
        """
        out = Headers()

        has_content_type = False
        has_content_length = False
        has_content_encoding = False
        has_user_agent = False
        has_host = False
        accept_value = ""
        accept_encoding_value = ""

        for raw_name, raw_value in self._data.items():
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
            elif name_lower == "host":
                has_host = True
            elif name_lower == "accept":
                accept_value = value
            elif name_lower == "accept-encoding":
                accept_encoding_value = value

            if name_lower == "authorization":
                sanitized = _sanitize_authorization_value(value, mode=mode, anonymize=anonymize)
            elif name_lower in SENSITIVE_HEADER_KEYS or _looks_like_token(value):
                sanitized = _sanitize_sensitive_value(value, mode=mode, anonymize=anonymize)
            else:
                sanitized = value

            if sanitized is not None:
                out[name] = sanitized

        if add_missing:
            if body is not None:
                if not has_content_type:
                    media_type = body.media_type
                    if media_type is not None:
                        out["Content-Type"] = media_type.full_mime_type(concat_codec=False).value
                        if not has_content_encoding:
                            codec = media_type.codec
                            if codec is not None:
                                out["Content-Encoding"] = codec.name
                if not has_content_length:
                    out["Content-Length"] = str(body.size)

            if is_request:
                if not has_user_agent:
                    out["User-Agent"] = get_default_user_agent()

                out.update(DEFAULT_HEADERS)

                # Accept-Encoding per RFC 7231 §5.3.4 is a
                # comma-separated preference list with optional
                # q-values; ``Codec.from_(default=None)`` only
                # rewrites when the value is a single recognized
                # codec, otherwise the caller's verbatim value (still
                # populated above) is left alone.
                single_accept_encoding_codec = (
                    Codec.from_(accept_encoding_value, default=None)
                    if accept_encoding_value
                    else None
                )

                if accept_value:
                    media_type = MediaType.from_(accept_value, codec=single_accept_encoding_codec)
                    out["Accept"] = (
                        "*/*"
                        if media_type.mime_type == MimeTypes.OCTET_STREAM
                        else media_type.mime_type.value
                    )
                    if media_type.codec and single_accept_encoding_codec is not None:
                        out["Accept-Encoding"] = single_accept_encoding_codec.name
                elif accept_encoding_value:
                    out["Accept"] = "*/*"
                    if single_accept_encoding_codec is not None:
                        out["Accept-Encoding"] = single_accept_encoding_codec.name
                else:
                    out["Accept"] = "*/*"

        if has_host:
            del out["Host"]

        return out

    # ------------------------------------------------------------------
    # Versioned derivations — cached against ``_version``
    # ------------------------------------------------------------------

    @property
    def version(self) -> int:
        """Monotonically increasing counter — bumped on every mutation
        that actually changes :meth:`__eq__`. Callers can stash
        ``(id(headers), version)`` and re-check it later for an O(1)
        "did anything change?" test."""
        return self._version

    @property
    def byte_length(self) -> int:
        """Total byte length of all keys + values. Memoized — pays
        the walk once per :attr:`version` and serves an int the rest
        of the time."""
        if self._byte_length_version == self._version:
            return self._byte_length
        length = 0
        for k, v in self._data.items():
            length += len(k) + len(v)
        self._byte_length = length
        self._byte_length_version = self._version
        return length

    @property
    def canonical_bytes(self) -> bytes:
        """Sorted ``key=value\\x00key=value\\x00…`` byte sequence —
        the canonical wire form used by digest mixing. Order is
        deterministic (sorted by key) so ``Headers({A:1, B:2})`` and
        ``Headers({B:2, A:1})`` produce the same payload. Memoized
        against :attr:`version` so each request pays the encode walk
        once across :attr:`xxh3_64`, :attr:`PreparedRequest.hash`,
        and :attr:`Response.hash`.
        """
        if self._canonical_bytes_version == self._version:
            return self._canonical_bytes
        if not self._data:
            self._canonical_bytes = b""
            self._canonical_bytes_version = self._version
            return b""
        parts: list[bytes] = []
        for k in sorted(self._data):
            parts.append(k.encode("utf-8"))
            parts.append(b"=")
            parts.append(self._data[k].encode("utf-8"))
            parts.append(b"\x00")
        self._canonical_bytes = b"".join(parts)
        self._canonical_bytes_version = self._version
        return self._canonical_bytes

    @property
    def xxh3_64(self) -> int:
        """xxh3_64 digest over :attr:`canonical_bytes` — order-
        independent (same pairs → same digest). Returns a signed
        int64 to match the rest of the codebase's :func:`xxhash`
        use. Memoized against :attr:`version`."""
        if self._xxh3_64_version == self._version:
            return self._xxh3_64
        payload = self.canonical_bytes
        if not payload:
            self._xxh3_64 = 0
            self._xxh3_64_version = self._version
            return 0
        import xxhash
        u = xxhash.xxh3_64(payload).intdigest()
        signed = u if u < 2**63 else u - 2**64
        self._xxh3_64 = signed
        self._xxh3_64_version = self._version
        return signed

    # ------------------------------------------------------------------
    # Pickle / copy support
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        # Drop the cached digest slots — they re-derive cheaply on
        # demand and shipping them across the pickle wire is just
        # bytes for nothing. Carry the version forward so callers
        # holding a pre-pickle ``(id, version)`` tuple don't see a
        # phantom rewind on the receiving side.
        return {"data": dict(self._data), "version": self._version}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._data = dict(state.get("data", {}))
        self._version = int(state.get("version", 0))
        self._byte_length = 0
        self._byte_length_version = -1
        self._xxh3_64 = 0
        self._xxh3_64_version = -1
        self._canonical_bytes = b""
        self._canonical_bytes_version = -1
        self._anonymized_cache = None
        self._anonymized_cache_version = -1
Headers = HTTPHeaders
