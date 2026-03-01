from __future__ import annotations

import hashlib
import re
from typing import Mapping, MutableMapping, Union, Literal

__all__ = [
    "HeaderValue",
    "anonymize_headers"
]


HeaderValue = Union[str, bytes]


_SENSITIVE_KEYS = {
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

_BEARER_RE = re.compile(r"^\s*Bearer\s+(.+)\s*$", re.IGNORECASE)
_BASIC_RE = re.compile(r"^\s*Basic\s+(.+)\s*$", re.IGNORECASE)

def _to_text(v: HeaderValue) -> str:
    if isinstance(v, bytes):
        return v.decode("latin-1", errors="replace")
    return str(v)

def _hash(value: str, salt: str = "", algo: str = "blake2s", digest_size: int = 16) -> str:
    if algo == "blake2s":
        h = hashlib.blake2s(digest_size=digest_size)
        if salt:
            h.update(salt.encode("utf-8"))
        h.update(value.encode("utf-8", errors="replace"))
        return h.hexdigest()
    # fallback
    return hashlib.sha256((salt + value).encode("utf-8", errors="replace")).hexdigest()

def _mask_ip_like(s: str) -> str:
    # super light masking: keep shape, drop precision
    # IPv4: 1.2.3.4 -> 1.2.x.x
    s = re.sub(r"\b(\d{1,3}\.\d{1,3})\.\d{1,3}\.\d{1,3}\b", r"\1.x.x", s)
    # IPv6: keep first 2 hextets
    s = re.sub(r"\b([0-9a-fA-F]{0,4}:[0-9a-fA-F]{0,4}):[0-9a-fA-F:]+\b", r"\1:xxxx:xxxx:xxxx:xxxx", s)
    return s

def anonymize_headers(
    headers: Mapping[HeaderValue, HeaderValue],
    *,
    mode: Literal["remove", "redact", "hash"] = "remove",
    salt: str = "",
    keep_content_type: bool = True,
    keep_accept: bool = True,
    keep_host: bool = False,
    preserve_keys: bool = True,
) -> MutableMapping[str, str]:
    """
    Returns a sanitized copy of headers. Use mode="hash" to keep stable fingerprints.

    - remove: drops sensitive headers entirely
    - redact: replaces sensitive values with "<redacted>"
    - hash: replaces sensitive values with "<hash:...>" so you can correlate safely
    """
    out: MutableMapping[str, str] = {}  # keeps insertion order in py3.7+

    def _emit(out_key: str, value: str) -> None:
        out[out_key] = value

    def _handle_sensitive_value(out_key: str, raw_value: str) -> None:
        if mode == "remove":
            return
        if mode == "hash":
            _emit(out_key, f"<hash:{_hash(raw_value, salt=salt)}>")
        else:
            _emit(out_key, "<redacted>")

    for k_raw, v_raw in dict(headers).items():
        k = _to_text(k_raw)
        v = _to_text(v_raw)

        k_norm = k.strip()
        k_lc = k_norm.lower()

        out_key = k_norm if preserve_keys else f"h:{_hash(k_norm, salt=salt)}"

        # Allow some common safe headers to pass (optionally)
        if (
            (k_lc == "content-type" and keep_content_type)
            or (k_lc == "accept" and keep_accept)
            or (k_lc == "host" and keep_host)
        ):
            _emit(out_key, v)
            continue

        # Hard sensitive headers
        if k_lc in _SENSITIVE_KEYS:
            _handle_sensitive_value(out_key, v)
            continue

        # Authorization patterns even if header name isn’t in list
        if k_lc == "authorization":
            m = _BEARER_RE.match(v)
            if m:
                if mode == "remove":
                    continue
                token = m.group(1)
                if mode == "hash":
                    _emit(out_key, f"Bearer <hash:{_hash(token, salt=salt)}>")
                else:
                    _emit(out_key, "Bearer <redacted>")
                continue

            m = _BASIC_RE.match(v)
            if m:
                if mode == "remove":
                    continue
                creds = m.group(1)
                if mode == "hash":
                    _emit(out_key, f"Basic <hash:{_hash(creds, salt=salt)}>")
                else:
                    _emit(out_key, "Basic <redacted>")
                continue

            # Unknown auth scheme -> treat as sensitive
            _handle_sensitive_value(out_key, v)
            continue

        # Generic token-ish detection: redact/hash/remove
        if len(v) >= 40 and re.search(r"[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+", v):
            _handle_sensitive_value(out_key, v)
            continue

        # Otherwise keep as-is
        _emit(out_key, v)

    return out
