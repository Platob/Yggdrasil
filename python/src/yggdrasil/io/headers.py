from __future__ import annotations

import hashlib
import re
from typing import Mapping, MutableMapping, Union

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

# Often sensitive, but sometimes useful. We’ll partially mask.
_PARTIAL_KEYS = {
    "user-agent",
    "referer",
    "origin",
    "x-forwarded-for",
    "forwarded",
    "x-real-ip",
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
    mode: str = "redact",           # "redact" | "hash"
    salt: str = "",                 # used when mode="hash"
    keep_content_type: bool = True,
    keep_accept: bool = True,
    keep_host: bool = True,
    preserve_keys: bool = True,     # if False, hash header names too (rarely needed)
) -> MutableMapping[str, str]:
    """
    Returns a sanitized copy of headers. Use mode="hash" to keep stable fingerprints.

    - redact: replaces sensitive values with "<redacted>"
    - hash: replaces sensitive values with "<hash:...>" so you can correlate safely
    """
    out: MutableMapping[str, str] = {}  # keeps insertion order in py3.7+

    for k_raw, v_raw in headers.items():
        k = _to_text(k_raw)
        v = _to_text(v_raw)

        k_norm = k.strip()
        k_lc = k_norm.lower()

        out_key = k_norm if preserve_keys else f"h:{_hash(k_norm, salt=salt)}"

        # Allow some common safe headers to pass (optionally)
        if (k_lc == "content-type" and keep_content_type) or (k_lc == "accept" and keep_accept) or (k_lc == "host" and keep_host):
            out[out_key] = v
            continue

        # Hard sensitive headers: full wipe/hash
        if k_lc in _SENSITIVE_KEYS:
            if mode == "hash":
                out[out_key] = f"<hash:{_hash(v, salt=salt)}>"
            else:
                out[out_key] = "<redacted>"
            continue

        # Authorization patterns even if header name isn’t in list
        if k_lc == "authorization":
            m = _BEARER_RE.match(v)
            if m:
                token = m.group(1)
                out[out_key] = f"Bearer <{'hash:'+_hash(token, salt=salt) if mode=='hash' else 'redacted'}>"
                continue
            m = _BASIC_RE.match(v)
            if m:
                creds = m.group(1)
                out[out_key] = f"Basic <{'hash:'+_hash(creds, salt=salt) if mode=='hash' else 'redacted'}>"
                continue

        # Partially sensitive: mask a bit (IPs, full URLs, UA entropy)
        if k_lc in _PARTIAL_KEYS:
            vv = _mask_ip_like(v)

            # Trim high-entropy user-agent-ish strings while keeping product tokens
            if k_lc == "user-agent":
                # keep first ~60 chars; enough for “Chrome vs curl” debugging
                vv = vv[:60] + ("…" if len(vv) > 60 else "")
            # Drop query params from referer/origin-ish
            if k_lc in {"referer", "origin"}:
                vv = vv.split("?", 1)[0]

            out[out_key] = vv
            continue

        # Generic token-ish detection: if value looks like a long JWT-ish blob, redact/hash
        if len(v) >= 40 and re.search(r"[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+", v):
            out[out_key] = f"<hash:{_hash(v, salt=salt)}>" if mode == "hash" else "<redacted>"
            continue

        # Otherwise keep as-is
        out[out_key] = v

    return out
