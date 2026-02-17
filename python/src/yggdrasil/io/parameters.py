from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, MutableMapping

__all__ = [
    "anonymize_parameters"
]


_SENSITIVE_PARAM_KEYS = {
    "password", "pass", "pwd",
    "token", "access_token", "refresh_token", "id_token",
    "api_key", "apikey", "x_api_key",
    "secret", "client_secret",
    "authorization", "auth", "bearer",
    "session", "sessionid", "session_id",
    "cookie", "set_cookie",
    "email", "phone",
    "ssn", "social", "creditcard", "card", "pan", "cvv",
}

# keys that are often identifiers (safe-ish to hash for correlation)
_IDENTIFIER_KEYS = {
    "user", "user_id", "userid", "uid",
    "account", "account_id",
    "customer", "customer_id",
    "device_id", "installation_id",
}

# simple detectors for common sensitive shapes
_JWT_RE = re.compile(r"\b[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b")
_HEX_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{32,}$")
_BASE64ISH_RE = re.compile(r"^[A-Za-z0-9+/=_-]{32,}$")  # rough, but useful
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def _hash(value: str, salt: str = "", digest_size: int = 16) -> str:
    h = hashlib.blake2s(digest_size=digest_size)
    if salt:
        h.update(salt.encode("utf-8"))
    h.update(value.encode("utf-8", errors="replace"))
    return h.hexdigest()

def _key_norm(k: Any) -> str:
    return str(k).strip().lower()

def _to_str(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("latin-1", errors="replace")
    return str(v)

def anonymize_parameters(
    params: Any,
    *,
    mode: str = "redact",        # "redact" | "hash"
    salt: str = "",
    max_str_len: int = 200,      # avoid logging giant blobs
    preserve_keys: bool = True,  # if False, hash param names too
) -> Any:
    """
    Anonymize parameters in nested structures (dict/list/tuple).
    - Sensitive keys => redact or hash value
    - Identifier keys => prefer hash (even in redact mode, can keep as redacted if you want)
    - Token/JWT/email/phone-ish values => redact/hash
    - Truncates very long strings to keep logs sane
    """

    def sanitize_value(k_lc: str, v: Any) -> Any:
        # recurse for nested structures
        if isinstance(v, Mapping):
            return sanitize_mapping(v)
        if isinstance(v, (list, tuple)):
            out_seq = [sanitize_value(k_lc, item) for item in v]
            return out_seq if isinstance(v, list) else tuple(out_seq)

        # scalar
        s = _to_str(v)

        # trim huge values early
        if len(s) > max_str_len:
            s_trim = s[:max_str_len] + "…"
        else:
            s_trim = s

        # key-based rules
        if k_lc in _SENSITIVE_PARAM_KEYS:
            return f"<hash:{_hash(s, salt=salt)}>" if mode == "hash" else "<redacted>"

        if k_lc in _IDENTIFIER_KEYS:
            # identifiers are the classic “debugging needs correlation” case
            return f"<hash:{_hash(s, salt=salt)}>" if mode == "hash" else "<redacted>"

        # value-based rules (catch secrets even if key is innocent)
        if _JWT_RE.search(s) or _HEX_RE.match(s) or _BASE64ISH_RE.match(s):
            return f"<hash:{_hash(s, salt=salt)}>" if mode == "hash" else "<redacted>"

        if _EMAIL_RE.search(s):
            return f"<hash:{_hash(s, salt=salt)}>" if mode == "hash" else "<redacted>"

        if _PHONE_RE.search(s):
            return f"<hash:{_hash(s, salt=salt)}>" if mode == "hash" else "<redacted>"

        return s_trim

    def sanitize_mapping(m: Mapping[Any, Any]) -> MutableMapping[str, Any]:
        out: MutableMapping[str, Any] = {}  # preserves insertion order
        for k, v in m.items():
            k_str = str(k).strip()
            k_lc = _key_norm(k_str)
            out_k = k_str if preserve_keys else f"p:{_hash(k_str, salt=salt)}"
            out[out_k] = sanitize_value(k_lc, v)
        return out

    # entrypoint: preserve non-mapping structures too
    if isinstance(params, Mapping):
        return sanitize_mapping(params)
    if isinstance(params, (list, tuple)):
        out_seq = [anonymize_parameters(x, mode=mode, salt=salt, max_str_len=max_str_len, preserve_keys=preserve_keys) for x in params]
        return out_seq if isinstance(params, list) else tuple(out_seq)

    # scalar root
    s = _to_str(params)
    return f"<hash:{_hash(s, salt=salt)}>" if mode == "hash" else ("<redacted>" if len(s) > 0 else s)
