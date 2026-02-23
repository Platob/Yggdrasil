from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, MutableMapping, Literal

__all__ = [
    "anonymize_parameters"
]

_REMOVE = object()  # internal sentinel

_SENSITIVE_PARAM_KEYS = {
    "password", "pass", "pwd",
    "token", "access_token", "refresh_token", "id_token",
    "api_key", "apikey", "x_api_key",
    "secret", "client_secret",
    "authorization", "auth", "bearer",
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
    mode: Literal["remove", "redact", "hash"] = "remove",
    salt: str = "",
    max_str_len: int = 200,
    preserve_keys: bool = True,
) -> Any:
    """
    Anonymize parameters in nested structures (dict/list/tuple).

    - remove: drops sensitive/identifier keys from mappings; filters secret-ish items from sequences
    - redact: replaces sensitive values with "<redacted>"
    - hash: replaces sensitive values with "<hash:...>" for safe correlation

    Also:
    - token/JWT/email/phone-ish values => redact/hash/remove (depending on mode)
    - Truncates very long strings to keep logs sane
    """

    def _maybe_trim(s: str) -> str:
        return (s[:max_str_len] + "…") if len(s) > max_str_len else s

    def _handle_secret_scalar(s: str) -> Any:
        # value-based secret detection
        secretish = (
            bool(_JWT_RE.search(s))
            or bool(_HEX_RE.match(s))
            or bool(_BASE64ISH_RE.match(s))
            or bool(_EMAIL_RE.search(s))
            or bool(_PHONE_RE.search(s))
        )
        if not secretish:
            return _maybe_trim(s)

        if mode == "remove":
            return _REMOVE
        if mode == "hash":
            return f"<hash:{_hash(s, salt=salt)}>"
        return "<redacted>"

    def sanitize_value(k_lc: str, v: Any) -> Any:
        # recurse for nested structures
        if isinstance(v, Mapping):
            return sanitize_mapping(v)
        if isinstance(v, (list, tuple)):
            out_items = []
            for item in v:
                sv = sanitize_value(k_lc, item)
                if sv is _REMOVE:
                    continue
                out_items.append(sv)
            return out_items if isinstance(v, list) else tuple(out_items)

        # scalar
        s = _to_str(v)

        # key-based rules
        if k_lc in _SENSITIVE_PARAM_KEYS:
            if mode == "remove":
                return _REMOVE
            if mode == "hash":
                return f"<hash:{_hash(s, salt=salt)}>"
            return "<redacted>"

        # value-based rules (catch secrets even if key is innocent)
        return _handle_secret_scalar(s)

    def sanitize_mapping(m: Mapping[Any, Any]) -> MutableMapping[str, Any]:
        out: MutableMapping[str, Any] = {}  # preserves insertion order
        for k, v in m.items():
            k_str = str(k).strip()
            k_lc = _key_norm(k_str)
            out_k = k_str if preserve_keys else f"p:{_hash(k_str, salt=salt)}"

            sv = sanitize_value(k_lc, v)
            if sv is _REMOVE:
                continue  # <- the "remove mode" behavior
            out[out_k] = sv
        return out

    # entrypoint
    if isinstance(params, Mapping):
        return sanitize_mapping(params)

    if isinstance(params, (list, tuple)):
        out_items = []
        for x in params:
            sx = anonymize_parameters(
                x, mode=mode, salt=salt, max_str_len=max_str_len, preserve_keys=preserve_keys
            )
            if sx is _REMOVE:
                continue
            out_items.append(sx)
        return out_items if isinstance(params, list) else tuple(out_items)

    # scalar root
    s = _to_str(params)
    if mode == "remove":
        # only "remove" it if it looks secret-ish; otherwise keep trimmed
        v = _handle_secret_scalar(s)
        return None if v is _REMOVE else v

    if mode == "hash":
        return f"<hash:{_hash(s, salt=salt)}>"
    return "<redacted>" if len(s) > 0 else s
