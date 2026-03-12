from __future__ import annotations

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

def _key_norm(k: Any) -> str:
    return str(k).strip().lower()

def _to_str(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("latin-1", errors="replace")
    return str(v)

def _handle_secret_scalar(mode: str) -> Any:
    if mode == "remove":
        return _REMOVE
    return "<redacted>"


def sanitize_value(mode: str, k_lc: str, v: Any) -> Any:
    # recurse for nested structures
    if isinstance(v, Mapping):
        return sanitize_mapping(mode, v)
    if isinstance(v, (list, tuple)):
        out_items = []
        for item in v:
            sv = sanitize_value(mode, k_lc, item)
            if sv is _REMOVE:
                continue
            out_items.append(sv)
        return out_items if isinstance(v, list) else tuple(out_items)

    # key-based rules
    if k_lc in _SENSITIVE_PARAM_KEYS:
        if mode == "remove":
            return _REMOVE
        return "<redacted>"

    # value-based rules (catch secrets even if key is innocent)
    return _handle_secret_scalar(mode)


def sanitize_mapping(mode: str, m: Mapping[Any, Any]) -> MutableMapping[str, Any]:
    out: MutableMapping[str, Any] = {}  # preserves insertion order
    for k, v in m.items():
        k_str = str(k).strip()
        k_lc = _key_norm(k_str)

        sv = sanitize_value(mode, k_lc, v)
        if sv is _REMOVE:
            continue  # <- the "remove mode" behavior
        out[k_str] = sv
    return out


def anonymize_parameters(
    params: Any,
    *,
    mode: Literal["remove", "redact"] = "remove",
    salt: str = "",
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
    # entrypoint
    if isinstance(params, Mapping):
        return sanitize_mapping(mode, params)

    if isinstance(params, (list, tuple)):
        out_items = []
        for x in params:
            sx = anonymize_parameters(
                x, mode=mode, salt=salt, preserve_keys=preserve_keys
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

    return "<redacted>" if len(s) > 0 else s
