from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Literal

__all__ = ["anonymize_parameters"]

_REMOVE: Any = ...

_SENSITIVE_PARAM_KEYS = {
    "password", "pass", "pwd",
    "token", "access_token", "refresh_token", "id_token",
    "api_key", "apikey", "x_api_key",
    "secret", "client_secret",
    "authorization", "auth", "bearer",
}


def _key_norm(key: Any) -> str:
    return str(key).strip().lower()


def _secret_replacement(mode: Literal["remove", "redact"]) -> Any:
    return _REMOVE if mode == "remove" else "<redacted>"


def _sanitize_sequence(
    items: list | tuple,
    *,
    mode: Literal["remove", "redact"],
    sensitive: bool,
) -> list | tuple:
    out = [
        s for item in items
        if (s := _sanitize(item, mode=mode, sensitive=sensitive)) is not _REMOVE
    ]
    return tuple(out) if isinstance(items, tuple) else out


def _sanitize(value: Any, *, mode: Literal["remove", "redact"], sensitive: bool) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_mapping(value, mode=mode)

    if isinstance(value, (list, tuple)):
        return _sanitize_sequence(value, mode=mode, sensitive=sensitive)

    if sensitive:
        return _secret_replacement(mode)

    return value


def _sanitize_mapping(
    mapping: Mapping[Any, Any],
    *,
    mode: Literal["remove", "redact"],
) -> MutableMapping[str, Any]:
    out: MutableMapping[str, Any] = {}

    for key, value in mapping.items():
        key_str = str(key).strip()
        sensitive = _key_norm(key_str) in _SENSITIVE_PARAM_KEYS

        sanitized = _sanitize(value, mode=mode, sensitive=sensitive)
        if sanitized is not _REMOVE:
            out[key_str] = sanitized

    return out


def anonymize_parameters(
    params: Any,
    *,
    mode: Literal["remove", "redact"] = "remove",
) -> Any:
    if mode not in {"remove", "redact"}:
        raise ValueError(f"Unsupported mode: {mode!r}")

    if isinstance(params, Mapping):
        return _sanitize_mapping(params, mode=mode)

    if isinstance(params, (list, tuple)):
        return _sanitize_sequence(params, mode=mode, sensitive=False)

    return params
