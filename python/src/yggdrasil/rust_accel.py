from __future__ import annotations

from typing import Iterable

__all__ = ["HAS_RUST_ACCEL", "utf8_len"]


try:
    from yggdrasil_rust import utf8_len as _utf8_len_impl

    HAS_RUST_ACCEL = True
except Exception:
    HAS_RUST_ACCEL = False

    def _utf8_len_impl(values: list[str | None]) -> list[int | None]:
        return [None if value is None else len(value) for value in values]


def utf8_len(values: Iterable[str | None]) -> list[int | None]:
    """Return character counts for UTF-8 strings with an optional Rust fast path.

    This helper is intentionally tiny so we can benchmark and expand Rust-backed
    acceleration without forcing a hard runtime dependency.
    """
    return _utf8_len_impl(list(values))
