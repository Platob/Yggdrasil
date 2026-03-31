from __future__ import annotations

from yggdrasil.rust_accel import HAS_RUST_ACCEL, utf8_len


def test_utf8_len_counts_unicode_characters() -> None:
    assert utf8_len(["abc", "mañana", "🙂🙂", None]) == [3, 6, 2, None]


def test_has_rust_accel_is_boolean() -> None:
    assert isinstance(HAS_RUST_ACCEL, bool)
