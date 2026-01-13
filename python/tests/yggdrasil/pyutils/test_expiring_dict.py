# test_expiring_dict.py
# Run: pytest -q
#
# Assumes your class is in expiring_dict.py as:
#   from expiring_dict import ExpiringDict
#
# If your file/module name differs, update the import below.

import pytest

from yggdrasil.pyutils.expiring_dict import ExpiringDict


class FakeTime:
    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def make_dict(*, default_ttl=None, refresh_on_get=False, on_expire=None, thread_safe=False):
    ft = FakeTime()
    d = ExpiringDict(
        default_ttl=default_ttl,
        refresh_on_get=refresh_on_get,
        on_expire=on_expire,
        thread_safe=thread_safe,
    )
    # override internal clock (no sleeps, fully deterministic)
    d._now = ft.monotonic  # type: ignore[attr-defined]
    return d, ft


def test_set_get_before_expiry():
    d, ft = make_dict()
    d.set("a", 123, ttl=10.0)
    assert d["a"] == 123
    ft.advance(9.999)
    assert d["a"] == 123


def test_get_after_expiry_raises_keyerror_and_get_returns_default():
    d, ft = make_dict()
    d.set("a", 123, ttl=1.0)
    ft.advance(1.0)
    assert d.get("a") is None
    with pytest.raises(KeyError):
        _ = d["a"]


def test_ttl_none_means_use_default_ttl():
    d, ft = make_dict(default_ttl=5.0)
    d.set("a", 1)  # ttl=None -> default_ttl
    ft.advance(4.9)
    assert d["a"] == 1
    ft.advance(0.2)
    with pytest.raises(KeyError):
        _ = d["a"]


def test_no_expiration_when_default_ttl_is_none_and_ttl_is_none():
    d, ft = make_dict(default_ttl=None)
    d.set("a", 1, ttl=None)  # no expiration
    ft.advance(10_000)
    assert d["a"] == 1
    assert "a" in d


def test_ttl_zero_or_negative_deletes_key():
    d, _ = make_dict()
    d.set("a", 1, ttl=10.0)
    d.set("a", 2, ttl=0)
    assert d.get("a") is None
    d.set("b", 1, ttl=-5)
    assert d.get("b") is None


def test_dunder_setitem_uses_default_ttl():
    d, ft = make_dict(default_ttl=2.0)
    d["x"] = 42
    ft.advance(1.9)
    assert d["x"] == 42
    ft.advance(0.2)
    with pytest.raises(KeyError):
        _ = d["x"]


def test_contains_prunes_expired():
    d, ft = make_dict()
    d.set("a", 1, ttl=1.0)
    assert "a" in d
    ft.advance(1.0)
    assert "a" not in d
    assert len(d) == 0


def test_len_prunes_expired():
    d, ft = make_dict()
    d.set("a", 1, ttl=1.0)
    d.set("b", 2, ttl=10.0)
    assert len(d) == 2
    ft.advance(1.0)
    assert len(d) == 1
    assert "b" in d


def test_iteration_prunes_expired():
    d, ft = make_dict()
    d.set("a", 1, ttl=1.0)
    d.set("b", 2, ttl=10.0)
    ft.advance(1.0)
    keys = list(iter(d))
    assert keys == ["b"]


def test_items_keys_values_prune_expired():
    d, ft = make_dict()
    d.set("a", 1, ttl=1.0)
    d.set("b", 2, ttl=10.0)
    ft.advance(1.0)
    assert d.items() == [("b", 2)]
    assert d.keys() == ["b"]
    assert d.values() == [2]


def test_cleanup_returns_remaining_count():
    d, ft = make_dict()
    d.set("a", 1, ttl=1.0)
    d.set("b", 2, ttl=10.0)
    ft.advance(1.0)
    assert d.cleanup() == 1


def test_overwrite_creates_stale_heap_entries_but_works():
    d, ft = make_dict()
    d.set("a", 1, ttl=10.0)   # expires at 10
    d.set("a", 2, ttl=2.0)    # expires at 2 (new value)
    ft.advance(2.0)
    with pytest.raises(KeyError):
        _ = d["a"]

    # Advance to the old expiry and ensure no weird resurrection happens
    ft.advance(8.0)  # now 10
    assert d.get("a") is None


def test_on_expire_callback_called_once_for_real_expiry():
    expired = []

    def cb(k, v):
        expired.append((k, v))

    d, ft = make_dict(on_expire=cb)
    d.set("a", 1, ttl=1.0)
    d.set("a", 2, ttl=10.0)  # overwrites; the old heap row should NOT call cb
    ft.advance(1.0)
    assert d["a"] == 2
    assert expired == []  # old expiry ignored as stale

    ft.advance(9.0)  # total 10.0
    assert d.get("a") is None
    # callback should fire when prune runs (get triggers prune)
    assert expired == [("a", 2)]


def test_refresh_on_get_extends_ttl():
    d, ft = make_dict(default_ttl=5.0, refresh_on_get=True)
    d.set("a", 1, ttl=2.0)

    # Touch before original expiry -> refresh to now + default_ttl
    ft.advance(1.9)          # now 1.9 (original expiry was 2.0)
    assert d["a"] == 1       # refreshes expiry to 1.9 + 5.0 = 6.9

    # Don't touch it anymore; let it expire past refreshed TTL
    ft.advance(5.2)          # now 7.1 > 6.9
    with pytest.raises(KeyError):
        _ = d["a"]


def test_refresh_on_get_requires_default_ttl():
    d, _ = make_dict(default_ttl=None, refresh_on_get=True)
    d.set("a", 1, ttl=10.0)
    with pytest.raises(ValueError):
        _ = d["a"]


def test_delitem_deletes():
    d, _ = make_dict()
    d.set("a", 1, ttl=10.0)
    del d["a"]
    assert d.get("a") is None
    with pytest.raises(KeyError):
        _ = d["a"]
