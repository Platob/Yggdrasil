"""Tests for the YggDataclass mixin: copy / with_<field> / pickle defaults."""
from __future__ import annotations

import copy as copy_module
import pickle
from dataclasses import dataclass, field

import pytest

from yggdrasil.dataclasses import YggDataclass


# ── fixtures ───────────────────────────────────────────────────


@dataclass
class Point(YggDataclass):
    x: int
    y: int = 0
    label: str = "p"


@dataclass(frozen=True)
class FrozenPoint(YggDataclass):
    x: int
    y: int = 0


@dataclass(slots=True)
class SlotsPoint(YggDataclass):
    x: int
    y: int = 0


@dataclass
class WithCache(YggDataclass):
    name: str
    tags: list = field(default_factory=list)
    note: str = field(init=False, default="auto")
    _cache: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._cache["init"] = True


# ── copy ───────────────────────────────────────────────────────


def test_copy_returns_new_instance_same_values():
    p = Point(1, 2, "a")
    q = p.copy()
    assert q == p
    assert q is not p


def test_copy_kwargs_override_field():
    p = Point(1, 2, "a")
    q = p.copy(y=99)
    assert q == Point(1, 99, "a")
    assert p.y == 2  # original untouched


def test_copy_positional_args_replace_in_order():
    p = Point(1, 2, "a")
    q = p.copy(10, 20)
    assert q == Point(10, 20, "a")


def test_copy_positional_and_keyword_conflict_raises():
    p = Point(1, 2, "a")
    with pytest.raises(TypeError, match="both positionally and by keyword"):
        p.copy(10, x=99)


def test_copy_too_many_positional_args_raises():
    p = Point(1, 2, "a")
    with pytest.raises(TypeError, match="up to 3 positional"):
        p.copy(1, 2, 3, 4)


def test_copy_unknown_field_raises():
    p = Point(1, 2, "a")
    with pytest.raises(TypeError, match="unknown init field 'z'"):
        p.copy(z=5)


def test_copy_works_on_frozen_dataclass():
    fp = FrozenPoint(1, 2)
    fq = fp.copy(y=9)
    assert fq == FrozenPoint(1, 9)
    assert fp.y == 2


def test_copy_works_on_slots_dataclass():
    sp = SlotsPoint(1, 2)
    sq = sp.copy(y=9)
    assert sq == SlotsPoint(1, 9)


# ── with_<field> ───────────────────────────────────────────────


def test_with_field_returns_new_instance():
    p = Point(1, 2, "a")
    q = p.with_y(99)
    assert q == Point(1, 99, "a")
    assert p.y == 2


def test_with_field_inplace_mutates_in_place():
    p = Point(1, 2, "a")
    result = p.with_y(99, inplace=True)
    assert result is p
    assert p.y == 99


def test_with_field_inplace_works_on_frozen():
    # object.__setattr__ goes around the frozen check by design — useful for
    # rehydration paths but a backdoor users opt into knowingly via inplace=True.
    fp = FrozenPoint(1, 2)
    fp.with_y(99, inplace=True)
    assert fp.y == 99


def test_with_field_unknown_field_raises_attribute_error():
    p = Point(1, 2, "a")
    with pytest.raises(AttributeError, match="with_-able field 'z'"):
        _ = p.with_z


def test_with_field_skips_private_non_init_field():
    obj = WithCache(name="x")
    with pytest.raises(AttributeError, match="with_-able"):
        _ = obj.with__cache


def test_with_field_works_on_non_init_public_field():
    obj = WithCache(name="x")
    new = obj.with_note("manual")
    assert new.note == "manual"
    assert obj.note == "auto"


def test_with_field_signature_keyword_only_inplace():
    p = Point(1, 2, "a")
    setter = p.with_y
    # inplace must be keyword-only — passing it positionally targets `value`.
    with pytest.raises(TypeError):
        setter(99, True)  # type: ignore[call-arg]


def test_attribute_error_message_lists_fields():
    p = Point(1, 2, "a")
    with pytest.raises(AttributeError) as exc:
        _ = p.with_does_not_exist
    msg = str(exc.value)
    assert "x" in msg and "y" in msg and "label" in msg


# ── pickle / copy.deepcopy ─────────────────────────────────────


def test_getstate_includes_init_and_public_non_init():
    obj = WithCache(name="x", tags=["a"])
    state = obj.__getstate__()
    assert state["name"] == "x"
    assert state["tags"] == ["a"]
    assert state["note"] == "auto"
    assert "_cache" not in state  # private non-init dropped


def test_setstate_restores_init_and_public_non_init():
    obj = WithCache.__new__(WithCache)
    obj.__setstate__({"name": "y", "tags": [1, 2], "note": "manual"})
    assert obj.name == "y"
    assert obj.tags == [1, 2]
    assert obj.note == "manual"


def test_setstate_missing_required_init_field_raises():
    obj = WithCache.__new__(WithCache)
    with pytest.raises(TypeError, match="missing required field 'name'"):
        obj.__setstate__({})


def test_setstate_fills_defaults_for_missing_optional():
    obj = WithCache.__new__(WithCache)
    obj.__setstate__({"name": "y"})
    assert obj.name == "y"
    assert obj.tags == []
    assert obj.note == "auto"


def test_setstate_rejects_non_mapping():
    obj = WithCache.__new__(WithCache)
    with pytest.raises(TypeError, match="expected a dict-like state"):
        obj.__setstate__(42)


def test_pickle_roundtrip_regular():
    p = Point(1, 2, "a")
    blob = pickle.dumps(p)
    q = pickle.loads(blob)
    assert q == p


def test_pickle_roundtrip_frozen():
    fp = FrozenPoint(1, 2)
    blob = pickle.dumps(fp)
    fq = pickle.loads(blob)
    assert fq == fp


def test_pickle_roundtrip_slots():
    sp = SlotsPoint(1, 2)
    blob = pickle.dumps(sp)
    sq = pickle.loads(blob)
    assert sq == sp


def test_deepcopy_roundtrip():
    obj = WithCache(name="x", tags=["a", "b"])
    obj.note = "manual"
    clone = copy_module.deepcopy(obj)
    assert clone.name == obj.name
    assert clone.tags == obj.tags
    assert clone.tags is not obj.tags
    assert clone.note == "manual"


# ── non-dataclass subclasses ───────────────────────────────────


class PlainConfig(YggDataclass):
    """Regular class with __init__, NOT decorated with @dataclass."""

    def __init__(self, host: str, port: int = 443, scheme: str = "https") -> None:
        self.host = host
        self.port = port
        self.scheme = scheme
        # post-init derived state — not a constructor param
        self.endpoint = f"{scheme}://{host}:{port}"


class PlainSlots(YggDataclass):
    __slots__ = ("a", "b", "_secret")

    def __init__(self, a: int, b: int = 0) -> None:
        self.a = a
        self.b = b
        self._secret = "hidden"


def test_plain_copy_carries_init_and_public_state():
    cfg = PlainConfig("api.example.com", 8443)
    cfg.endpoint = "custom://override"
    clone = cfg.copy()
    assert isinstance(clone, PlainConfig)
    assert clone.host == "api.example.com"
    assert clone.port == 8443
    assert clone.scheme == "https"
    # Non-init public attribute is carried over even though __init__ would
    # normally recompute it.
    assert clone.endpoint == "custom://override"


def test_plain_copy_keyword_override():
    cfg = PlainConfig("api.example.com", 8443)
    clone = cfg.copy(port=9000)
    assert clone.port == 9000
    assert clone.host == "api.example.com"
    assert cfg.port == 8443


def test_plain_copy_positional_args_use_init_order():
    cfg = PlainConfig("api.example.com", 8443)
    clone = cfg.copy("other.example.com", 80)
    assert clone.host == "other.example.com"
    assert clone.port == 80
    assert clone.scheme == "https"


def test_plain_copy_too_many_positional_raises():
    cfg = PlainConfig("api.example.com")
    with pytest.raises(TypeError, match="up to 3 positional"):
        cfg.copy(1, 2, 3, 4)


def test_plain_copy_unknown_kwarg_applied_after_construction():
    # Non-dataclass policy: unknown init params become post-construction
    # attribute writes — lets users update post-init state by name.
    cfg = PlainConfig("api.example.com")
    clone = cfg.copy(endpoint="manual://thing")
    assert clone.endpoint == "manual://thing"
    assert clone.host == "api.example.com"


def test_plain_with_init_field_returns_copy():
    cfg = PlainConfig("api.example.com", 8443)
    clone = cfg.with_port(9000)
    assert clone.port == 9000
    assert cfg.port == 8443


def test_plain_with_init_field_inplace():
    cfg = PlainConfig("api.example.com", 8443)
    result = cfg.with_port(9000, inplace=True)
    assert result is cfg
    assert cfg.port == 9000


def test_plain_with_non_init_public_field():
    cfg = PlainConfig("api.example.com")
    clone = cfg.with_endpoint("manual://x")
    assert clone.endpoint == "manual://x"
    assert cfg.endpoint != "manual://x"


def test_plain_with_unknown_field_raises():
    cfg = PlainConfig("api.example.com")
    with pytest.raises(AttributeError, match="with_-able field 'nope'"):
        _ = cfg.with_nope


def test_plain_with_private_field_blocked():
    obj = PlainSlots(1, 2)
    with pytest.raises(AttributeError, match="private"):
        _ = obj.with__secret


def test_plain_getstate_includes_public_only():
    cfg = PlainConfig("api.example.com", 8443)
    state = cfg.__getstate__()
    assert state["host"] == "api.example.com"
    assert state["port"] == 8443
    assert state["scheme"] == "https"
    assert state["endpoint"] == "https://api.example.com:8443"


def test_plain_getstate_skips_private_attrs():
    obj = PlainSlots(1, 2)
    state = obj.__getstate__()
    assert state == {"a": 1, "b": 2}
    assert "_secret" not in state


def test_plain_setstate_restores_public_only():
    obj = PlainConfig.__new__(PlainConfig)
    obj.__setstate__({
        "host": "x",
        "port": 80,
        "scheme": "http",
        "endpoint": "http://x:80",
    })
    assert obj.host == "x"
    assert obj.port == 80
    assert obj.scheme == "http"
    assert obj.endpoint == "http://x:80"


def test_plain_setstate_drops_private_keys():
    obj = PlainSlots.__new__(PlainSlots)
    obj.__setstate__({"a": 1, "b": 2, "_secret": "tampered"})
    assert obj.a == 1
    assert obj.b == 2
    # Tampered private key should NOT have been restored.
    with pytest.raises(AttributeError):
        _ = obj._secret


def test_plain_pickle_roundtrip():
    cfg = PlainConfig("api.example.com", 8443)
    cfg.endpoint = "custom://override"
    blob = pickle.dumps(cfg)
    clone = pickle.loads(blob)
    assert clone.host == cfg.host
    assert clone.port == cfg.port
    assert clone.scheme == cfg.scheme
    assert clone.endpoint == "custom://override"


def test_plain_slots_pickle_roundtrip():
    obj = PlainSlots(1, 2)
    blob = pickle.dumps(obj)
    clone = pickle.loads(blob)
    assert clone.a == 1
    assert clone.b == 2


def test_plain_deepcopy_roundtrip():
    cfg = PlainConfig("api.example.com")
    cfg.endpoint = "custom://override"
    clone = copy_module.deepcopy(cfg)
    assert clone.host == cfg.host
    assert clone.endpoint == "custom://override"
    assert clone is not cfg
