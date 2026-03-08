from __future__ import annotations

from dataclasses import MISSING, dataclass, field

import pytest

# Replace with your real module path
import yggdrasil.dataclasses.dataclass as mod


@dataclass
class ExampleState:
    a: int
    b: str = "default-b"
    c: list[int] = field(default_factory=list)
    d: int | None = None
    _private_init_field: int = field(default=99, init=False)
    lazy_cache: dict[str, int] = field(default_factory=dict, init=False)


@dataclass
class RequiredOnly:
    a: int
    b: str


@dataclass
class WithNonInitDefaults:
    a: int
    cache: dict[str, int] = field(default_factory=dict, init=False)
    marker: str = field(default="x", init=False)


def test_get_from_dict_returns_direct_key_first() -> None:
    obj = {"a": 1, "prea": 2}

    assert mod.get_from_dict(obj, keys=["a"], prefix="pre") == 1


def test_get_from_dict_falls_back_to_prefixed_key() -> None:
    obj = {"prea": 2}

    assert mod.get_from_dict(obj, keys=["a"], prefix="pre") == 2


def test_get_from_dict_respects_key_order() -> None:
    obj = {"prec": 30, "b": 20}

    assert mod.get_from_dict(obj, keys=["a", "b", "c"], prefix="pre") == 20


def test_get_from_dict_returns_missing_when_absent() -> None:
    obj = {}

    assert mod.get_from_dict(obj, keys=["a", "b"], prefix="pre") is MISSING


def test_default_value_returns_static_default() -> None:
    f = ExampleState.__dataclass_fields__["b"]

    assert mod.default_value(f) == "default-b"


def test_default_value_returns_factory_value() -> None:
    f = ExampleState.__dataclass_fields__["c"]

    value = mod.default_value(f)

    assert value == []
    assert isinstance(value, list)


def test_default_value_returns_missing_for_required_field() -> None:
    f = ExampleState.__dataclass_fields__["a"]

    assert mod.default_value(f) is MISSING


def test_serialize_dataclass_state_keeps_only_required_non_default_values() -> None:
    obj = ExampleState(a=123)

    assert mod.serialize_dataclass_state(obj) == {"a": 123}


def test_serialize_dataclass_state_drops_static_defaults() -> None:
    obj = ExampleState(a=5, b="default-b")

    assert mod.serialize_dataclass_state(obj) == {"a": 5}


def test_serialize_dataclass_state_drops_default_factory_values() -> None:
    obj = ExampleState(a=5, c=[])

    assert mod.serialize_dataclass_state(obj) == {"a": 5}


def test_serialize_dataclass_state_skips_none_private_and_non_init_fields() -> None:
    obj = ExampleState(a=1, d=None)
    obj.lazy_cache["x"] = 1

    result = mod.serialize_dataclass_state(obj)

    assert result == {"a": 1}
    assert "lazy_cache" not in result
    assert "_private_init_field" not in result
    assert "d" not in result


def test_serialize_dataclass_state_keeps_non_default_values() -> None:
    obj = ExampleState(a=1, b="changed", c=[7], d=5)

    assert mod.serialize_dataclass_state(obj) == {
        "a": 1,
        "b": "changed",
        "c": [7],
        "d": 5,
    }


def test_restore_dataclass_state_from_raw_payload() -> None:
    obj = object.__new__(ExampleState)

    mod.restore_dataclass_state(
        obj,
        {
            "a": 10,
            "b": "hello",
            "c": [1, 2],
            "d": 99,
        },
    )

    assert obj.a == 10
    assert obj.b == "hello"
    assert obj.c == [1, 2]
    assert obj.d == 99
    assert obj._private_init_field == 99
    assert obj.lazy_cache == {}


def test_restore_dataclass_state_uses_defaults_for_missing_optional_fields() -> None:
    obj = object.__new__(ExampleState)

    mod.restore_dataclass_state(obj, {"a": 7})

    assert obj.a == 7
    assert obj.b == "default-b"
    assert obj.c == []
    assert obj.d is None
    assert obj._private_init_field == 99
    assert obj.lazy_cache == {}


def test_restore_dataclass_state_treats_none_as_empty_payload() -> None:
    obj = object.__new__(ExampleState)

    with pytest.raises(TypeError, match=r"missing required field 'a'"):
        mod.restore_dataclass_state(obj, None)


def test_restore_dataclass_state_ignores_unknown_keys() -> None:
    obj = object.__new__(ExampleState)

    mod.restore_dataclass_state(obj, {"a": 1, "unknown": "ignored"})

    assert obj.a == 1
    assert obj.b == "default-b"
    assert obj.c == []
    assert obj.d is None


def test_restore_dataclass_state_raises_for_invalid_state_type() -> None:
    obj = object.__new__(ExampleState)

    with pytest.raises(TypeError, match="Invalid pickle state"):
        mod.restore_dataclass_state(obj, "nope")


def test_restore_dataclass_state_raises_when_required_field_missing() -> None:
    obj = object.__new__(RequiredOnly)

    with pytest.raises(TypeError, match=r"missing required field 'a'"):
        mod.restore_dataclass_state(obj, {})


def test_restore_dataclass_state_resets_non_init_fields_to_defaults() -> None:
    obj = object.__new__(WithNonInitDefaults)
    object.__setattr__(obj, "cache", {"stale": 1})
    object.__setattr__(obj, "marker", "old")

    mod.restore_dataclass_state(obj, {"a": 42})

    assert obj.a == 42
    assert obj.cache == {}
    assert obj.marker == "x"


def test_dataclass_to_arrow_field_raises_for_non_dataclass() -> None:
    with pytest.raises(ValueError, match="is not a dataclass"):
        mod.dataclass_to_arrow_field(123)


def test_dataclass_to_arrow_field_builds_and_caches_for_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod.DATACLASS_ARROW_FIELD_CACHE.clear()

    sentinel = object()
    calls: list[type] = []

    def fake_arrow_field_from_hint(cls: type) -> object:
        calls.append(cls)
        return sentinel

    import sys
    import types

    fake_module = types.SimpleNamespace(arrow_field_from_hint=fake_arrow_field_from_hint)
    monkeypatch.setitem(sys.modules, "yggdrasil.arrow.python_arrow", fake_module)

    result1 = mod.dataclass_to_arrow_field(ExampleState)
    result2 = mod.dataclass_to_arrow_field(ExampleState)

    assert result1 is sentinel
    assert result2 is sentinel
    assert calls == [ExampleState]


def test_dataclass_to_arrow_field_uses_instance_class_for_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod.DATACLASS_ARROW_FIELD_CACHE.clear()

    sentinel = object()
    calls: list[type] = []

    def fake_arrow_field_from_hint(cls: type) -> object:
        calls.append(cls)
        return sentinel

    import sys
    import types

    fake_module = types.SimpleNamespace(arrow_field_from_hint=fake_arrow_field_from_hint)
    monkeypatch.setitem(sys.modules, "yggdrasil.arrow.python_arrow", fake_module)

    instance = ExampleState(a=5)

    result1 = mod.dataclass_to_arrow_field(instance)
    result2 = mod.dataclass_to_arrow_field(ExampleState)

    assert result1 is sentinel
    assert result2 is sentinel
    assert calls == [ExampleState]