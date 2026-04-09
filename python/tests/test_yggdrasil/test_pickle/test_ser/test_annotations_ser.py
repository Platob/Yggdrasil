from __future__ import annotations

from yggdrasil.pickle.ser.annotations import (
    dump_function_annotations,
    load_function_annotations,
)


class _BadReduceAnnotation:
    def __reduce_ex__(self, protocol: int):
        raise RecursionError("annotation reduce recursion")

    def __repr__(self) -> str:
        return "BadReduceAnnotation()"


class _BadReduceAndReprAnnotation:
    def __reduce_ex__(self, protocol: int):
        raise RecursionError("annotation reduce recursion")

    def __repr__(self) -> str:
        raise RecursionError("annotation repr recursion")


def test_dump_and_load_function_annotations_roundtrip() -> None:
    payload = dump_function_annotations({"x": int, "return": str})
    out = load_function_annotations(payload)

    assert out == {"x": int, "return": str}


def test_dump_annotation_failure_falls_back_to_repr() -> None:
    payload = dump_function_annotations({"x": _BadReduceAnnotation(), "return": int})
    out = load_function_annotations(payload)

    assert out["return"] is int
    assert isinstance(out["x"], str)
    assert "BadReduceAnnotation" in out["x"]


def test_dump_annotation_failure_and_repr_failure_is_safe() -> None:
    payload = dump_function_annotations(
        {"x": _BadReduceAndReprAnnotation(), "return": int}
    )
    out = load_function_annotations(payload)

    assert out["return"] is int
    assert isinstance(out["x"], str)
    assert "unrepresentable" in out["x"]


def test_load_function_annotations_accepts_legacy_raw_dict() -> None:
    out = load_function_annotations({"x": int, "return": int})
    assert out == {"x": int, "return": int}


def test_load_function_annotations_malformed_payload_returns_empty_dict() -> None:
    assert load_function_annotations((999, {"x": ("v", b"bad")})) == {}

