"""Tests for the Arrow casting utilities."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

# Ensure the package rooted under ``python/src`` is importable when tests run in-place.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Skip the suite if the optional dependency is not present in the environment.
pytest.importorskip("pyarrow")

import pyarrow as pa

from yggdrasil.cli import main as cli_main
from yggdrasil.data.arrow import ArrowCastRegistry


def test_registry_builds_nested_list_caster() -> None:
    registry = ArrowCastRegistry()
    source_field = pa.field("values", pa.list_(pa.int32()))
    target_field = pa.field("values", pa.list_(pa.int64()))

    caster = registry.get_or_build(source_field, target_field)
    again = registry.get_or_build(source_field, target_field)

    assert caster is again

    array = pa.array([[1, 2, 3]], type=source_field.type)
    cast_array = caster.cast(array)

    assert cast_array.type.equals(target_field.type)
    assert cast_array.to_pylist() == [[1, 2, 3]]


def test_registry_handles_struct_fields() -> None:
    registry = ArrowCastRegistry()
    source_field = pa.field(
        "record",
        pa.struct([pa.field("count", pa.int32()), pa.field("label", pa.string())]),
    )
    target_field = pa.field(
        "record",
        pa.struct([pa.field("count", pa.int64()), pa.field("label", pa.string())]),
    )

    caster = registry.get_or_build(source_field, target_field)

    struct_array = pa.array(
        [{"count": 1, "label": "alpha"}, {"count": 2, "label": "beta"}],
        type=source_field.type,
    )
    cast_struct = caster.cast(struct_array)
    assert cast_struct.type.equals(target_field.type)

    array = pa.array(
        [[{"count": 1, "label": "alpha"}, {"count": 2, "label": "beta"}]],
        type=pa.list_(source_field.type),
    )

    list_caster = registry.get_or_build(
        pa.field("records", pa.list_(source_field.type)),
        pa.field("records", pa.list_(target_field.type)),
    )

    cast_array = list_caster.cast(array)
    assert cast_array.type.equals(pa.list_(target_field.type))
    assert cast_array.to_pylist()[0][0]["count"] == 1
    assert cast_array.to_pylist()[0][1]["count"] == 2


def test_cli_greet_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(["greet", "Loki"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Hail, Loki!" in captured.out


def test_cli_arrow_cast_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(["arrow-cast", "5", "6"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["target_type"] == "list<item: int64>"
    assert payload["values"] == [[5, 6]]


def test_caster_rejects_mismatched_array_type() -> None:
    registry = ArrowCastRegistry()
    source_field = pa.field("value", pa.int32())
    target_field = pa.field("value", pa.int64())

    caster = registry.get_or_build(source_field, target_field)

    with pytest.raises(ValueError):
        caster.cast(pa.array([1, 2], type=pa.int64()))
