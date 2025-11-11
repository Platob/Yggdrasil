"""Array-level tests for the Arrow casting utilities."""

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


@pytest.fixture(scope="module")
def registry() -> ArrowCastRegistry:
    """Provide a shared registry instance to avoid redundant setup."""

    return ArrowCastRegistry()


def test_registry_builds_nested_list_caster(registry: ArrowCastRegistry) -> None:
    source_field = pa.field("values", pa.list_(pa.int32()))
    target_field = pa.field("values", pa.list_(pa.int64()))

    caster = registry.get_or_build(source_field, target_field)
    again = registry.get_or_build(source_field, target_field)

    assert caster is again

    array = pa.array([[1, 2, 3]], type=source_field.type)
    cast_array = caster.cast(array)

    assert cast_array.type.equals(target_field.type)
    assert cast_array.to_pylist() == [[1, 2, 3]]


def test_registry_handles_struct_fields(registry: ArrowCastRegistry) -> None:
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


def test_registry_accepts_data_types(registry: ArrowCastRegistry) -> None:
    source_type = pa.list_(pa.int32())
    target_type = pa.list_(pa.int64())

    caster = registry.get_or_build(source_type, target_type)

    array = pa.array([[1, 2, 3]], type=source_type)
    cast_array = caster.cast(array)

    assert cast_array.type.equals(pa.list_(pa.int64()))
    assert cast_array.to_pylist() == [[1, 2, 3]]


def test_string_to_timestamp_with_timezone(registry: ArrowCastRegistry) -> None:
    caster = registry.get_or_build(pa.string(), pa.timestamp("s", tz="UTC"))

    array = pa.array([
        "2024-01-01T00:00:00+00:00",
        "2024-01-01T03:00:00+03:00",
    ])

    cast_array = caster.cast(array, )

    assert cast_array.type.equals(pa.timestamp("s", tz="UTC"))
    assert [value.isoformat() for value in cast_array.to_pylist()] == [
        "2024-01-01T00:00:00+00:00",
        "2024-01-01T00:00:00+00:00",
    ]


def test_string_to_timestamp_from_naive_sources(registry: ArrowCastRegistry) -> None:
    caster = registry.get_or_build(pa.string(), pa.timestamp("s", tz="UTC"))

    array = pa.array([
        "2024-01-01T00:00:00",
        "2024-01-01T03:30:00",
    ])

    cast_array = caster.cast(array)

    assert cast_array.type.equals(pa.timestamp("s", tz="UTC"))
    assert [value.isoformat() for value in cast_array.to_pylist()] == [
        "2024-01-01T00:00:00+00:00",
        "2024-01-01T03:30:00+00:00",
    ]


def test_string_to_timestamp_preserves_fractional_seconds(
    registry: ArrowCastRegistry,
) -> None:
    caster = registry.get_or_build(pa.string(), pa.timestamp("us", tz="UTC"))

    array = pa.array(
        ["2024-01-01T00:00:00.123456+00:00", "2024-01-01T01:02:03.987654+00:00"]
    )

    cast_array = caster.cast(array)

    assert cast_array.type.equals(pa.timestamp("us", tz="UTC"))
    assert [value.isoformat() for value in cast_array.to_pylist()] == [
        "2024-01-01T00:00:00.123456+00:00",
        "2024-01-01T01:02:03.987654+00:00",
    ]


def test_string_to_date_and_time(registry: ArrowCastRegistry) -> None:
    date_caster = registry.get_or_build(pa.string(), pa.date32())
    time_caster = registry.get_or_build(pa.string(), pa.time32("s"))

    date_array = pa.array(["2024-01-01", "2024-01-02"])
    time_array = pa.array(["12:30:00", "05:15:30"])

    cast_dates = date_caster.cast(date_array)
    cast_times = time_caster.cast(time_array)

    assert cast_dates.type.equals(pa.date32())
    assert [value.isoformat() for value in cast_dates.to_pylist()] == [
        "2024-01-01",
        "2024-01-02",
    ]

    assert cast_times.type.equals(pa.time32("s"))
    assert [value.isoformat() for value in cast_times.to_pylist()] == [
        "12:30:00",
        "05:15:30",
    ]


def test_string_to_time_preserves_fractional_seconds(registry: ArrowCastRegistry) -> None:
    caster = registry.get_or_build(pa.string(), pa.time64("us"))

    array = pa.array(["12:30:00.123456", "05:15:30.654321"])

    cast_times = caster.cast(array)

    assert cast_times.type.equals(pa.time64("us"))
    assert [value.isoformat() for value in cast_times.to_pylist()] == [
        "12:30:00.123456",
        "05:15:30.654321",
    ]


def test_cli_arrow_cast_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli_main(["arrow-cast", "5", "6"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["target_type"] == "list<item: int64>"
    assert payload["values"] == [[5, 6]]


def test_caster_rejects_mismatched_array_type(registry: ArrowCastRegistry) -> None:
    source_field = pa.field("value", pa.int32())
    target_field = pa.field("value", pa.int64())

    caster = registry.get_or_build(source_field, target_field)

    with pytest.raises(ValueError):
        caster.cast(pa.array([1, 2], type=pa.int64()))


def test_array_caster_accepts_extra_cast_arguments(registry: ArrowCastRegistry) -> None:
    source_field = pa.field("value", pa.int32())
    target_field = pa.field("value", pa.int64())

    caster = registry.get_or_build(source_field, target_field)

    cast_array = caster.cast(
        pa.array([1, 2, 3], type=source_field.type),
        target_type=pa.float64(),
        safe=False,
    )

    assert cast_array.type.equals(pa.float64())
