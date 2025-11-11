"""Scalar-level tests for the Arrow casting utilities."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

# Ensure the package rooted under ``python/src`` is importable when tests run in-place.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Skip the suite if the optional dependency is not present in the environment.
pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.arrow import ArrowCastRegistry, ArrowScalarCastRegistry


def test_scalar_registry_uses_array_fallback() -> None:
    array_registry = ArrowCastRegistry()
    scalar_registry = ArrowScalarCastRegistry(array_registry=array_registry)

    source_field = pa.field("value", pa.int32())
    target_field = pa.field("value", pa.int64())

    caster = scalar_registry.get_or_build(source_field, target_field)

    scalar = pa.scalar(7, type=source_field.type)

    cast_scalar = caster.cast(
        scalar,
        options=pc.CastOptions(target_type=pa.float64(), allow_float_truncate=True),
    )

    assert isinstance(cast_scalar, pa.Scalar)
    assert cast_scalar.type.equals(pa.float64())
    assert cast_scalar.as_py() == pytest.approx(7.0)


def test_scalar_string_to_timestamp_preserves_fractional_seconds() -> None:
    array_registry = ArrowCastRegistry()
    scalar_registry = ArrowScalarCastRegistry(array_registry=array_registry)

    caster = scalar_registry.get_or_build(
        pa.field("value", pa.string()), pa.field("value", pa.timestamp("us", tz="UTC"))
    )

    scalar = pa.scalar("2024-01-01T00:00:00.111222+00:00", type=pa.string())
    cast_scalar = caster.cast(scalar)

    assert cast_scalar.type.equals(pa.timestamp("us", tz="UTC"))
    assert cast_scalar.as_py().isoformat() == "2024-01-01T00:00:00.111222+00:00"


def test_scalar_string_to_time_preserves_fractional_seconds() -> None:
    array_registry = ArrowCastRegistry()
    scalar_registry = ArrowScalarCastRegistry(array_registry=array_registry)

    caster = scalar_registry.get_or_build(
        pa.field("value", pa.string()), pa.field("value", pa.time64("us"))
    )

    scalar = pa.scalar("05:15:30.765432", type=pa.string())
    cast_scalar = caster.cast(scalar)

    assert cast_scalar.type.equals(pa.time64("us"))
    assert cast_scalar.as_py().isoformat() == "05:15:30.765432"
