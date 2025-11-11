"""Tests for the :mod:`yggdrasil.example` module."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

# Ensure the package rooted under ``python/src`` is importable when tests run in-place.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Skip the suite if the optional dependency is not present in the environment.
pytest.importorskip("pyarrow")

from yggdrasil.example import DataFormat, demo_table, greet


def test_greet_returns_expected_salutation() -> None:
    """The ``greet`` helper should format the input name into the salutation."""

    assert greet("Skadi") == "Hail, Skadi! Welcome to Yggdrasil."


def test_data_format_metadata_round_trip() -> None:
    """Metadata produced by :class:`DataFormat` should be attached to tables."""

    format_hint = DataFormat(name="ledger", mime_type="application/x-ledger+json", version="2.1")
    table = demo_table([("Freya", 1), ("Odin", 2)])

    stamped_table = format_hint.attach_to(table)

    assert stamped_table.schema.metadata is not None
    metadata = {key.decode(): value.decode() for key, value in stamped_table.schema.metadata.items()}
    assert metadata["name"] == "character-roster"
    assert metadata["mime_type"] == "application/x-yggdrasil+json"
    assert metadata["version"] == "1.0"
