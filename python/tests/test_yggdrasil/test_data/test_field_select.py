"""Field.select_in_* + CastOptions.match_by behaviour."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import Field
from yggdrasil.data.options import CastOptions


def _id_field() -> Field:
    return Field.from_(pa.field("id", pa.int64()))


# ---------------------------------------------------------------------------
# Field.position
# ---------------------------------------------------------------------------


class TestPosition:
    def test_default_is_none(self) -> None:
        assert _id_field().position is None

    def test_set_position(self) -> None:
        f = _id_field()
        f.set_position(2)
        assert f.position == 2

    def test_set_position_clear(self) -> None:
        f = _id_field()
        f.set_position(2)
        f.set_position(None)
        assert f.position is None

    def test_with_position_returns_copy(self) -> None:
        f = _id_field()
        g = f.with_position(1)
        assert f.position is None
        assert g.position == 1

    def test_negative_position_rejected(self) -> None:
        with pytest.raises(ValueError):
            _id_field().set_position(-1)


class TestCastOptionsMatchBy:
    def test_string_keys_coerced_to_fields(self) -> None:
        opts = CastOptions(match_by=["id", "tenant"])
        assert opts.match_by_keys == ["id", "tenant"]
        assert all(isinstance(f, Field) for f in opts.match_by)

    def test_field_keys_passthrough(self) -> None:
        f = _id_field()
        opts = CastOptions(match_by=[f])
        assert opts.match_by_keys == ["id"]
        assert opts.match_by[0] is f

    def test_mixed_keys_normalized(self) -> None:
        f = _id_field()
        opts = CastOptions(match_by=[f, "tenant"])
        assert opts.match_by_keys == ["id", "tenant"]

    def test_match_by_keys_none_when_unset(self) -> None:
        assert CastOptions().match_by_keys is None

    def test_empty_match_by_collapses_to_none(self) -> None:
        assert CastOptions(match_by=[]).match_by is None
        assert CastOptions(match_by=[]).match_by_keys is None
