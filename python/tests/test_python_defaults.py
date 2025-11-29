import dataclasses
import datetime
import decimal
import uuid
from typing import Optional

import pytest

from yggdrasil.types import default_from_hint


@dataclasses.dataclass
class Inner:
    name: str


@dataclasses.dataclass
class Example:
    title: str
    tags: list[str]
    info: Inner
    created: int = 1
    skip: bool = dataclasses.field(default=False, init=False)


def test_default_for_primitives_and_optional():
    assert default_from_hint(str) == ""
    assert default_from_hint(Optional[str]) is None


def test_default_for_common_special_types():
    assert default_from_hint(datetime.datetime) == datetime.datetime(
        1970, 1, 1, tzinfo=datetime.timezone.utc
    )
    assert default_from_hint(datetime.date) == datetime.date(1970, 1, 1)
    assert default_from_hint(datetime.time) == datetime.time(
        0, 0, 0, tzinfo=datetime.timezone.utc
    )
    assert default_from_hint(datetime.timedelta) == datetime.timedelta(0)
    assert default_from_hint(uuid.UUID) == uuid.UUID(int=0)
    assert default_from_hint(decimal.Decimal) == decimal.Decimal(0)


def test_default_for_collections():
    assert default_from_hint(list[int]) == []
    assert default_from_hint(dict[str, int]) == {}
    assert default_from_hint(set[str]) == set()


def test_default_for_tuple():
    assert default_from_hint(tuple[str, int]) == ("", 0)


def test_default_for_dataclass():
    value = default_from_hint(Example)

    assert value == Example(title="", tags=[], info=Inner(name=""))


def test_unsupported_type_raises_type_error():
    class NeedsArgs:
        def __init__(self, value):
            self.value = value

    with pytest.raises(TypeError):
        default_from_hint(NeedsArgs)
