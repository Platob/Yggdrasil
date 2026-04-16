from abc import ABC
from enum import IntEnum
from typing import TYPE_CHECKING

from yggdrasil.io.url import URLResource

if TYPE_CHECKING:
    pass

__all__ = [
    "EngineId",
    "Engine"
]


class EngineId(IntEnum):
    SPARK = 1
    POLARS = 2
    ARROW = 3
    PANDAS = 4


class Engine(URLResource, ABC):
    pass

