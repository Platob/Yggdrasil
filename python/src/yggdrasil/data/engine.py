from abc import ABC
from typing import TYPE_CHECKING

from yggdrasil.io.url import URLResource

if TYPE_CHECKING:
    pass

__all__ = [
    "Engine"
]


class Engine(URLResource, ABC):
    pass

