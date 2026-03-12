from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from .serialized import Serialized

__all__ = ["Serializable"]


class Serializable(ABC):
    @abstractmethod
    def to_serialized(self) -> Serialized:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_serialized(cls, value: Serialized) -> Self:
        raise NotImplementedError