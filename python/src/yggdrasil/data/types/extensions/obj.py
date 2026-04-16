from dataclasses import dataclass

from ..base import DataType


__all__ = ['ObjectType']


@dataclass(frozen=True)
class ObjectType(DataType):
    pass
