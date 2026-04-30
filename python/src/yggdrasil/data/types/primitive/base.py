from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from yggdrasil.io.enums import Mode

from ..base import DataType

if TYPE_CHECKING:
    from ...data_field import Field


__all__ = ["PrimitiveType"]


@dataclass(frozen=True, repr=False)
class PrimitiveType(DataType, ABC):
    """Shared scalar-shaped base for every non-nested leaf type.

    The only state is ``byte_size`` — a physical width hint that flows into
    Arrow / DDL encoding. Subclasses layer on their own frozen-dataclass
    fields (``signed``, ``precision``/``scale``, ``unit``/``tz``, ...) and
    override the cast / autotag / to_* hooks as needed.
    """

    byte_size: int | None = None

    @property
    def children_fields(self) -> list["Field"]:
        return []

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        if self.byte_size is not None:
            base["byte_size"] = self.byte_size
        return base

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        if self.byte_size is not None:
            tags[b"byte_size"] = str(self.byte_size).encode("utf-8")
        return tags

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: "Mode" = Mode.AUTO,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "PrimitiveType":
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot merge {self.__class__.__name__} with {other.__class__.__name__}"
            )

        if mode is Mode.IGNORE:
            return self
        elif mode is Mode.OVERWRITE:
            return other
        elif mode is Mode.AUTO:
            byte_size = self.byte_size or other.byte_size
            if byte_size != self.byte_size:
                return self.__class__(byte_size=byte_size)
            return self

        if downcast == upcast:
            return self

        left = self.byte_size
        right = other.byte_size

        if left is None:
            return other if right is not None else self
        if right is None:
            return self

        if downcast:
            return self.__class__(byte_size=min(left, right))
        return self.__class__(byte_size=max(left, right))
