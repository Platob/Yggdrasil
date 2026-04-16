from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

from yggdrasil.data.types.base import DataType

if TYPE_CHECKING:
    from yggdrasil.data import Field


__all__ = [
    "DataType",
    "NestedType",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NestedType(DataType, ABC):

    def equals(
        self,
        other: "Field",
        check_names: bool = True,
        check_dtypes: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        seen = set()
        for i, self_field in enumerate(self.children_fields):
            other_field = other.field_by(name=self_field.name, index=i, raise_error=False)

            if other_field is None:
                return False

            elif not self_field.equals(
                other_field, check_names=check_names, check_dtypes=check_dtypes,
                check_metadata=check_metadata
            ):
                return False

            seen.add(self_field.name)

        for i, other_field in enumerate(other.children_fields):
            if other_field.name not in seen:
                return False
        return True
