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
    """Base class for nested data types (struct, map, array).

    Nested types expose their inner fields via ``children_fields`` and
    therefore share a common ``equals`` implementation that compares
    their children pairwise.
    """

    def equals(
        self,
        other: "DataType",
        check_names: bool = True,
        check_dtypes: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        # A non-nested dtype cannot structurally equal a nested one.
        if not hasattr(other, "children_fields"):
            return False

        if check_dtypes and self.type_id != other.type_id:
            return False

        self_children = list(self.children_fields)
        other_children = list(other.children_fields)

        if len(self_children) != len(other_children):
            return False

        # When names are not being checked (or a nested type like MapType
        # pins its children to canonical names regardless of the user),
        # compare positionally.  Otherwise resolve matches by name so that
        # reordered structs still count as equal.
        if not check_names:
            return all(
                self_child.equals(
                    other_child,
                    check_names=check_names,
                    check_dtypes=check_dtypes,
                    check_metadata=check_metadata,
                )
                for self_child, other_child in zip(self_children, other_children)
            )

        seen: set[str] = set()
        for i, self_child in enumerate(self_children):
            other_child = other.field_by(
                name=self_child.name, index=i, raise_error=False
            )

            if other_child is None:
                return False

            if not self_child.equals(
                other_child,
                check_names=check_names,
                check_dtypes=check_dtypes,
                check_metadata=check_metadata,
            ):
                return False

            seen.add(self_child.name)

        for other_child in other_children:
            if other_child.name not in seen:
                return False

        return True
