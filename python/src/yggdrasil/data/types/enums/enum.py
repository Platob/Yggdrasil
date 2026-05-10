"""EnumType + typed specializations.

:class:`EnumType` is the semantic enum form of :class:`DictionaryType`
— same wire format, distinct ``DataTypeId`` so a column declared as
``enum`` round-trips as ``enum`` rather than as ``dictionary``. The
optional ``name`` field carries the originating Python enum class
name (``"Color"``, ``"Status"``) so a dataclass schema dumped to JSON
keeps the symbol; ``members`` is the optional ordered ``name → value``
mapping for display / lookup.

:class:`StrEnumType` and :class:`IntEnumType` mirror the stdlib
``enum.StrEnum`` / ``enum.IntEnum`` distinction at the schema layer:

* ``StrEnumType`` pins ``value_type`` to :class:`StringType` and is
  the natural target for a ``class Color(str, Enum)``;
* ``IntEnumType`` pins ``value_type`` to :class:`IntegerType` (8-byte
  signed by default) and is the natural target for a ``class
  Status(IntEnum)``.

Both subclasses get their own ``DataTypeId`` so ``to_dict`` /
``from_dict`` round-trips preserve the typed kind. Unknown values
still map to null on cast (lenient default) or raise (``safe=True``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..id import DataTypeId
from ..primitive.base import PrimitiveType
from ..primitive.numeric.integer import IntegerType
from ..primitive.string import StringType
from .dictionary import DictionaryType

if TYPE_CHECKING:
    from ..base import DataType


__all__ = [
    "EnumType",
    "StrEnumType",
    "IntEnumType",
]


def _normalize_members(
    members: Any, categories: tuple[Any, ...]
) -> dict[str, Any] | None:
    if members is None:
        return None
    if isinstance(members, dict):
        out = dict(members)
    else:
        out = dict(members)
    # Keep only members whose value made it into the (possibly
    # de-duplicated) categories tuple — otherwise the mapping lies.
    cat_set = set()
    for c in categories:
        try:
            cat_set.add(c)
        except TypeError:
            pass
    if cat_set:
        out = {k: v for k, v in out.items() if v in cat_set}
    return out or None


@dataclass(frozen=True, repr=False)
class EnumType(DictionaryType):
    """Semantic enum — strict named value set, dictionary-encoded on
    the wire."""

    name: str | None = None
    members: dict[str, Any] | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        normalized = _normalize_members(self.members, self.categories)
        object.__setattr__(self, "members", normalized)
        if self.name is not None and not isinstance(self.name, str):
            object.__setattr__(self, "name", str(self.name))

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.ENUM

    def _head_name(self) -> str:
        return "enum" if self.name is None else f"enum:{self.name}"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pyenum(cls, hint: Any) -> "EnumType":
        """Build an :class:`EnumType` from a Python ``enum.Enum`` subclass.

        Promotes to :class:`StrEnumType` / :class:`IntEnumType` when
        every member value matches a typed specialization — gives the
        round-trip path on ``DataType.from_pytype`` a single entry
        point that lands on the most specific class available.
        """
        import enum as _enum

        if not (isinstance(hint, type) and issubclass(hint, _enum.Enum)):
            raise TypeError(
                f"from_pyenum expected an Enum subclass, got {hint!r}"
            )

        members = list(hint)
        if not members:
            return cls(
                name=hint.__name__,
                value_type=StringType(),
                categories=(),
                members={},
            )

        values = [m.value for m in members]
        member_map = {m.name: m.value for m in members}

        # Promote to the typed specialization when every member value
        # fits — keeps the Python ``StrEnum`` / ``IntEnum`` distinction
        # visible at the schema layer.
        if all(isinstance(v, bool) is False and isinstance(v, int) for v in values):
            target_cls: type[EnumType] = IntEnumType
            value_type: PrimitiveType = IntegerType(byte_size=8, signed=True)
        elif all(isinstance(v, str) for v in values):
            target_cls = StrEnumType
            value_type = StringType()
        else:
            target_cls = cls
            from ..base import DataType

            value_type = DataType.from_pytype(type(values[0]))
            if not isinstance(value_type, PrimitiveType):
                value_type = StringType()

        return target_cls(
            name=hint.__name__,
            value_type=value_type,
            categories=tuple(values),
            members=member_map,
        )

    # ------------------------------------------------------------------
    # Merge — preserve ``name`` when both sides agree
    # ------------------------------------------------------------------

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode=None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "EnumType":
        merged = super()._merge_with_same_id(
            other, mode=mode, downcast=downcast, upcast=upcast
        )
        if not isinstance(merged, EnumType):
            return merged

        merged_name = (
            self.name
            if isinstance(other, EnumType) and other.name == self.name
            else None
        )
        # Re-stitch the name / members onto the merged result; the
        # parent already settled value_type, categories, and ordered.
        merged_members = self._merge_members(
            self.members,
            other.members if isinstance(other, EnumType) else None,
            merged.categories,
        )
        return type(merged)(
            value_type=merged.value_type,
            categories=merged.categories,
            ordered=merged.ordered,
            name=merged_name,
            members=merged_members,
        )

    @staticmethod
    def _merge_members(
        left: dict[str, Any] | None,
        right: dict[str, Any] | None,
        categories: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        if not left and not right:
            return None
        merged: dict[str, Any] = {}
        for src in (left or {}, right or {}):
            for k, v in src.items():
                merged.setdefault(k, v)
        return _normalize_members(merged, categories)

    # ------------------------------------------------------------------
    # Dict — pull `name` / `members` through serialization
    # ------------------------------------------------------------------

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(
            value, cls.class_type_id(), "ENUM", "STRENUM", "INTENUM"
        )

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], default: Any = ...
    ) -> "EnumType":
        try:
            from ..base import DataType

            value_type_payload = (
                value.get("value_type") or value.get("valueType")
            )
            if value_type_payload is not None:
                value_type = DataType.from_any(value_type_payload)
            else:
                value_type = cls._default_value_type()

            categories = value.get("categories") or ()
            return cls(
                byte_size=value.get("byte_size"),
                value_type=value_type,
                categories=tuple(categories),
                ordered=bool(value.get("ordered", False)),
                name=value.get("name"),
                members=value.get("members"),
            )
        except Exception as e:
            if default is ...:
                raise ValueError(
                    f"Cannot construct {cls.__name__} from {value!r}"
                ) from e
            return default

    @classmethod
    def _default_value_type(cls) -> PrimitiveType:
        return StringType()

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        if self.name is not None:
            base["name"] = self.name
        if self.members:
            base["members"] = dict(self.members)
        return base

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        tags[b"encoding"] = b"enum"
        if self.name:
            tags[b"enum_name"] = self.name.encode("utf-8")
        return tags


# ---------------------------------------------------------------------------
# Typed specializations — pin value_type to a single primitive kind.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, repr=False)
class StrEnumType(EnumType):
    """String-valued enum — schema parallel of stdlib ``enum.StrEnum``."""

    value_type: PrimitiveType = field(default_factory=StringType)

    def __post_init__(self):
        # Force value_type to a StringType — accepting other shapes
        # would lie about the class's contract. We allow a caller-
        # supplied StringType subclass (e.g. with ``large=True``) to
        # pass through unchanged.
        vt = self.value_type
        if vt is None or not isinstance(vt, StringType):
            vt = StringType()
            object.__setattr__(self, "value_type", vt)
        super().__post_init__()

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.STR_ENUM

    def _head_name(self) -> str:
        return "str_enum" if self.name is None else f"str_enum:{self.name}"

    @classmethod
    def _default_value_type(cls) -> PrimitiveType:
        return StringType()

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, cls.class_type_id(), "STR_ENUM", "STRENUM")


@dataclass(frozen=True, repr=False)
class IntEnumType(EnumType):
    """Integer-valued enum — schema parallel of stdlib ``enum.IntEnum``.

    Defaults to signed int64 storage (``IntegerType(byte_size=8,
    signed=True)``) which matches Python's native ``int`` width. Pass
    a narrower :class:`IntegerType` (e.g. ``Int32Type()``) to pin
    storage when the enum's range is known.
    """

    value_type: PrimitiveType = field(
        default_factory=lambda: IntegerType(byte_size=8, signed=True)
    )

    def __post_init__(self):
        vt = self.value_type
        if vt is None or not isinstance(vt, IntegerType):
            vt = IntegerType(byte_size=8, signed=True)
            object.__setattr__(self, "value_type", vt)
        super().__post_init__()

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.INT_ENUM

    def _head_name(self) -> str:
        return "int_enum" if self.name is None else f"int_enum:{self.name}"

    @classmethod
    def _default_value_type(cls) -> PrimitiveType:
        return IntegerType(byte_size=8, signed=True)

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, cls.class_type_id(), "INT_ENUM", "INTENUM")
