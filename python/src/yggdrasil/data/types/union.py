""":class:`UnionType` — a ``DataType`` for Python ``Union`` / ``T | None``.

The DataType-layer representation of a multi-arm Python type hint.
``Optional[int]`` / ``int | None`` parses into ``UnionType(IntegerType(),
NullType())``; ``int | str`` parses into ``UnionType(IntegerType(),
StringType())``; ``int | str | None`` parses into ``UnionType(IntegerType(),
StringType(), NullType())``.

Field materialisation flattens the union
--------------------------------------

When a :class:`UnionType` is wrapped into a :class:`yggdrasil.data.Field`
via :meth:`to_field`, the field collapses ``NullType`` into the
``nullable`` flag and un-nests the union when only one non-null member
remains:

* ``UnionType(IntegerType(), NullType()).to_field()`` →
  ``Field(dtype=IntegerType(), nullable=True)``  (drop Null, unnest).
* ``UnionType(IntegerType(), StringType()).to_field()`` →
  ``Field(dtype=UnionType(IntegerType(), StringType()), nullable=False)``
  (no Null in the union — keep the multi-arm shape).
* ``UnionType(IntegerType(), StringType(), NullType()).to_field()`` →
  ``Field(dtype=UnionType(IntegerType(), StringType()), nullable=True)``
  (drop Null, keep the multi-arm shape).

This is the bridge between the DataType layer (union-rich) and the
Field layer (nullable-flat): the union survives long enough to round-
trip ``to_pyhint() == Union[*members]`` but doesn't leak into the
engine projections, which would have nothing useful to do with a
generic union anyway.

Engine projection
-----------------

Arrow / Polars / Spark don't have a clean generic-union scalar type
that fits yggdrasil's column model. :meth:`to_arrow` /
:meth:`to_polars` / :meth:`to_spark` delegate as follows:

* one non-null member → delegate to that member's projection
  (``UnionType(Int, Null).to_arrow()`` == ``IntegerType().to_arrow()``);
* zero non-null members → :class:`NullType` projection;
* multi-arm non-null → fall back to :class:`StringType` projection,
  matching the existing ``from_pytype`` behaviour for mixed unions.

So a Field carrying ``UnionType(Int, Null)`` produces the same Arrow
type as the original ``IntegerType()`` did before the union was
introduced — no engine-side disruption.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import pyarrow as pa

from yggdrasil.enums import Mode

from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from yggdrasil.lazy_imports import field_class

from .base import DataType
from .id import DataTypeId
from .primitive import NullType, StringType

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst
    from ..cast.options import CastOptions
    from ..data_field import Field


__all__ = ["UnionType"]


@dataclass(frozen=True, repr=False)
class UnionType(DataType):
    """DataType wrapper around a Python ``Union`` / ``Optional``.

    *members* is the (frozen) tuple of inner :class:`DataType` arms.
    ``NullType`` membership is what makes the union *nullable* — the
    canonical pattern for ``Optional[T]`` is ``UnionType(T, Null)``.
    """

    members: tuple[DataType, ...] = dc_field(default_factory=tuple)

    _SINGLETON_BY_DEFAULT_ARGS: ClassVar[bool] = False

    # ------------------------------------------------------------------
    # Identity / housekeeping
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Coerce list → tuple so equality / hash stay stable across
        # both ``UnionType([Int, Null])`` and ``UnionType((Int, Null))``
        # call shapes.
        if not isinstance(self.members, tuple):
            object.__setattr__(self, "members", tuple(self.members))
        for m in self.members:
            if not isinstance(m, DataType):
                raise TypeError(
                    f"UnionType members must be DataType instances; "
                    f"got {type(m).__name__}: {m!r}. Pass DataType "
                    "subclasses (IntegerType(), NullType(), ...) or "
                    "DataType.from_pytype(hint) results."
                )

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.UNION

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.UNION

    @property
    def children(self) -> list["Field"]:
        # Members aren't named fields — the union picks one of them
        # per row, there's no per-arm column. Returning ``[]`` keeps
        # the children walker quiet; ``to_dict`` serialises the
        # members directly.
        return []

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        inner = ", ".join(m.pretty_format(indent=0, level=0) for m in self.members)
        return f"{pad}union<{inner}>"

    def __repr__(self) -> str:
        return f"UnionType({', '.join(repr(m) for m in self.members)})"

    # ------------------------------------------------------------------
    # Nullable / member helpers
    # ------------------------------------------------------------------

    @property
    def nullable(self) -> bool:
        """``True`` iff at least one member is :class:`NullType`."""
        return any(isinstance(m, NullType) for m in self.members)

    @property
    def non_null_members(self) -> tuple[DataType, ...]:
        """The members minus any :class:`NullType` arms (order preserved)."""
        return tuple(m for m in self.members if not isinstance(m, NullType))

    def without_null(self) -> "UnionType":
        """Return a copy with all :class:`NullType` members removed.

        Preserves the multi-arm shape — even when only one non-null
        member is left the result is still a :class:`UnionType`. Use
        :meth:`to_field` (or the explicit ``non_null_members[0]``)
        when you want the un-nested type.
        """
        return type(self)(members=self.non_null_members)

    # ------------------------------------------------------------------
    # Field materialisation — drop NullType + un-nest
    # ------------------------------------------------------------------

    def to_field(
        self,
        name: str = DEFAULT_FIELD_NAME,
        nullable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> "Field":
        """Wrap into a :class:`Field`, flattening the union.

        Bridge from the union-rich DataType layer to the nullable-flat
        Field layer:

        * If any member is :class:`NullType`, drop those arms and set
          ``Field.nullable=True`` regardless of the *nullable* arg
          (Null membership is the stronger signal of intent).
        * If only one non-null member remains, un-nest the union and
          use that member as ``Field.dtype`` directly.
        * Otherwise (multi-arm non-null), keep the trimmed union as
          ``Field.dtype``.

        Empty union (no members) and union-of-only-Null both collapse
        to a :class:`Field` with ``NullType()`` dtype, nullable=True.
        """
        non_null = self.non_null_members
        had_null = len(non_null) != len(self.members)

        if not non_null:
            dtype: DataType = NullType()
            final_nullable = True
        elif len(non_null) == 1:
            dtype = non_null[0]
            final_nullable = nullable or had_null
        else:
            dtype = type(self)(members=non_null) if had_null else self
            final_nullable = nullable or had_null

        return field_class()(
            name=name or DEFAULT_FIELD_NAME,
            dtype=dtype,
            nullable=final_nullable,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Python hint round-trip
    # ------------------------------------------------------------------

    def _default_pyhint(self) -> Any:
        """``Union[m1.to_pyhint(), m2.to_pyhint(), ...]``.

        Single-member union collapses to that member's hint; the empty
        union collapses to ``None`` (matches ``NullType``'s default).
        """
        if not self.members:
            return type(None)
        if len(self.members) == 1:
            return self.members[0].to_pyhint()
        args = tuple(m.to_pyhint() for m in self.members)
        return Union[args]

    # ------------------------------------------------------------------
    # Engine probes / constructors
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        # Arrow's dense / sparse union types exist but aren't the
        # same intent as Python's ``Union[T, U]``; we don't claim
        # them here — the cast pipeline keeps them in their native
        # form on the Arrow side.
        return False

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "UnionType":
        raise TypeError(
            f"Cannot infer UnionType from Arrow type {dtype!r}. "
            "UnionType is the Python-side wrapper around ``Union[T, U]`` / "
            "``Optional[T]``; the engine projections delegate to a single "
            "non-null member or fall back to StringType."
        )

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        return False

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "UnionType":
        raise TypeError(
            f"Cannot infer UnionType from Polars dtype {dtype!r}."
        )

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return False

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "UnionType":
        raise TypeError(
            f"Cannot infer UnionType from Spark type {dtype!r}."
        )

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        type_id = value.get("id")
        if type_id == int(DataTypeId.UNION):
            return True
        name = str(value.get("name", "")).upper()
        return name == "UNION"

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "UnionType":
        raw_members = value.get("members", [])
        try:
            members = tuple(
                m if isinstance(m, DataType) else DataType.from_dict(m)
                for m in raw_members
            )
            return cls(members=members)
        except (TypeError, ValueError):
            if default is ...:
                raise
            return default

    # ------------------------------------------------------------------
    # Exporters — delegate to a single canonical member
    # ------------------------------------------------------------------

    def _projection_member(self) -> DataType:
        """Pick the member used for engine projection.

        * one non-null member → that member;
        * zero non-null members → ``NullType``;
        * multi-arm non-null → ``StringType`` (the same fallback the
          legacy ``from_pytype`` path collapsed mixed unions to).
        """
        non_null = self.non_null_members
        if not non_null:
            return NullType()
        if len(non_null) == 1:
            return non_null[0]
        return StringType()

    def to_arrow(self) -> pa.DataType:
        return self._projection_member().to_arrow()

    def to_polars(self) -> "polars.DataType":
        return self._projection_member().to_polars()

    def to_spark(self) -> Any:
        return self._projection_member().to_spark()

    def to_spark_name(self) -> str:
        return self._projection_member().to_spark_name()

    def to_dict(self) -> dict[str, Any]:
        base = super(UnionType, self).to_dict()
        base["members"] = [m.to_dict() for m in self.members]
        return base

    # ------------------------------------------------------------------
    # Merge — schema reconciliation
    # ------------------------------------------------------------------

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: Optional[Mode] = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "UnionType":
        if not isinstance(other, UnionType):
            return self  # defensive — dispatcher should never route here
        # Concatenate members, deduplicate by structural equality.
        seen: list[DataType] = []
        for member in (*self.members, *other.members):
            if not any(member == s for s in seen):
                seen.append(member)
        return type(self)(members=tuple(seen))

    # ------------------------------------------------------------------
    # Cast — delegate to the projection member
    # ------------------------------------------------------------------
    #
    # Sophisticated per-member runtime dispatch ("try cast as Int, if
    # that raises try cast as Str") is a future extension. For now the
    # union's ``cast_*`` methods delegate to the same single member
    # ``to_arrow`` / ``to_polars`` / ``to_spark`` pick. That keeps the
    # behaviour consistent with how the legacy ``from_pytype`` path
    # would have cast ``Optional[int]`` values — through the inner
    # ``IntegerType`` cast — without the multi-arm complexity.

    def _cast_arrow_array(self, array: pa.Array, options: "CastOptions") -> pa.Array:
        return self._projection_member()._cast_arrow_array(array, options)

    def _cast_polars_series(self, series: "polars.Series", options: "CastOptions"):
        return self._projection_member()._cast_polars_series(series, options)

    def _cast_pandas_series(self, series: Any, options: "CastOptions"):
        return self._projection_member()._cast_pandas_series(series, options)

    def _cast_spark_column(self, column: Any, options: "CastOptions"):
        return self._projection_member()._cast_spark_column(column, options)

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        if self.nullable or nullable:
            return None
        return self._projection_member().default_pyobj(nullable=False)

    def default_arrow_scalar(self, nullable: bool = True) -> pa.Scalar:
        if self.nullable or nullable:
            return pa.scalar(None, type=self.to_arrow())
        return self._projection_member().default_arrow_scalar(nullable=False)
