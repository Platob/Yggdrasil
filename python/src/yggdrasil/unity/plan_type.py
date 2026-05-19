"""Classification enums for :class:`ExecutionStatement` and :class:`ExecutionPlan`.

:class:`PlanTypeId` is the per-statement identifier; each
:class:`ExecutionStatement` subclass declares one through a ``plan_type_id``
ClassVar. :class:`PlanCategory` is the coarser grouping (DDL / DML /
DQL / META) callers reach for when filtering a heterogeneous plan
("does this plan mutate anything?", "is this purely read-only?").

Numeric ranges
--------------
The integer values are deliberately spaced so a category collapses to
a single integer prefix:

* ``100–199`` :attr:`PlanCategory.DDL`  — schema creation / removal.
* ``200–299`` :attr:`PlanCategory.DML`  — row-level mutation.
* ``300–399`` :attr:`PlanCategory.DQL`  — row-level read.
* ``400–499`` :attr:`PlanCategory.META` — metadata listings.

The :attr:`PlanTypeId.category` / :attr:`PlanTypeId.is_mutation` /
:attr:`PlanTypeId.is_query` helpers derive their answer from the
integer prefix, so adding a new statement only requires picking a
slot in the right range.
"""

from __future__ import annotations

from enum import IntEnum


__all__ = ["PlanCategory", "PlanTypeId"]


class PlanCategory(IntEnum):
    """Coarse grouping of :class:`ExecutionStatement` shapes."""

    DDL = 1
    DML = 2
    DQL = 3
    META = 4

    @property
    def is_mutation(self) -> bool:
        """``True`` for categories that change backend state (DDL / DML)."""
        return self in (PlanCategory.DDL, PlanCategory.DML)

    @property
    def is_query(self) -> bool:
        """``True`` for read-only categories (DQL / META)."""
        return self in (PlanCategory.DQL, PlanCategory.META)


class PlanTypeId(IntEnum):
    """Stable identifier for every concrete :class:`ExecutionStatement` shape.

    The numeric prefix encodes :class:`PlanCategory`:

    * ``1xx`` → :attr:`PlanCategory.DDL`
    * ``2xx`` → :attr:`PlanCategory.DML`
    * ``3xx`` → :attr:`PlanCategory.DQL`
    * ``4xx`` → :attr:`PlanCategory.META`
    """

    # ── DDL ─────────────────────────────────────────────────────────────
    CREATE_CATALOG = 100
    CREATE_SCHEMA = 101
    CREATE_TABLE = 102
    CREATE_VIEW = 103
    DROP_CATALOG = 110
    DROP_SCHEMA = 111
    DROP_TABLE = 112
    DROP_VIEW = 113

    # ── DML ─────────────────────────────────────────────────────────────
    INSERT = 200
    # Reserved for future mutations:
    #   UPDATE = 201
    #   DELETE = 202
    #   MERGE  = 203

    # ── DQL ─────────────────────────────────────────────────────────────
    SELECT = 300

    # ── META ────────────────────────────────────────────────────────────
    SHOW_CATALOGS = 400
    SHOW_SCHEMAS = 401
    SHOW_TABLES = 402
    SHOW_VIEWS = 403

    # ── derived properties ─────────────────────────────────────────────

    @property
    def category(self) -> PlanCategory:
        """Coarse :class:`PlanCategory` derived from the numeric prefix."""
        prefix = self.value // 100
        # IntEnum lookup avoids a chained ``if`` ladder and stays O(1).
        return PlanCategory(prefix)

    @property
    def is_mutation(self) -> bool:
        """``True`` for DDL / DML — anything that changes backend state."""
        return self.category.is_mutation

    @property
    def is_query(self) -> bool:
        """``True`` for DQL / META — read-only operations."""
        return self.category.is_query

    @property
    def is_create(self) -> bool:
        return 100 <= self.value < 110

    @property
    def is_drop(self) -> bool:
        return 110 <= self.value < 120

    @property
    def is_show(self) -> bool:
        return self.category is PlanCategory.META
