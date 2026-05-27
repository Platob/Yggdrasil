""":class:`LazyTabular` — a :class:`Tabular` that defers execution.

Wraps a concrete source :class:`Tabular` and an :class:`SelectPlan`.
Transformation methods (``select``, ``filter``, ``join``, …) mutate
the plan instead of touching data.  Every ``read_*`` surface triggers
:meth:`SelectPlan.execute`, materialising the result on demand.

Create via :meth:`Tabular.lazy`::

    lazy = my_tabular.lazy()
    lazy.select("a", "b").filter(col("a") > 10)
    table = lazy.read_arrow_table()       # plan executes here
    table2 = lazy.read_arrow_table()      # plan executes again

Or construct directly::

    from yggdrasil.plan import LazyTabular, SelectPlan
    plan = SelectPlan()
    plan.select("a").filter("x > 0")
    lazy = LazyTabular(source, plan)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterator,
    TypeVar,
)

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular.base import Tabular

from .execution_plan import SelectPlan

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema
    from yggdrasil.enums import JoinType, Mode, ModeLike
    from yggdrasil.execution.expr import PredicateLike

O = TypeVar("O", bound=CastOptions)


class LazyTabular(Tabular[O], Generic[O]):
    """Lazy wrapper: accumulates a plan, executes on every read."""

    def __init__(
        self,
        source: Tabular[O],
        plan: SelectPlan[O] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._source: Tabular[O] = source
        self._plan: SelectPlan[O] = plan if plan is not None else SelectPlan()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def source(self) -> Tabular[O]:
        return self._source

    @property
    def plan(self) -> SelectPlan[O]:
        return self._plan

    @classmethod
    def options_class(cls) -> type[O]:
        return CastOptions  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Plan execution — materialises on every call
    # ------------------------------------------------------------------

    def _execute_plan(self) -> Tabular:
        return self._plan.execute(self._source)

    def collect(self, options: "O | None" = None, **kwargs: Any) -> Tabular:
        """Execute the plan and return the materialised :class:`Tabular`."""
        result = self._execute_plan()
        if options is not None:
            return result.read_arrow_tabular(options)
        return result

    # ------------------------------------------------------------------
    # Tabular abstract hooks — delegate to plan execution
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        result = self._execute_plan()
        yield from result.read_arrow_batches(options)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        raise TypeError(
            "LazyTabular is read-only. Write to the source Tabular "
            "directly or collect() first."
        )

    # ------------------------------------------------------------------
    # Schema — inferred from plan execution
    # ------------------------------------------------------------------

    def _collect_schema(self, options: O) -> "Schema":
        result = self._execute_plan()
        return result.collect_schema(options)

    # ------------------------------------------------------------------
    # Lazy transformation methods — mutate plan, return self
    # ------------------------------------------------------------------

    def select(self, *columns: "str | Any") -> "LazyTabular[O]":
        """Set the column projection on the plan."""
        self._plan.select(*columns)
        return self

    def drop(self, *columns: "str | Any") -> "LazyTabular[O]":
        """Set columns to drop on the plan."""
        self._plan.drop(*columns)
        return self

    def filter(self, predicate: "PredicateLike") -> "LazyTabular[O]":
        """Add a pushdown predicate (ANDed with existing)."""
        self._plan.filter(predicate)
        return self

    def join(
        self,
        right: "Tabular",
        on: "str | list[str] | Any",
        how: "str | JoinType" = "inner",
        *,
        suffix: str = "_right",
    ) -> "LazyTabular[O]":
        """Append a join to the plan."""
        self._plan.join(right, on=on, how=how, suffix=suffix)
        return self

    def union(self, other: Any, *, mode: "ModeLike | None" = None) -> "LazyTabular[O]":
        """Append a UNION ALL to the plan."""
        from yggdrasil.io.tabular.base import Tabular as _Tab

        if isinstance(other, _Tab):
            self._plan.union(other, mode=mode)
            return self
        return super().union(other, mode=mode)

    def unique(self, by: "str | Any | list[Any]") -> "LazyTabular[O]":
        """Set dedup keys on the plan."""
        self._plan.unique(by)
        return self

    def resample(
        self,
        on: "str | Any",
        sampling: "int | float | Any",
        *,
        partition_by: "str | Any | list[Any] | None" = None,
        fill_strategy: "str | None" = "ffill",
    ) -> "LazyTabular[O]":
        """Set a time-grid resample on the plan."""
        self._plan.resample(
            on, sampling,
            partition_by=partition_by,
            fill_strategy=fill_strategy,
        )
        return self

    def cast(self, options: "O | None" = None, **kwargs: Any) -> "LazyTabular[O]":
        """Set a cast options override on the plan."""
        self._plan.cast(options, **kwargs)
        return self

    def limit(self, n: int | None) -> "LazyTabular[O]":
        """Cap the result to *n* rows."""
        self._plan.limit(n)
        return self

    # ------------------------------------------------------------------
    # Lazy returns self — no double-wrapping
    # ------------------------------------------------------------------

    def lazy(self) -> "LazyTabular[O]":
        """Return self — already lazy."""
        return self

    # ------------------------------------------------------------------
    # Copy — fork the plan so mutations don't alias
    # ------------------------------------------------------------------

    def copy(self) -> "LazyTabular[O]":
        """Return a new :class:`LazyTabular` sharing the source but
        with an independent copy of the plan."""
        return LazyTabular(self._source, self._plan.copy())

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"LazyTabular(source={self._source!r}, plan={self._plan!r})"
