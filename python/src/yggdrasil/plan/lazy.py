""":class:`LazyTabular` — defers execution until a ``read_*`` call.

Wraps a concrete source :class:`Tabular` and a :class:`SelectPlan`.
Transformation methods mutate the plan; every ``read_*`` materialises.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, Iterator, TypeVar

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular.base import Tabular

from .execution_plan import SelectPlan

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema

O = TypeVar("O", bound=CastOptions)


class LazyTabular(Tabular[O], Generic[O]):
    """Lazy wrapper: accumulates a plan, executes on every read."""

    def __init__(self, source: Tabular[O], plan: SelectPlan[O] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._source = source
        self._plan: SelectPlan[O] = plan if plan is not None else SelectPlan()

    @property
    def source(self) -> Tabular[O]: return self._source
    @property
    def plan(self) -> SelectPlan[O]: return self._plan
    @classmethod
    def options_class(cls) -> type[O]: return CastOptions  # type: ignore[return-value]

    def collect(self, options: "O | None" = None, **kw: Any) -> Tabular:
        result = self._plan.execute(self._source)
        return result.read_arrow_tabular(options) if options else result

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        yield from self._plan.execute(self._source).read_arrow_batches(options)

    def _write_arrow_batches(self, batches: Iterable[pa.RecordBatch], options: O) -> None:
        raise TypeError("LazyTabular is read-only.")

    def _collect_schema(self, options: O) -> "Schema":
        return self._plan.execute(self._source).collect_schema(options)

    # Lazy transforms — mutate plan, return self
    def select(self, *columns) -> "LazyTabular[O]": self._plan.select(*columns); return self
    def drop(self, *columns) -> "LazyTabular[O]": self._plan.drop(*columns); return self
    def filter(self, predicate) -> "LazyTabular[O]": self._plan.filter(predicate); return self
    def unique(self, by) -> "LazyTabular[O]": self._plan.unique(by); return self
    def limit(self, n) -> "LazyTabular[O]": self._plan.limit(n); return self
    def offset(self, n) -> "LazyTabular[O]": self._plan.offset(n); return self
    def cast(self, options=None, **kw) -> "LazyTabular[O]": self._plan.cast(options, **kw); return self

    def group_by(self, *keys, aggregations=None) -> "LazyTabular[O]":
        self._plan.group_by(*keys, aggregations=aggregations); return self

    def having(self, predicate) -> "LazyTabular[O]":
        self._plan.having(predicate); return self

    def order_by(self, *keys) -> "LazyTabular[O]":
        self._plan.order_by(*keys); return self

    def with_cte(self, name, plan) -> "LazyTabular[O]":
        self._plan.with_cte(name, plan); return self

    def join(self, right, on, how="inner", *, suffix="_right") -> "LazyTabular[O]":
        self._plan.join(right, on=on, how=how, suffix=suffix); return self

    def union(self, other: Any, *, mode=None) -> "LazyTabular[O]":
        if isinstance(other, Tabular):
            self._plan.union(other, mode=mode); return self
        return super().union(other, mode=mode)

    def resample(self, on, sampling, *, partition_by=None, fill_strategy="ffill") -> "LazyTabular[O]":
        self._plan.resample(on, sampling, partition_by=partition_by, fill_strategy=fill_strategy)
        return self

    def lazy(self) -> "LazyTabular[O]": return self

    def copy(self) -> "LazyTabular[O]":
        return LazyTabular(self._source, self._plan.copy())

    def __repr__(self) -> str:
        return f"LazyTabular(source={self._source!r}, plan={self._plan!r})"
