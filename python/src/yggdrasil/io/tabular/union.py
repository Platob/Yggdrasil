"""Union of multiple Tabulars with broadcast pushdown + schema sync.

:class:`UnionTabular` is a :class:`LazyTabular` whose source is a
*tuple of children* rather than a single Tabular. The plan executes
in two phases:

1. **Pushdown.** :meth:`ExecutionPlan.split_pushdownable` slices the
   plan at the first non-commutative op. Every commutative op
   (:class:`Select`, :class:`Filter`) runs per-child *before* the
   concat, so each source only reads the rows / columns the caller
   asked for. Non-commutative ops (:class:`GroupByAgg`,
   :class:`Apply`) run on the unioned LazyFrame.

2. **Schema sync.** The yggdrasil-canonical union schema is computed
   once via :meth:`Schema.merge_with` (mode ``APPEND``) folded across
   every child's :meth:`Tabular.collect_schema`. That single Schema
   drives the per-child alignment (cast dtypes, fill missing columns
   with null, enforce column order). Because every branch of the
   concat now matches that Schema exactly, the concat itself is a
   plain ``vertical`` — we never ask polars to invent the union
   shape. :meth:`collect_schema` short-circuits to the merged Schema
   when the plan is schema-preserving.

The wrapper is read-only; :meth:`_write_arrow_batches` raises rather
than picking one child silently or fan-writing to all of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator, Tuple

import pyarrow as pa

from yggdrasil.data.enums import Mode
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.io.tabular.execution.plan import ExecutionPlan
from yggdrasil.io.tabular.lazy import LazyTabular
from yggdrasil.lazy_imports import polars_module

if TYPE_CHECKING:
    import polars as pl


__all__ = ["UnionTabular"]


class UnionTabular(LazyTabular):
    """Lazy union of multiple Tabulars with broadcast pushdown.

    Construction takes any iterable of children; iteration order is
    preserved on read. All :class:`LazyTabular` builder methods extend
    this UnionTabular's :class:`ExecutionPlan` rather than wrapping it
    in a plain :class:`LazyTabular`.

    An empty union reads zero rows.
    """

    def __init__(
        self,
        children: Iterable[Tabular],
        *,
        plan: "ExecutionPlan | Iterable | None" = None,
        **kwargs: Any,
    ) -> None:
        children = tuple(children)
        # LazyTabular needs a "source" anchor for ``.source`` and the
        # default :meth:`_clone` path; first child is the natural pick.
        # Empty unions stash ``None`` and short-circuit on read.
        anchor = children[0] if children else None
        super().__init__(anchor, plan=plan, **kwargs)  # type: ignore[arg-type]
        self._children: Tuple[Tabular, ...] = children

    def __repr__(self) -> str:
        return (
            f"UnionTabular(children={list(self._children)!r}, "
            f"plan={self._plan!r})"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def children(self) -> Tuple[Tabular, ...]:
        return self._children

    def _clone(self, plan: ExecutionPlan) -> "UnionTabular":
        return UnionTabular(self._children, plan=plan)

    # ------------------------------------------------------------------
    # Plan execution — broadcast pushdown + yggdrasil schema sync
    # ------------------------------------------------------------------

    def _build_lazy(self, options: CastOptions) -> "pl.LazyFrame":
        pl = polars_module()
        if not self._children:
            return pl.LazyFrame()

        # Yggdrasil-canonical union schema. Computed once, used for
        # alignment + exposed via :meth:`merged_schema` — no polars
        # fallback for the union shape itself.
        base_schema = self._merged_child_schema(options)
        target_polars_schema = base_schema.to_polars_schema()

        prefix, tail = self._plan.split_pushdownable()

        # Per child: scan, align to base_schema (cast dtypes, fill
        # missing as null, enforce column order), apply pushdown
        # prefix. Aligning before the prefix means a ``select`` in the
        # prefix can name a column that one child is missing — the
        # alignment step has already added it as null.
        children_lf = [
            prefix.apply_polars(
                self._align_to_schema(
                    child._scan_polars_frame(options),
                    target_polars_schema,
                )
            )
            for child in self._children
        ]

        if len(children_lf) == 1:
            unioned = children_lf[0]
        else:
            # Plain ``vertical`` — every child already conforms to
            # ``base_schema``, so polars doesn't need to invent any
            # alignment of its own.
            unioned = pl.concat(children_lf, how="vertical")

        return tail.apply_polars(unioned)

    @staticmethod
    def _align_to_schema(
        lf: "pl.LazyFrame",
        target: "pl.Schema",
    ) -> "pl.LazyFrame":
        """Cast / pad *lf* so its columns match *target* by name and dtype.

        Columns not in *target* are dropped, missing ones are added
        as ``null`` of the right dtype, dtype mismatches are cast,
        and the final order matches *target*.
        """
        pl = polars_module()
        current = lf.collect_schema()
        projections: list[Any] = []
        for name, dtype in target.items():
            if name in current:
                if current[name] != dtype:
                    projections.append(pl.col(name).cast(dtype))
                else:
                    projections.append(pl.col(name))
            else:
                projections.append(pl.lit(None, dtype=dtype).alias(name))
        return lf.select(projections)

    def merged_schema(self, options: "CastOptions | None" = None) -> Schema:
        """Yggdrasil-canonical union schema across every child.

        Folds each child's ``collect_schema`` through
        :meth:`Schema.merge_with`, ignoring the lazy plan — this is
        the "raw" union schema, not the post-plan output schema (use
        :meth:`collect_schema` for that).
        """
        return self._merged_child_schema(self.check_options(options))

    def _merged_child_schema(self, options: CastOptions) -> Schema:
        merged: "Schema | None" = None
        for child in self._children:
            child_schema = child._collect_schema(options)
            if merged is None:
                merged = child_schema
            else:
                # ``Mode.APPEND`` is the true-union semantics: keep
                # every field present on either side, widen dtypes
                # via the per-field merger. UPSERT / AUTO would drop
                # right-hand-side-only columns.
                merged = merged.merge_with(
                    child_schema, inplace=False, mode=Mode.APPEND,
                )
        return merged if merged is not None else Schema.empty()

    # ------------------------------------------------------------------
    # Tabular contract — read hooks (always go through _build_lazy)
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self, options: CastOptions,
    ) -> Iterator[pa.RecordBatch]:
        if not self._children:
            return
        lf = self._build_lazy(options)
        table: pa.Table = lf.collect().to_arrow()
        row_size = getattr(options, "row_size", None) or None
        for batch in table.to_batches(max_chunksize=row_size):
            yield options.cast_arrow_tabular(batch)

    def _scan_polars_frame(self, options: CastOptions) -> "pl.LazyFrame":
        return self._build_lazy(options)

    def _read_polars_frame(self, options: CastOptions) -> "pl.DataFrame":
        return self._build_lazy(options).collect()

    def _collect_schema(self, options: CastOptions) -> Schema:
        # Schema-preserving plan → answer comes straight from the
        # canonical merged Schema, no polars roundtrip.
        if self._plan.is_schema_preserving():
            return self._merged_child_schema(options)

        pl = polars_module()
        lf = self._build_lazy(options)
        pl_schema = lf.collect_schema()
        if isinstance(pl_schema, pl.Schema):
            return Schema.from_polars(pl_schema)
        return Schema.from_polars(lf)

    # ------------------------------------------------------------------
    # Tabular contract — writes
    # ------------------------------------------------------------------

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        # A union view doesn't have a single, well-defined write
        # target. Force the caller to pick a concrete sink instead.
        raise TypeError(
            f"{type(self).__name__} is a read-side union view and "
            f"doesn't accept writes. Pick a specific child Tabular "
            f"({len(self._children)} available) and write to that "
            "instead."
        )
