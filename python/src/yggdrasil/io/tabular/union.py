"""Union of multiple Tabulars with schema sync + filter pushdown.

:class:`UnionTabular` extends :class:`LazyTabular` to fan out reads
across an arbitrary number of inner Tabulars. Two things happen on
every read:

1. **Pushdown**. Pending :meth:`select` / :meth:`filter` ops form a
   "pushdown prefix" — they apply per-child *before* the union, so each
   child only reads the rows / columns the caller actually asked for.
   Anything after the first non-pushdownable op
   (:meth:`group_by`, :meth:`apply`) becomes the "post-union tail" and
   runs on the unioned LazyFrame.

2. **Schema synchronization**. Children with different column sets or
   dtypes are reconciled into a single union schema:

   - The yggdrasil-canonical schema is the per-child Arrow schema
     folded through :meth:`Schema.merge_with` — column-by-column union,
     supertype dtype merging, nullability widened.
   - On the polars side, :func:`polars.concat` with
     ``how="diagonal_relaxed"`` lines up columns by name, fills missing
     with null, and relaxes dtype mismatches. The two answers agree
     because both are doing the same column-union under the hood.

The wrapper is read-only; :meth:`_write_arrow_batches` raises rather
than silently routing writes to one child or fan-writing to all of them
(neither matches what a "union view" means as a write target).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator, Tuple

import pyarrow as pa

from yggdrasil.data.enums import Mode
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.io.tabular.lazy import LazyTabular, _Op
from yggdrasil.lazy_imports import polars_module

if TYPE_CHECKING:
    import polars as pl


__all__ = ["UnionTabular"]


# Op kinds that commute with a vertical union — they read the same way
# whether they run per-child before the concat or once on the unioned
# frame, so we push them per-child for I/O savings. Group_by and apply
# don't commute with union (group_by needs all rows; apply is opaque)
# and run post-union.
_PUSHDOWN_KINDS = frozenset({"select", "filter"})


class UnionTabular(LazyTabular):
    """Lazy union of multiple Tabulars with broadcast pushdown.

    Construction takes any iterable of inner Tabulars; the order is
    preserved on read (first child's rows come out first). All
    :class:`LazyTabular` builder methods (:meth:`select`, :meth:`where`,
    :meth:`group_by`, …) work as usual — they extend this UnionTabular's
    op chain rather than wrapping it in a new :class:`LazyTabular`.

    An empty union is legal and reads as zero rows under whatever schema
    the caller has bound on ``options`` (or empty when there is none).
    """

    def __init__(
        self,
        children: Iterable[Tabular],
        *,
        ops: Iterable[_Op] = (),
        **kwargs: Any,
    ) -> None:
        children = tuple(children)
        # LazyTabular needs an "inner" anchor for ``.inner`` introspection
        # and the default :meth:`_clone` path; first child is the natural
        # pick. Empty unions stash ``None`` and short-circuit on read.
        anchor = children[0] if children else None
        super().__init__(anchor, ops=ops, **kwargs)  # type: ignore[arg-type]
        self._children: Tuple[Tabular, ...] = children

    def __repr__(self) -> str:
        return (
            f"UnionTabular(children={list(self._children)!r}, "
            f"ops={[op[0] for op in self._ops]!r})"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def children(self) -> Tuple[Tabular, ...]:
        return self._children

    def _clone(self, ops: Tuple[_Op, ...]) -> "UnionTabular":
        return UnionTabular(self._children, ops=ops)

    # ------------------------------------------------------------------
    # Plan execution — broadcast pushdown + schema sync
    # ------------------------------------------------------------------

    @staticmethod
    def _split_ops(
        ops: Tuple[_Op, ...],
    ) -> "tuple[Tuple[_Op, ...], Tuple[_Op, ...]]":
        """Slice ops into a pushdown prefix and a post-union tail.

        The split is at the first non-pushdownable op — everything
        before commutes with the concat (so it's safe to broadcast),
        everything from there onward needs the unioned frame.
        """
        for i, op in enumerate(ops):
            if op[0] not in _PUSHDOWN_KINDS:
                return ops[:i], ops[i:]
        return ops, ()

    def _build_lazy(self, options: CastOptions) -> "pl.LazyFrame":
        pl = polars_module()
        if not self._children:
            return pl.LazyFrame()

        # Yggdrasil-canonical union schema. We compute it once up
        # front, then drive everything (alignment, concat, exposed
        # metadata) off that single source of truth — never falling
        # back to polars to *invent* the union schema.
        base_schema = self._merged_child_schema(options)

        prefix, tail = self._split_ops(self._ops)

        # Build per-child LazyFrames, align each to ``base_schema``
        # (cast dtypes, fill missing columns with null, enforce column
        # order), then broadcast the pushdown prefix per child.
        # Aligning before the prefix is important: a ``select`` in the
        # prefix can name a column that one child is missing — the
        # alignment step makes that column exist (as null) so the
        # select doesn't blow up on that branch.
        target_polars_schema = base_schema.to_polars_schema()
        children_lf = [
            self._apply_ops(
                self._align_to_schema(
                    child._scan_polars_frame(options), target_polars_schema,
                ),
                prefix,
            )
            for child in self._children
        ]

        if len(children_lf) == 1:
            unioned = children_lf[0]
        else:
            # Plain ``vertical`` concat — every child already conforms
            # to ``base_schema``, so polars doesn't need to invent any
            # alignment of its own.
            unioned = pl.concat(children_lf, how="vertical")

        return self._apply_ops(unioned, tail)

    @staticmethod
    def _align_to_schema(
        lf: "pl.LazyFrame",
        target: "pl.Schema",
    ) -> "pl.LazyFrame":
        """Cast / pad *lf* so its columns match *target* by name and dtype.

        - Columns present in *lf* but missing from *target* are dropped.
        - Columns present in *target* but missing from *lf* are added
          as ``null`` of the right dtype.
        - Existing columns are cast to the target dtype.
        - Column order matches *target*.
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
        :meth:`Schema.merge_with`, ignoring the lazy ops — this is the
        "raw" union schema, not the post-op output schema (use
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
                # every field present on either side, widen dtypes via
                # the per-field merger. UPSERT / AUTO would drop the
                # right-hand-side-only columns we explicitly want to
                # preserve.
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
        # Schema-preserving op chain → answer comes straight from our
        # canonical merged Schema, no polars roundtrip. ``filter`` is
        # the only built-in op that's guaranteed schema-preserving;
        # ``select`` / ``group_by`` / ``apply`` reshape columns and
        # need the planned LazyFrame to derive the post-op schema.
        if all(op[0] == "filter" for op in self._ops):
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
        # A union view doesn't have a single, well-defined write target.
        # Routing to the first child silently changes meaning when
        # callers reorder children; fan-writing duplicates rows. Force
        # the caller to pick a concrete sink instead.
        raise TypeError(
            f"{type(self).__name__} is a read-side union view and "
            f"doesn't accept writes. Pick a specific child Tabular "
            f"({len(self._children)} available) and write to that "
            "instead."
        )
