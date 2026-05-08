"""Format-specific :class:`LazyTabular` subclasses.

One :class:`LazyTabular` subclass per concrete :class:`Tabular`
leaf, wired through :attr:`LazyTabular.source_cls` so
:func:`yggdrasil.io.tabular.lazy.lazy_for` (called by
:meth:`Tabular.execute_plan`) picks the right wrapper for the
source type.

The pushdown story
------------------

Primitive-format leaves (Parquet, Arrow IPC, CSV, NDJSON, JSON)
already expose format-aware polars scanners through their
:meth:`Tabular._scan_polars_frame` overrides — ``pl.scan_parquet``,
``pl.scan_ipc``, ``pl.scan_csv``, ``pl.scan_ndjson``,
``pl.scan_pyarrow_dataset``. The base :class:`LazyTabular` lifts
those into a polars LazyFrame and the planner pushes column
projection and predicate filters at the format level. The Lazy IO
subclasses defined here are mostly type-narrowing markers — the
plan dispatch is concrete, but the per-format pushdown itself is
inherited from the existing scanners.

:class:`LazyFolderIO` is the one with new behavior on top of the
base: a folder read goes through :class:`UnionTabular` so
commutative ops (:class:`Filter`, :class:`Select`) push *into*
each child's format-aware scanner rather than running over an
eagerly materialized record-batch reader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from yggdrasil.data.options import CastOptions
from yggdrasil.io.nested.folder_io import FolderIO
from yggdrasil.io.primitive.arrow_ipc_io import ArrowIPCIO
from yggdrasil.io.primitive.csv_io import CsvIO
from yggdrasil.io.primitive.json_io import JsonIO
from yggdrasil.io.primitive.ndjson_io import NDJsonIO
from yggdrasil.io.primitive.parquet_io import ParquetIO
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.io.tabular.lazy import LazyTabular
from yggdrasil.lazy_imports import polars_module

if TYPE_CHECKING:
    import polars as pl


__all__ = [
    "LazyArrowIPCIO",
    "LazyCsvIO",
    "LazyFolderIO",
    "LazyJsonIO",
    "LazyNDJsonIO",
    "LazyParquetIO",
]


class LazyParquetIO(LazyTabular):
    """:class:`LazyTabular` over a :class:`ParquetIO` source.

    The base build path lifts the source through
    :meth:`ParquetIO._scan_polars_frame` (``pl.scan_parquet(path)``);
    polars pushes column projection and predicate filters into the
    parquet reader at plan time. For local-path holders this also
    enables row-group pruning via the parquet footer statistics.
    """

    source_cls: ClassVar["type[Tabular]"] = ParquetIO

    @property
    def source(self) -> ParquetIO:
        return self._source  # type: ignore[return-value]


class LazyArrowIPCIO(LazyTabular):
    """:class:`LazyTabular` over an :class:`ArrowIPCIO` source.

    The base build path lifts the source through
    :meth:`Tabular._scan_polars_frame`'s default
    (``pl.scan_pyarrow_dataset``); polars pushes projection and
    predicates into the dataset reader.
    """

    source_cls: ClassVar["type[Tabular]"] = ArrowIPCIO

    @property
    def source(self) -> ArrowIPCIO:
        return self._source  # type: ignore[return-value]


class LazyCsvIO(LazyTabular):
    """:class:`LazyTabular` over a :class:`CsvIO` source.

    Routes through :meth:`CsvIO._scan_polars_frame`
    (``pl.scan_csv(path)`` for local-path holders), which gives
    polars enough scaffolding to push projection and filters into
    the CSV reader.
    """

    source_cls: ClassVar["type[Tabular]"] = CsvIO

    @property
    def source(self) -> CsvIO:
        return self._source  # type: ignore[return-value]


class LazyNDJsonIO(LazyTabular):
    """:class:`LazyTabular` over an :class:`NDJsonIO` source.

    Routes through :meth:`NDJsonIO._scan_polars_frame`
    (``pl.scan_ndjson(path)`` for local-path holders).
    """

    source_cls: ClassVar["type[Tabular]"] = NDJsonIO

    @property
    def source(self) -> NDJsonIO:
        return self._source  # type: ignore[return-value]


class LazyJsonIO(LazyTabular):
    """:class:`LazyTabular` over a :class:`JsonIO` source."""

    source_cls: ClassVar["type[Tabular]"] = JsonIO

    @property
    def source(self) -> JsonIO:
        return self._source  # type: ignore[return-value]


class LazyFolderIO(LazyTabular):
    """:class:`LazyTabular` over a :class:`FolderIO` source.

    Materializes the folder's children once per build and delegates
    to :class:`UnionTabular`, which slices the plan at the first
    non-commutative op via
    :meth:`ExecutionPlan.split_pushdownable`. The commutative
    prefix (``Select`` / ``Filter``) runs *inside* each child's
    format-aware polars scanner, so a ``where`` on a folder of
    parquet files compiles to per-leaf
    ``pl.scan_parquet(...).filter(...)`` and polars pushes the
    predicate into the parquet reader. The non-commutative tail
    (``GroupByAgg`` / ``Apply`` / ``Join``) runs once over the
    unioned LazyFrame.

    An empty folder yields an empty LazyFrame.
    """

    source_cls: ClassVar["type[Tabular]"] = FolderIO

    @property
    def source(self) -> FolderIO:
        return self._source  # type: ignore[return-value]

    def _build_lazy(self, options: CastOptions) -> "pl.LazyFrame":
        from yggdrasil.io.tabular.union import UnionTabular

        children = tuple(self._source.iter_children())
        if not children:
            return polars_module().LazyFrame()
        return UnionTabular(children, plan=self._plan)._build_lazy(options)
