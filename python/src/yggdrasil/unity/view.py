"""Abstract Unity view — read-only :class:`Tabular` over a source reference.

A view is metadata-only: its info records a dotted ``catalog.schema.name``
pointing at a registered table (or another view), plus an optional SQL
projection. Reads resolve the source through the engine and stream its
batches; writes are unsupported.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.unity.base import UnityResource
from yggdrasil.unity.info import ViewInfo

if TYPE_CHECKING:
    from yggdrasil.unity.schema import UnitySchema


__all__ = ["UnityView"]


logger = logging.getLogger(__name__)


class UnityView(UnityResource, Tabular[CastOptions]):
    """Abstract view — resolves to another :class:`Tabular` on read."""

    def __init__(self) -> None:
        Tabular.__init__(self)

    # ── identity ────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def schema_handle(self) -> "UnitySchema":
        """The :class:`UnitySchema` owning this view."""

    @property
    def catalog_name(self) -> str:
        return self.schema_handle.catalog_name

    @property
    def schema_name(self) -> str:
        return self.schema_handle.name

    @property
    def full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}.{self.name}"

    # ── info ────────────────────────────────────────────────────────────

    @abstractmethod
    def _read_info(self) -> ViewInfo: ...

    @property
    def info(self) -> ViewInfo:  # type: ignore[override]
        return super().info  # type: ignore[return-value]

    # ── source resolution ───────────────────────────────────────────────

    @abstractmethod
    def _resolve_source(self) -> Tabular:
        """Return the live :class:`Tabular` this view points at.

        Backends use :attr:`ViewInfo.source_full_name` to look the
        target up through their parent engine. Raises
        :class:`FileNotFoundError` when the target no longer exists.
        """

    @property
    def schema(self) -> Schema:
        """Pass-through of the source's schema."""
        return self._resolve_source().collect_schema()

    def _collect_schema(self, options: CastOptions) -> Schema:
        return self.schema

    # ── Tabular contract ────────────────────────────────────────────────

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        source = self._resolve_source()
        yield from source.read_arrow_batches(options)

    def _write_arrow_batches(
        self, batches: Iterable[pa.RecordBatch], options: CastOptions,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} {self.full_name!r} is read-only. "
            f"Write into its source table {self.info.source_full_name!r} "
            "directly."
        )
