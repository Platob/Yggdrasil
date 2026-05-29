"""Pickle Tabular leaf over the :class:`IO` substrate.

:class:`PickleFile` is an :class:`IO` subclass with :attr:`mime_type`
set to :data:`MimeTypes.PICKLE`, so it auto-registers in the Tabular
registry and a holder with that media type dispatches here.

Unlike the columnar leaves (Parquet / Arrow IPC / CSV / …), a pickle
file holds **any** Python object — a :class:`pa.Table`, a DataFrame, a
list of dicts, or an arbitrary user object. Two surfaces sit on the
same bytes:

* **Object** — :meth:`dump` / :meth:`load` round-trip the raw object
  through :mod:`yggdrasil.pickle` (the codebase's canonical serializer),
  codec-aware via the holder's media type and zero-copy on read.

* **Tabular** — the standard :class:`Tabular` reads (``read_arrow_table``
  / ``read_polars_frame`` / ``read_pandas_frame`` / ``read_arrow_batches``)
  unpickle the object and convert it to the requested shape through
  :func:`yggdrasil.arrow.cast.any_to_arrow_table`, which routes Arrow
  inputs straight through, casts engine frames in place, and coerces
  every other tabular source via :meth:`Tabular.from_`. So a
  ``PickleFile`` of a polars frame reads back as Arrow / pandas / a
  projected column subset just like any other leaf.

Pickle carries no cheap footer, so schema discovery unpickles the
object once; writes serialize the whole object in one shot
(read-modify-rewrite for the merge modes, the same shape the
single-footer formats use).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.arrow.cast import any_to_arrow_table
from yggdrasil.arrow.ops import upsert_arrow_batches
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.enums import MimeTypes, Mode
from yggdrasil.io.base import IO

if TYPE_CHECKING:
    pass


__all__ = ["PickleFile", "PickleOptions"]


#: Modes that read existing bytes, merge with the incoming stream, and
#: rewrite in one shot. Pickle is whole-object, so APPEND / UPSERT /
#: MERGE all share the read-modify-rewrite shape — only the per-row
#: dedup strategy differs.
_MERGE_MODES = frozenset({Mode.APPEND, Mode.UPSERT, Mode.MERGE})


@dataclasses.dataclass(frozen=True, slots=True)
class PickleOptions(CastOptions):
    """:class:`CastOptions` for the pickle leaf.

    The tabular cast knobs (``target`` projection, ``row_limit``,
    ``match_by`` for merges) all apply on read / write the same as the
    other leaves. ``pickle_codec`` optionally compresses the pickle
    payload itself (a :class:`yggdrasil.enums.Codec` value); leave it
    ``None`` to let any codec on the holder's media type
    (e.g. ``.pkl.gz``) do the compression instead.
    """

    pickle_codec: "int | None" = None


class PickleFile(IO[Any, PickleOptions]):
    """:class:`Tabular` leaf for pickled Python objects."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.PICKLE

    @classmethod
    def options_class(cls):
        return PickleOptions

    # ==================================================================
    # Object surface — the pickle specialization
    # ==================================================================

    def load(self) -> Any:
        """Unpickle and return the stored object (``None`` when empty).

        Reads through :meth:`arrow_input_stream`, so a codec on the
        holder's media type is peeled transparently and the bytes are a
        zero-copy view; :mod:`yggdrasil.pickle` parses the memoryview
        directly.
        """
        if self.size_known and self.size == 0:
            return None
        from yggdrasil.pickle import loads

        try:
            with self.arrow_input_stream() as v:
                data = v.read_buffer()
        except FileNotFoundError:
            return None
        if data.size == 0:
            return None
        return loads(memoryview(data))

    def dump(self, obj: Any, *, mode: Mode = Mode.OVERWRITE) -> int:
        """Pickle *obj* into this holder; return the byte count.

        Whole-object write — ``mode`` only distinguishes OVERWRITE
        (default) from the existence guards (IGNORE / ERROR_IF_EXISTS).
        Tabular merge modes go through :meth:`write_arrow_table`.
        """
        from yggdrasil.pickle import dumps

        options = self.options_class().check(None)
        has_existing = not self.holder_is_overwrite and self.size_known and self.size > 0
        if mode is Mode.IGNORE and has_existing:
            return self.size
        if mode is Mode.ERROR_IF_EXISTS and has_existing:
            raise FileExistsError(
                f"{type(self).__name__} buffer is non-empty "
                f"({self.size} bytes); refusing to overwrite under mode={mode!r}."
            )
        payload = dumps(obj, codec=options.pickle_codec)
        # ``arrow_output_stream`` applies any holder-media-type codec and
        # bulk-commits on exit (truncating first), matching the other
        # leaves' write contract.
        self.truncate(0)
        with self.arrow_output_stream() as sink:
            sink.write(payload)
        return len(payload)

    # ==================================================================
    # Schema — unpickle once and infer
    # ==================================================================

    def _collect_schema(self, options: PickleOptions) -> Schema:
        if options.target:
            return options.target
        if self.size_known and self.size == 0:
            return Schema.empty()
        try:
            obj = self.load()
        except Exception:
            return Schema.empty()
        if obj is None:
            return Schema.empty()
        schema = Schema.from_arrow(any_to_arrow_table(obj, options).schema)
        self._persist_schema(schema)
        return schema

    # ==================================================================
    # Read path — unpickle, then cast to the requested tabular shape
    # ==================================================================

    def _read_arrow_table(self, options: PickleOptions) -> pa.Table:
        """Unpickle the object and convert it to a :class:`pa.Table`.

        :func:`any_to_arrow_table` does the heavy lifting: Arrow inputs
        pass through (with projection), engine frames cast in place, and
        any other tabular source is coerced via :meth:`Tabular.from_`.
        The cast honours ``options.target`` (column projection); a
        non-tabular object falls back to the single-row pylist wrap.
        """
        if self.size_known and self.size == 0:
            return super()._read_arrow_table(options)
        obj = self.load()
        if obj is None:
            return super()._read_arrow_table(options)
        table = any_to_arrow_table(obj, options)
        table = options.apply_post_read_table(table)
        if options.row_limit is not None and table.num_rows > options.row_limit:
            table = table.slice(0, options.row_limit)
        return table

    def _read_arrow_batches(
        self,
        options: PickleOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield the unpickled-and-cast table's batches.

        Pickle has no streaming reader — the object materializes whole —
        so this assembles the table via :meth:`_read_arrow_table` and
        re-emits its batches (the base :class:`Tabular` engine fan-out
        for polars / pandas reads bottoms out here).
        """
        if self.size_known and self.size == 0:
            return
        table = self._read_arrow_table(options)
        for batch in table.to_batches():
            yield batch

    # ==================================================================
    # Write path — pickle the (merged) table
    # ==================================================================

    def _write_arrow_table(self, table: pa.Table, options: PickleOptions) -> None:
        """Pickle *table* as the stored object.

        Fast path when the effective action replaces the buffer
        wholesale (OVERWRITE / TRUNCATE, or any mode on an empty
        buffer): pickle the table directly. Every other shape (merge
        against existing rows, guarded IGNORE / ERROR_IF_EXISTS on a
        non-empty buffer) falls through to :meth:`_write_arrow_batches`
        where the read-modify-rewrite / skip / raise logic lives.
        """
        has_existing = (
            not self.holder_is_overwrite and self.size_known and self.size > 0
        )
        truly_overwrite = (
            options.mode in (Mode.OVERWRITE, Mode.TRUNCATE) or not has_existing
        )
        if not truly_overwrite:
            return self._write_arrow_batches(iter(table.to_batches()), options)
        self.dump(options.cast_arrow_table(table), mode=Mode.OVERWRITE)
        return None

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: PickleOptions,
    ) -> None:
        """Assemble batches into one table and pickle it.

        Mode dispatch mirrors the single-footer leaves: OVERWRITE /
        AUTO / TRUNCATE pickle the incoming table; APPEND / UPSERT /
        MERGE read the existing object back as a table, merge via
        :func:`upsert_arrow_batches`, and re-pickle. IGNORE /
        ERROR_IF_EXISTS guard a non-empty holder.
        """
        action = options.mode
        if action is Mode.AUTO:
            action = Mode.UPSERT if options.match_by_keys else Mode.APPEND
        elif action is Mode.TRUNCATE:
            action = Mode.OVERWRITE

        has_existing = (
            not self.holder_is_overwrite and self.size_known and self.size > 0
        )
        if action is Mode.IGNORE:
            if has_existing:
                return None
            action = Mode.OVERWRITE
        elif action is Mode.ERROR_IF_EXISTS:
            if has_existing:
                raise FileExistsError(
                    f"{type(self).__name__} buffer is non-empty "
                    f"({self.size} bytes); refusing to overwrite under "
                    f"mode={options.mode!r}."
                )
            action = Mode.OVERWRITE

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None:
            if action is Mode.OVERWRITE:
                self.truncate(0)
            return None

        if action in _MERGE_MODES and has_existing:
            rewrite_options = options.with_target(self.collect_schema(options))
            existing = list(self._read_arrow_batches(rewrite_options))
            incoming = rewrite_options.cast_arrow_batch_iterator(
                iter([first, *iterator])
            )
            merged = upsert_arrow_batches(
                iter(existing),
                incoming,
                options.match_by_keys,
                Mode.APPEND if action is Mode.APPEND else Mode.UPSERT,
                memory_pool=options.arrow_memory_pool,
            )
            table = pa.Table.from_batches(
                list(merged), schema=rewrite_options.merged.to_arrow_schema(),
            )
            self.dump(table, mode=Mode.OVERWRITE)
            return None

        write_options = options.check_source(first.schema)
        table = pa.Table.from_batches(
            [write_options.cast_arrow_batch(b) for b in [first, *iterator]]
        )
        self.dump(table, mode=Mode.OVERWRITE)
        return None
