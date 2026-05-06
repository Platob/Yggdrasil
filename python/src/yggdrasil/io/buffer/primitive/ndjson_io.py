"""Newline-delimited JSON I/O for :class:`PrimitiveIO`.

:class:`NDJsonIO` is the concrete leaf for NDJSON / JSONL / "JSON
Lines" — one JSON object per line, no array wrapper, no trailing
comma. This is the only JSON shape that's natively streamable and
appendable, which matches the rest of the leaf set's contract.

Pretty-printed JSON arrays of objects (``[{...}, {...}]``) are out
of scope at the leaf level: they're not streamable without
parsing the whole document, and they're not appendable without
rewriting the closing bracket. Transform upstream into NDJSON or
use a different format.

Reads
-----

Reads use :func:`pa.json.open_json`, which streams record batches
from NDJSON input. The reader auto-infers types from the first
batch's worth of rows and applies them to subsequent batches —
type drift across the file produces parse errors, same as CSV.
Schema collection is "read the first batch's schema."

Writes
------

pyarrow has no NDJSON writer (only a reader). We emit lines
manually via :func:`json.dumps` per row plus a configurable line
ending. For high-volume writes consider Parquet or Arrow IPC —
NDJSON's per-row Python overhead is significant.

Save modes
----------

OVERWRITE truncates and writes fresh.

APPEND seeks to end and writes — NDJSON's line-per-record framing
makes byte-level concatenation a valid append. The leaf supports
this natively (no read-modify-write), which is the same shape as
CSV.

UPSERT goes through :meth:`DataIO._arrow_upsert_via_rewrite` —
NDJSON has no native row identity, but the rewrite helper handles
it via read-modify-write.

Native engine dispatch
----------------------

When the buffer is backed by a real local path (no compression
wrapper, no in-flight target cast, non-empty),
:meth:`_read_arrow_dataset` dispatches to
``pds.dataset(format="json")`` and the polars overrides use
:func:`pl.scan_ndjson` / :func:`pl.read_ndjson`. Same gating as
CSV — see :meth:`_can_use_native_scanner`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.json as pa_json

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import (
    polars_module,
    pyarrow_dataset_module,
)
from yggdrasil.io.buffer.bytes_io import BytesIO

if TYPE_CHECKING:
    import polars as pl
    import pyarrow.dataset as pds


__all__ = ["NDJsonIO", "NDJsonOptions"]


# ---------------------------------------------------------------------------
# NDJsonOptions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class NDJsonOptions(CastOptions):
    """:class:`CastOptions` extended with NDJSON-specific knobs.

    Defaults match the canonical NDJSON shape: UTF-8, ``\\n``
    line terminator, threaded reader. Tweak ``encoding`` for
    locale-encoded files; tweak ``line_ending`` to ``"\\r\\n"`` for
    Windows-flavored files (the reader is line-ending agnostic but
    the writer needs to be told).
    """

    # Reader knobs — pyarrow's JSON reader has surprisingly few.
    use_threads: bool = True
    block_size: "int | None" = None  # bytes per parse block; None = default

    # Writer knobs — we emit lines via ``json.dumps`` per row.
    encoding: str = "utf-8"
    line_ending: str = "\n"
    ensure_ascii: bool = False
    sort_keys: bool = False

    def to_read_options(self) -> "pa_json.ReadOptions":
        kwargs = {"use_threads": self.use_threads}
        if self.block_size is not None:
            kwargs["block_size"] = self.block_size
        return pa_json.ReadOptions(**kwargs)

    def to_parse_options(self) -> "pa_json.ParseOptions":
        # ParseOptions is tiny — newlines_in_values defaults to
        # False which matches the strict NDJSON contract (one
        # object per physical line).
        return pa_json.ParseOptions()


# ---------------------------------------------------------------------------
# NDJsonIO
# ---------------------------------------------------------------------------


class NDJsonIO(BytesIO):
    """:class:`PrimitiveIO` for newline-delimited JSON.

    Reads stream via :func:`pa_json.open_json`, which yields record
    batches as the parser consumes the input. Writes emit one line
    per row via :func:`json.dumps`.
    """

    # No cached reader — NDJSON has no footer to amortize, same as CSV.
    _FINAL_TABULAR_IO: ClassVar[bool] = True

    # Polars / pyarrow-dataset have native NDJSON scanners we can
    # push projection / predicate into when the buffer is bound to
    # a local path; ``_can_use_native_scanner`` gates the dispatch.
    _NATIVE_SCANNER_OK: ClassVar[bool] = True

    # ==================================================================
    # Class-level config
    # ==================================================================

    @classmethod
    def default_media_type(cls):
        return MimeTypes.NDJSON

    @classmethod
    def options_class(cls):
        return NDJsonOptions

    # ==================================================================
    # Schema — read the first batch
    # ==================================================================

    def _collect_schema(self, options: NDJsonOptions) -> Schema:
        """Read the schema from the first parsed batch.

        NDJSON has no metadata; the parser must read at least one
        block of input to type-infer. Empty buffer short-circuits.
        """
        if self.is_empty():
            return Schema.empty()

        with self._reading_context(options) as io:
            source = io.arrow_io(mode="rb")
            reader = pa_json.open_json(
                source,
                read_options=options.to_read_options(),
                parse_options=options.to_parse_options(),
            )
            try:
                return Schema.from_arrow(reader.schema)
            finally:
                reader.close()

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: NDJsonOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from a streaming NDJSON parse.

        Empty buffer short-circuits; pyarrow's JSON reader raises
        on zero-byte input.

        ``options.cast_arrow_tabular`` runs unconditionally per
        batch — internal no-op when ``target_field`` is None.
        """
        if self.is_empty():
            return

        with self._reading_context(options) as io:
            source = io.arrow_io(mode="rb")

            try:
                reader = pa_json.open_json(
                    source,
                    read_options=options.to_read_options(),
                    parse_options=options.to_parse_options(),
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to open JSON reader for NDJSON source: {io}\n{io.head(100).decode()}"
                ) from e
            try:
                for batch in reader:
                    yield options.cast_arrow_tabular(batch)
            finally:
                reader.close()

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: NDJsonOptions,
    ) -> None:
        """Persist Arrow record batches as NDJSON.

        Save-mode resolution:

        - **OVERWRITE / AUTO / TRUNCATE** — truncate + emit fresh
          lines.
        - **APPEND** — seek to end, emit additional lines. The
          line-ending separates appended runs naturally; no
          header-skip equivalent (NDJSON has no header).
        - **UPSERT** — :meth:`DataIO._arrow_upsert_via_rewrite`.
        - **IGNORE** — skip.

        pyarrow has no NDJSON writer, so we emit lines manually
        via :func:`json.dumps`. The per-row Python overhead is the
        cost of NDJSON's portability — for high-volume writes,
        prefer Parquet or Arrow IPC.
        """
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.UPSERT:
            return self._arrow_upsert_via_rewrite(batches, options)
        if action not in (Mode.OVERWRITE, Mode.APPEND):
            raise NotImplementedError(
                f"{type(self).__name__}._write_arrow_batches handles "
                f"OVERWRITE / APPEND / UPSERT; got resolved action "
                f"{action!r}."
            )

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None:
            return

        if options.target_field is not None:
            first = options.cast_arrow_tabular(first)

        # APPEND on an empty buffer is overwrite — there's no
        # existing content to append to.
        is_append = action is Mode.APPEND and not self.is_empty()

        lifecycle = options.copy(
            truncate_before_write=not is_append,
            write_seek=-1 if is_append else None,
        )

        line_sep = options.line_ending.encode(options.encoding)

        def emit(batch: pa.RecordBatch, sink) -> None:
            """Encode one record batch as NDJSON lines.

            Each row becomes ``json.dumps(row) + line_ending``.
            """
            for row in batch.to_pylist():
                line = json.dumps(
                    row,
                    ensure_ascii=options.ensure_ascii,
                    sort_keys=options.sort_keys,
                    default=_json_default,
                )
                sink.write(line.encode(options.encoding))
                sink.write(line_sep)

        with self._writing_context(lifecycle) as io:
            with contextlib.ExitStack() as stack:
                sink = io.arrow_io(mode="ab" if is_append else "wb")
                stack.callback(sink.close)

                emit(first, sink)
                for batch in iterator:
                    if options.target_field is not None:
                        batch = options.cast_arrow_tabular(batch)
                    emit(batch, sink)

    # ==================================================================
    # Native engine overrides
    # ==================================================================

    def _can_use_native_scanner(self, options: NDJsonOptions) -> bool:
        """True iff the native NDJSON scanners can serve *options*.

        Same gating shape as CSV / Parquet / IPC — see those leaves
        for the full rationale. Briefly: empty buffer / target_field
        / codec / non-local path all force fallback to the generic
        shim that goes through :meth:`_read_arrow_batches`.
        """
        if not type(self)._NATIVE_SCANNER_OK:
            return False
        if self.is_empty():
            return False
        if options.target_field is not None:
            return False
        if self.codec is not None:
            return False
        if self.path is None:
            return False
        if not self.path.is_local:
            return False
        return True

    def _read_arrow_dataset(self, options: NDJsonOptions) -> "pds.Dataset":
        """Native :class:`pyarrow.dataset.Dataset` over the NDJSON file.

        ``format="json"`` in pyarrow's dataset API means NDJSON
        (line-delimited). Pretty-printed JSON arrays aren't
        supported by the dataset reader either way, so this matches
        the leaf's contract.
        """
        if not self._can_use_native_scanner(options):
            return super()._read_arrow_dataset(options)

        pds = pyarrow_dataset_module()
        return pds.dataset(self.path.__fspath__(), format="json")

    def _scan_polars_frame(self, options: NDJsonOptions) -> "pl.LazyFrame":
        """Native :func:`polars.scan_ndjson` LazyFrame.

        Polars' rust-native NDJSON scanner pushes projections /
        filters into the parser at plan time, faster than the
        ``scan_pyarrow_dataset`` shim.
        """
        if not self._can_use_native_scanner(options):
            return super()._scan_polars_frame(options)

        pl = polars_module()
        return pl.scan_ndjson(self.path.__fspath__())

    def _read_polars_frame(self, options: NDJsonOptions) -> "pl.DataFrame":
        """Native :func:`polars.read_ndjson` eager :class:`pl.DataFrame`.

        Skips the Arrow Table → ``pl.from_arrow`` conversion in the
        base path.
        """
        if not self._can_use_native_scanner(options):
            return super()._read_polars_frame(options)

        pl = polars_module()
        return pl.read_ndjson(self.path.__fspath__())


# ---------------------------------------------------------------------------
# json.dumps fallback
# ---------------------------------------------------------------------------


def _json_default(obj):
    """Fallback serializer for types ``json.dumps`` doesn't handle.

    pyarrow's ``RecordBatch.to_pylist`` produces Python natives for
    most types but emits :class:`datetime.date` / :class:`datetime.datetime`
    / :class:`decimal.Decimal` / :class:`bytes` directly — none of
    which the stdlib JSON encoder accepts.

    Conversions:

    - ``datetime`` / ``date`` / ``time`` → ISO-8601 string
    - ``Decimal`` → string (preserves precision; floats would lose it)
    - ``bytes`` / ``bytearray`` → base64-encoded ASCII
    - anything else → :class:`TypeError` (caller bug)
    """
    import base64
    import datetime
    import decimal

    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    raise TypeError(
        f"NDJSON encoder cannot serialize {type(obj).__name__!r}; "
        "convert upstream or extend _json_default."
    )