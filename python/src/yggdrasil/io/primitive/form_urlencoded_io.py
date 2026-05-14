"""``application/x-www-form-urlencoded`` Tabular leaf.

The wire format is a single string of ``key=value`` pairs joined by
``&`` (the body of an HTML form submission). One submission is one
row. To round-trip a multi-row table through the same buffer we
extend the spec the same way :class:`NDJsonIO` extends JSON:
**newline-separated form bodies** — each line is a complete
``urlencode``-d submission. A reader handed a single-line payload
yields one row; a reader handed a multi-line payload yields one row
per line.

Values are strings on the wire. The reader emits each form field
as a ``pa.string`` column; when a key repeats inside one submission
the behavior is controlled by ``multi_values``:

- ``"last"`` (default, browser convention): later occurrences win.
- ``"first"``: first occurrence wins.
- ``"list"``: the column becomes a ``list<string>`` for that key.

The writer accepts arbitrary scalar / list columns and casts each
cell to its string form via ``str()``; nulls map to empty strings
when ``keep_blank_values=True`` (the default) and are dropped from
the encoded body otherwise. List-typed columns emit one
``key=value`` pair per element, preserving the standard form-encoding
convention for repeated keys.
"""

from __future__ import annotations

import dataclasses
import itertools as _it
from typing import Any, ClassVar, Iterable, Iterator, Literal
from urllib.parse import parse_qsl, urlencode

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.base import IO


__all__ = ["FormUrlencodedIO", "FormUrlencodedOptions"]


#: Modes that may need to read existing bytes and merge with the
#: incoming stream. Mirrors :mod:`ndjson_io` — APPEND keeps the
#: byte-level fast path on uncompressed buffers without key dedup;
#: UPSERT / MERGE always trigger the read-modify-rewrite.
_MERGE_MODES = frozenset({Mode.APPEND, Mode.UPSERT, Mode.MERGE})


MultiValuePolicy = Literal["last", "first", "list"]


@dataclasses.dataclass(frozen=True, slots=True)
class FormUrlencodedOptions(CastOptions):
    """:class:`CastOptions` extended with form-urlencoded knobs."""

    encoding: str = "utf-8"
    line_ending: str = "\n"
    #: Treat ``key=`` (and ``key`` alone) as ``key -> ""`` rather than
    #: dropping the field. ``True`` matches :func:`urllib.parse.parse_qsl`'s
    #: ``keep_blank_values=True`` and the HTML form convention.
    keep_blank_values: bool = True
    #: ``&`` is the spec separator; some encoders use ``;``. Reader
    #: accepts either when this is the default. Writer always emits
    #: this character.
    separator: str = "&"
    #: How to fold repeated keys *within one submission*:
    #: ``"last"`` keeps the last value (browser default), ``"first"``
    #: keeps the first, ``"list"`` promotes the column to
    #: ``list<string>``.
    multi_values: MultiValuePolicy = "last"
    #: When ``False`` (default), null cells are skipped on write.
    #: When ``True``, nulls emit ``key=`` (blank value).
    emit_null_as_blank: bool = False


def _row_to_pairs(
    row: dict,
    *,
    emit_null_as_blank: bool,
) -> list[tuple[str, str]]:
    """Flatten one record-batch row into ``(key, value)`` pairs.

    List cells emit one pair per element (the form-encoding convention
    for repeated keys). Scalars emit a single pair. ``None`` is
    skipped unless ``emit_null_as_blank`` is set.
    """
    pairs: list[tuple[str, str]] = []
    for key, value in row.items():
        if value is None:
            if emit_null_as_blank:
                pairs.append((key, ""))
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is None:
                    if emit_null_as_blank:
                        pairs.append((key, ""))
                    continue
                pairs.append((key, str(item)))
            continue
        pairs.append((key, str(value)))
    return pairs


def _fold_pairs(
    pairs: Iterable[tuple[str, str]],
    policy: MultiValuePolicy,
) -> dict[str, Any]:
    """Collapse ordered pairs into a row dict using *policy*.

    First-seen order is preserved across keys. ``"last"`` overwrites
    the slot, ``"first"`` keeps the original, ``"list"`` accumulates
    into a list (single-occurrence keys stay scalar — promotion only
    fires on the second hit).
    """
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key not in out:
            out[key] = value
            continue
        if policy == "first":
            continue
        if policy == "last":
            out[key] = value
            continue
        # policy == "list"
        existing = out[key]
        if isinstance(existing, list):
            existing.append(value)
        else:
            out[key] = [existing, value]
    return out


class FormUrlencodedIO(IO[bytes, FormUrlencodedOptions]):
    """:class:`Tabular` leaf for ``application/x-www-form-urlencoded``."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.FORM_URLENCODED

    @classmethod
    def options_class(cls):
        return FormUrlencodedOptions

    # ==================================================================
    # Schema
    # ==================================================================

    def _collect_schema(self, options: FormUrlencodedOptions) -> Schema:
        if self.size_known and self.size == 0:
            return Schema.empty()
        first = next(iter(self._read_arrow_batches(options)), None)
        if first is None:
            return Schema.empty()
        return Schema.from_arrow(first.schema)

    # ==================================================================
    # Read path
    # ==================================================================

    def _iter_rows(
        self,
        options: FormUrlencodedOptions,
    ) -> Iterator[dict[str, Any]]:
        """Decode the payload into row dicts.

        One row per non-empty line. Each line is parsed with
        :func:`urllib.parse.parse_qsl`, then folded via
        :func:`_fold_pairs` according to ``options.multi_values``.
        """
        with self.arrow_input_stream() as v:
            size = v.size()
            if size == 0:
                return
            v.seek(0)
            data = v.read()

        text = data.decode(options.encoding)
        for line in text.split(options.line_ending):
            stripped = line.strip()
            if not stripped:
                continue
            pairs = parse_qsl(
                stripped,
                keep_blank_values=options.keep_blank_values,
                encoding=options.encoding,
                separator=options.separator,
            )
            yield _fold_pairs(pairs, options.multi_values)

    def _read_arrow_batches(
        self,
        options: FormUrlencodedOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield one :class:`pa.RecordBatch` per submission line.

        Each batch is funneled through :meth:`CastOptions.cast_arrow_tabular`
        so a bound ``target_field`` reshapes the row to the caller's
        schema before it leaves the reader. Without a target the
        cast is a passthrough.
        """
        if self.size_known and self.size == 0:
            return
        for row in self._iter_rows(options):
            if not row:
                continue
            batch = pa.RecordBatch.from_pylist([row])
            yield options.cast_arrow_tabular(batch)

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: FormUrlencodedOptions,
    ) -> None:
        """Serialize each row as one ``urlencode``-d line.

        Modes mirror :class:`NDJsonIO`: OVERWRITE truncates; APPEND
        seeks to EOF and writes (concatenation is valid on
        uncompressed buffers without ``match_by``); UPSERT / MERGE /
        keyed APPEND read the existing payload and merge via
        :func:`yggdrasil.arrow.ops.upsert_arrow_batches` before a
        single rewrite. IGNORE / ERROR_IF_EXISTS guard non-empty
        buffers.
        """
        mode = options.mode
        if mode is Mode.AUTO:
            action = Mode.UPSERT if options.match_by_keys else Mode.APPEND
        elif mode is Mode.TRUNCATE:
            action = Mode.OVERWRITE
        elif mode in _MERGE_MODES or mode in (
            Mode.IGNORE, Mode.ERROR_IF_EXISTS, Mode.OVERWRITE,
        ):
            action = mode
        else:
            action = Mode.OVERWRITE

        if action is Mode.IGNORE:
            if self.size > 0:
                return
            action = Mode.OVERWRITE
        elif action is Mode.ERROR_IF_EXISTS:
            if self.size > 0:
                raise FileExistsError(
                    f"{type(self).__name__} buffer is non-empty "
                    f"({self.size} bytes); refusing to overwrite under "
                    f"mode={options.mode!r}."
                )
            action = Mode.OVERWRITE

        codec = self._codec()
        match_by = list(options.match_by_keys or ())
        is_append_uncompressed = (
            action is Mode.APPEND
            and self.size > 0
            and codec is None
            and not match_by
        )
        needs_rewrite = (
            action in _MERGE_MODES
            and self.size > 0
            and not is_append_uncompressed
        )

        if needs_rewrite:
            from yggdrasil.arrow.ops import upsert_arrow_batches

            existing = list(self._read_arrow_batches(options))
            merged = upsert_arrow_batches(
                iter(existing),
                iter(batches),
                options.match_by_keys,
                Mode.APPEND if action is Mode.APPEND else Mode.UPSERT,
                memory_pool=options.arrow_memory_pool,
            )
            return self._write_arrow_batches(
                merged, dataclasses.replace(options, mode=Mode.OVERWRITE),
            )

        line_term = options.line_ending.encode(options.encoding)
        needs_newline_prefix = (
            is_append_uncompressed
            and self.size > 0
            and self.pread(1, self.size - 1) != b"\n"
        )

        iterator = iter(batches)
        first = next(iterator, None)
        cast_opts = (
            options.check_source(first.schema) if first is not None else options
        )

        with self.arrow_output_stream(append=is_append_uncompressed) as sink:
            if needs_newline_prefix:
                sink.write(line_term)
            if first is None:
                return
            for batch in _it.chain([first], iterator):
                # Cast to a bound target schema (passthrough when
                # none) before materializing rows. Materialization
                # via ``to_pylist`` is the documented endpoint for
                # row-oriented text encoders — there's no vectorised
                # primitive that produces ``urlencode``-d strings.
                casted = cast_opts.cast_arrow_tabular(batch)
                for row in casted.to_pylist():
                    pairs = _row_to_pairs(
                        row, emit_null_as_blank=options.emit_null_as_blank,
                    )
                    if not pairs:
                        continue
                    encoded = urlencode(pairs, encoding=options.encoding)
                    sink.write(encoded.encode(options.encoding))
                    sink.write(line_term)


