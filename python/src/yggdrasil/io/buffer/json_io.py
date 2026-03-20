"""JSON I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Reading:
    * Parses the raw JSON bytes.
    * If the top-level value is a **list**, each element becomes a row:
      - list of dicts → each dict is a row (standard array-of-objects).
      - list of scalars → each scalar becomes a row with a ``"value"`` column.
    * If the top-level value is a **dict** (single object), it is wrapped
      in ``[obj]`` so the table has exactly one row.
    * An empty buffer yields no batches.

Writing:
    * Converts all record batches into a flat ``list[dict]`` and serialises
      as a compact JSON **array** (``[{…}, {…}, …]``), optimised for size.

Transport-level compression is handled transparently by the base class.
"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional, Self, Union

import yggdrasil.pickle.json as json_mod
from yggdrasil.io.enums import SaveMode
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["JsonOptions", "JsonIO"]


_VALUE_COLUMN = "value"


@dataclass(slots=True)
class JsonOptions(MediaOptions):
    """Options for JSON I/O.

    Parameters
    ----------
    encoding:
        Character encoding used when decoding / writing.
    errors:
        Error-handling mode for encoding (``"strict"``, ``"replace"``, …).
    """


    encoding: str = "utf-8"
    errors: str = "strict"

    @classmethod
    def resolve(cls, *, options: Self | None = None, **overrides) -> Self:
        """Merge *overrides* into *options* (or a fresh default)."""
        base = options or cls()
        valid = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        unknown = set(overrides) - set(valid)
        if unknown:
            raise TypeError(f"{cls.__name__}.resolve(): unknown option(s): {sorted(unknown)}")
        for k, v in overrides.items():
            setattr(base, k, v)
        return base


@dataclass(slots=True)
class JsonIO(MediaIO[JsonOptions]):
    """JSON I/O with list-expansion on read and optimised list output on write."""

    @classmethod
    def check_options(cls, options: Optional[JsonOptions], *args, **kwargs) -> JsonOptions:
        """Validate and merge caller-supplied options."""
        return JsonOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Batch-oriented implementation
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, *, options: JsonOptions) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the JSON buffer.

        * ``[]``  or empty buffer → no batches.
        * ``[{…}, {…}]`` → one batch, each dict is a row.
        * ``[1, 2, 3]`` → one batch with a ``"value"`` column.
        * ``{…}`` (single object) → one batch with one row.
        """
        import pyarrow as pa

        if self.buffer.size <= 0:
            return

        raw = self.buffer.to_bytes()
        data = _json.loads(raw.decode(options.encoding, errors=options.errors))

        if isinstance(data, list):
            if not data:
                return
            # List of dicts → standard row-per-dict
            if isinstance(data[0], dict):
                records = data
            else:
                # List of scalars → wrap each in a {value: x} row
                records = [{_VALUE_COLUMN: v} for v in data]
        elif isinstance(data, dict):
            records = [data]
        else:
            # Scalar top-level value
            records = [{_VALUE_COLUMN: data}]

        batch = pa.RecordBatch.from_pylist(records)

        if options.columns is not None:
            tbl = pa.Table.from_batches([batch]).select(options.columns)
            yield from tbl.to_batches()
        else:
            yield batch

    def _write_arrow_batches(
        self,
        *,
        batches: Iterator["pyarrow.RecordBatch"],
        schema: "pyarrow.Schema",
        options: JsonOptions,
    ) -> None:
        """Write record batches as a JSON array into the buffer.

        Collects all rows and serialises as a single ``[{…}, …]`` list
        for compact, optimised output.
        """
        import pyarrow as pa

        all_batches = list(batches)
        if not all_batches:
            records: list[dict] = []
        else:
            table = pa.Table.from_batches(all_batches, schema=schema)
            records = table.to_pylist()

        payload = json_mod.dumps(records, encoding=options.encoding, errors=options.errors)
        self.buffer._replace_with_payload(payload)

    # ------------------------------------------------------------------
    # Optimised convenience overrides
    # ------------------------------------------------------------------

    def read_pylist(
        self,
        *,
        options: JsonOptions | None = None,
        **option_kwargs,
    ) -> Union[list[dict], Iterator[list[dict]]]:
        """Read JSON directly as a list of dicts (fast path, no Arrow overhead).

        When ``batch_size`` > 0 in options, falls back to the base-class
        Arrow-based batched path.
        """
        resolved = self.check_options(options=options, **option_kwargs)
        bs = resolved.batch_size
        if bs and bs > 0:
            return super().read_pylist(options=resolved)

        if self.buffer.size <= 0:
            return []

        raw = self.buffer.to_bytes()

        codec = self.codec
        if codec is not None:
            buf, _ = self._decompressed_buffer()
            raw = buf.to_bytes()

        data = _json.loads(raw.decode(resolved.encoding, errors=resolved.errors))

        if isinstance(data, list):
            if data and not isinstance(data[0], dict):
                return [{_VALUE_COLUMN: v} for v in data]
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return [{_VALUE_COLUMN: data}]

    def write_pylist(
        self,
        data: list[dict],
        *,
        options: JsonOptions | None = None,
        **option_kwargs,
    ):
        """Write a list of dicts directly as JSON (fast path, no Arrow overhead).

        Save-mode logic and ``batch_size`` fall back to the base class.
        """
        resolved = self.check_options(options=options, **option_kwargs)

        if self.skip_write(mode=resolved.mode):
            return

        bs = resolved.batch_size
        # For APPEND / UPSERT or batched writes we need the full merge, go via Arrow
        if resolved.mode in (SaveMode.APPEND, SaveMode.UPSERT) or (bs and bs > 0):
            return super().write_pylist(
                data,
                options=resolved,
            )

        payload = json_mod.dumps(data, encoding=resolved.encoding, errors=resolved.errors)

        codec = self.codec
        if codec is not None:
            from .bytes_io import BytesIO as _BIO
            plain = _BIO(payload, config=self.buffer.config)
            self._compress_into_buffer(plain)
        else:
            self.buffer._replace_with_payload(payload)
