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
from typing import TYPE_CHECKING, Any, Iterator, Optional, Self, Union

import yggdrasil.pickle.json as json_mod
from yggdrasil.io.enums import SaveMode
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["JsonOptions", "JsonIO"]


_VALUE_COLUMN = "value"


@dataclass
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

    def __post_init__(self) -> None:
        """Normalize and validate JSON-specific options."""
        super().__post_init__()

        if not isinstance(self.encoding, str):
            raise TypeError(
                f"encoding must be str, got {type(self.encoding).__name__}"
            )
        if not self.encoding:
            raise ValueError("encoding must not be empty")

        if not isinstance(self.errors, str):
            raise TypeError(
                f"errors must be str, got {type(self.errors).__name__}"
            )
        if not self.errors:
            raise ValueError("errors must not be empty")

    @classmethod
    def resolve(cls, *, options: Self | None = None, **overrides: Any) -> Self:
        """Merge *overrides* into *options* (or a fresh default)."""
        return cls.check_parameters(options=options, **overrides)


@dataclass(slots=True)
class JsonIO(MediaIO[JsonOptions]):
    """JSON I/O with list-expansion on read and optimised list output on write."""

    @classmethod
    def check_options(
        cls,
        options: Optional[JsonOptions],
        *args,
        **kwargs,
    ) -> JsonOptions:
        """Validate and merge caller-supplied options."""
        return JsonOptions.check_parameters(options=options, **kwargs)

    @staticmethod
    def _normalize_loaded_json(data: Any) -> list[dict]:
        """Normalize loaded JSON into row-oriented ``list[dict]`` form."""
        if isinstance(data, list):
            if not data:
                return []
            if isinstance(data[0], dict):
                return data
            return [{_VALUE_COLUMN: value} for value in data]

        if isinstance(data, dict):
            return [data]

        return [{_VALUE_COLUMN: data}]

    def _load_json_records(self, options: JsonOptions) -> list[dict]:
        """Decode buffer JSON and normalize it into row records."""
        if self.buffer.size <= 0:
            return []

        raw = self.buffer.to_bytes()
        data = _json.loads(raw.decode(options.encoding, errors=options.errors))
        return self._normalize_loaded_json(data)

    # ------------------------------------------------------------------
    # Batch-oriented implementation
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self,
        *,
        options: JsonOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the JSON buffer.

        * ``[]`` or empty buffer → no batches.
        * ``[{…}, {…}]`` → one batch, each dict is a row.
        * ``[1, 2, 3]`` → one batch with a ``"value"`` column.
        * ``{…}`` (single object) → one batch with one row.
        """
        import pyarrow as pa

        records = self._load_json_records(options)
        if not records:
            return

        batch = pa.RecordBatch.from_pylist(records)

        if options.columns is not None:
            table = pa.Table.from_batches([batch]).select(options.columns)
            yield from table.to_batches()
            return

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

        payload = json_mod.dumps(
            records,
            encoding=options.encoding,
            errors=options.errors,
        )
        self.buffer.replace_with_payload(payload)

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
        batch_size = resolved.batch_size
        if batch_size and batch_size > 0:
            return super().read_pylist(options=resolved)

        if self.buffer.size <= 0:
            return []

        raw = self.buffer.to_bytes()

        codec = self.codec
        if codec is not None:
            buf, _ = self._decompressed_buffer()
            raw = buf.to_bytes()

        data = _json.loads(raw.decode(resolved.encoding, errors=resolved.errors))
        return self._normalize_loaded_json(data)

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

        batch_size = resolved.batch_size
        if resolved.mode in (SaveMode.APPEND, SaveMode.UPSERT) or (batch_size and batch_size > 0):
            return super().write_pylist(
                data,
                options=resolved,
            )

        payload = json_mod.dumps(
            data,
            encoding=resolved.encoding,
            errors=resolved.errors,
        )

        codec = self.codec
        if codec is not None:
            from .bytes_io import BytesIO as _BIO

            plain = _BIO(payload, config=self.buffer.config)
            self._compress_into_buffer(plain)
        else:
            self.buffer.replace_with_payload(payload)

        return None