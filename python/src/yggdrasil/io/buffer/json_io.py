"""JSON array-of-objects I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Reads and writes a single JSON document — an array of objects (rows).
Transport-level compression is handled transparently by the base class.

Reading flow:

1. Parse the raw JSON bytes via :func:`json.load`.
2. Wrap non-list payloads in a single-element list (tolerant).
3. Convert the list of dicts to a :class:`pyarrow.Table`.

Writing flow:

1. Convert the Arrow table to ``list[dict]`` via :meth:`Table.to_pylist`.
2. Serialise with :func:`json.dump` into the buffer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional, Self, Union

import yggdrasil.pickle.json as json_mod
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["JsonOptions", "JsonIO"]


@dataclass(slots=True)
class JsonOptions(MediaOptions):
    """Options for JSON I/O.

    Parameters
    ----------
    columns:
        Column names to read (``None`` reads all).
    use_threads:
        Enable multi-threaded Arrow conversion.
    allow_newlines_in_values:
        Tolerate literal newlines inside JSON string values.
    encoding:
        Character encoding used when writing.
    errors:
        Error-handling mode for encoding (``"strict"``, ``"replace"``, …).
    """

    columns: list[str] | None = None
    use_threads: bool = True
    allow_newlines_in_values: bool = False

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
    """JSON array-of-objects I/O.

    The :meth:`read_pylist` / :meth:`write_pylist` overrides use native
    JSON serialisation, which is faster than the base-class Arrow
    round-trip for simple payloads.  The ``batch_size`` parameter is
    fully supported — when positive, :meth:`read_pylist` returns an
    ``Iterator[list[dict]]``.
    """

    @classmethod
    def check_options(cls, options: Optional[JsonOptions], *args, **kwargs) -> JsonOptions:
        """Validate and merge caller-supplied options."""
        return JsonOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Internal JSON helpers (no codec, no batching)
    # ------------------------------------------------------------------

    def _read_json_records(self) -> list[dict]:
        """Parse the buffer's raw bytes as a JSON list of dicts.

        Returns
        -------
        list[dict]
            An empty list when the buffer is empty.  A single object is
            wrapped in a one-element list.
        """
        if self.buffer.size <= 0:
            return []

        with self.buffer.view() as f:
            parsed = json_mod.load(f)

        if not isinstance(parsed, list):
            parsed = [parsed]

        return parsed

    def _write_json_records(self, data: list[dict]) -> None:
        """Write *data* as a JSON array into the buffer (no codec handling)."""
        with self.buffer.view() as f:
            json_mod.dump(data, f)

    # ------------------------------------------------------------------
    # Public overrides matching base-class signatures
    # ------------------------------------------------------------------

    def read_pylist(
        self,
        *,
        batch_size: Optional[int] = None,
    ) -> Union[list[dict], Iterator[list[dict]]]:
        """Read the buffer as a list of row dicts (native JSON path).

        When the media type carries a codec the base-class Arrow path is
        used so that transparent decompression is honoured.

        Parameters
        ----------
        batch_size:
            When ``None`` or ≤ 0, return a single ``list[dict]``.
            When a positive integer, return an ``Iterator[list[dict]]``
            yielding chunks of at most *batch_size* rows.

        Returns
        -------
        list[dict] | Iterator[list[dict]]
        """
        # If codec is set, delegate to base class (decompresses → Arrow → pylist)
        if self.codec is not None:
            return super().read_pylist(batch_size=batch_size)

        records = self._read_json_records()

        if batch_size is not None and batch_size > 0:
            def _chunks():
                for i in range(0, len(records), batch_size):
                    yield records[i : i + batch_size]
            return _chunks()

        return records

    def write_pylist(
        self,
        data: list[dict],
        *,
        batch_size: Optional[int] = None,
        mode=None,
        match_by=None,
        options=None,
        **option_kwargs,
    ) -> None:
        """Write a list of row dicts as JSON (native path).

        When the media type carries a codec the base-class Arrow path is
        used so that transparent compression is honoured.

        Parameters
        ----------
        data:
            List of ``{column: value}`` dicts.
        batch_size:
            Forwarded to base-class :meth:`write_arrow_table` when the
            codec path is used.  Ignored on the native JSON path (the
            entire list is always written as one document).
        mode, match_by, options, **option_kwargs:
            Forwarded to :meth:`write_arrow_table` on the codec path.
        """
        if self.codec is not None:
            return super().write_pylist(
                data, batch_size=batch_size, mode=mode,
                match_by=match_by, options=options, **option_kwargs,
            )
        self._write_json_records(data)

    # ------------------------------------------------------------------
    # Arrow implementation
    # ------------------------------------------------------------------

    def _read_arrow_table(self, *, options: JsonOptions) -> "pyarrow.Table":
        """Read JSON bytes into an Arrow table (uncompressed buffer)."""
        from yggdrasil.arrow.lib import pyarrow as _pa

        records = self._read_json_records()
        if not records:
            return _pa.table({})

        return _pa.Table.from_pylist(records)

    def _write_arrow_table(self, *, table: "pyarrow.Table", options: JsonOptions) -> None:
        """Write an Arrow table as JSON into the (uncompressed) buffer."""
        self._write_json_records(table.to_pylist())
