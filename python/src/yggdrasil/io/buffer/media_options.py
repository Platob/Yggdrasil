"""Base options dataclass for :class:`~yggdrasil.io.buffer.media_io.MediaIO`.

:class:`MediaOptions` collects every parameter that can influence how an
Arrow table is read from / written to a :class:`~yggdrasil.io.buffer.BytesIO`
buffer.  Format-specific subclasses (:class:`ParquetOptions`,
:class:`JsonOptions`, …) extend it with codec-level knobs, but the
fields defined here are shared across *all* formats:

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Field
     - Default
     - Description
   * - ``columns``
     - ``None``
     - Column names to read (``None`` = all).
   * - ``use_threads``
     - ``True``
     - Enable multi-threaded reads / writes.
   * - ``ignore_empty``
     - ``True``
     - Silently return an empty table when the buffer is empty.
   * - ``lazy``
     - ``False``
     - Return a lazy / streaming representation when the format supports it.
   * - ``raise_error``
     - ``True``
     - Raise on read/write errors instead of returning a sentinel.
   * - ``batch_size``
     - ``0``
     - When > 0, read/write in chunks of *batch_size* rows.
   * - ``mode``
     - ``AUTO``
     - :class:`~yggdrasil.io.enums.SaveMode` governing the write strategy.
   * - ``match_by``
     - ``None``
     - Column names forming the composite key for ``UPSERT`` mode.

Save modes
----------
``AUTO`` / ``OVERWRITE``
    Replace the buffer contents entirely.

``APPEND``
    Concatenate new data after the existing data.

``UPSERT``
    Merge by ``match_by`` columns — new rows replace matching old rows,
    unmatched old rows survive, unmatched new rows are appended.

``IGNORE``
    Skip the write when the buffer already contains data.

``ERROR_IF_EXISTS``
    Raise :exc:`IOError` when the buffer already contains data.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Sequence, Any

from ..enums.save_mode import SaveMode

__all__ = [
    "MediaOptions",
]

_MISSING = object()

_ALLOWED_COMPRESSION = {"auto", "none", "off", "zstd", "snappy", "gzip", "lz4"}


@dataclass(slots=True)
class MediaOptions:
    """Base options shared by all :class:`MediaIO` subclasses.

    Parameters
    ----------
    columns:
        Column names to read.  ``None`` means all columns.
    use_threads:
        Enable multi-threaded reading / writing.
    ignore_empty:
        When ``True``, an empty buffer returns an empty table instead of
        raising.
    lazy:
        When ``True``, return a lazy / streaming representation if the
        format supports it (e.g. a Polars ``LazyFrame``).
    raise_error:
        When ``True``, raise on read/write errors.
    batch_size:
        When > 0, read or write in chunks of *batch_size* rows.
        Reads return an ``Iterator`` of chunks; writes slice the table
        and write each chunk sequentially.
    mode:
        :class:`~yggdrasil.io.enums.SaveMode` governing the write
        strategy.  See the module docstring for semantics.
    match_by:
        Column names forming the composite deduplication key for
        ``UPSERT`` mode.  Ignored by other modes.
    """

    # global read properties
    columns: Optional[Sequence[str]] = None
    use_threads: bool = True
    ignore_empty: bool = True
    lazy: bool = False
    raise_error: bool = True

    # global read / write properties
    batch_size: int = 0

    # global write properties
    mode: SaveMode = SaveMode.AUTO
    match_by: Sequence[str] | None = None

    @classmethod
    def check_parameters(
        cls,
        options: Optional["MediaOptions"] = None,
        *,
        mode: SaveMode | str | None | Any = _MISSING,
        match_by: Sequence[str] | str | None | Any = _MISSING,
        columns: Optional[Sequence[str]] = _MISSING,
        use_threads: bool = _MISSING,
        ignore_empty: bool = _MISSING,
        lazy: bool = _MISSING,
        raise_error: bool = _MISSING,
        batch_size: int | None = _MISSING,
        **kwargs,
    ) -> "MediaOptions":
        """Merge and validate parameters into an options instance.

        * Only explicitly supplied kwargs override existing values.
        * Unknown kwargs raise :exc:`TypeError`.
        * ``mode`` is normalised via :meth:`SaveMode.parse`.
        * ``match_by`` is normalised to ``tuple[str, …] | None``.
        * ``columns`` is normalised to ``list[str] | None``.

        Parameters
        ----------
        options:
            An existing instance to update, or ``None`` for a fresh default.
        mode, match_by, columns, use_threads, ignore_empty, lazy, raise_error:
            Overrides for the corresponding fields.
        **kwargs:
            Subclass-specific overrides (e.g. ``compression``).

        Returns
        -------
        MediaOptions
            The validated, merged instance (mutated in-place when *options*
            is an instance of *cls*; a new object otherwise).

        Raises
        ------
        TypeError
            On unknown kwargs, bad types for ``columns``, ``match_by``,
            boolean fields, or ``compression``.
        ValueError
            On invalid ``compression`` or ``skip_rows`` values.
        """
        if options is None:
            out = cls()
        else:
            if not isinstance(options, MediaOptions):
                raise TypeError(f"options must be a MediaOptions instance or None, got {type(options)!r}")

            if isinstance(options, cls):
                out = options
            else:
                # upcast into cls
                out = cls()
                for f in fields(cls):
                    if hasattr(options, f.name):
                        setattr(out, f.name, getattr(options, f.name))

        allowed = {f.name for f in fields(cls)}

        updates: dict[str, Any] = {}

        if mode is not _MISSING:
            updates["mode"] = mode
        if match_by is not _MISSING:
            updates["match_by"] = match_by
        if columns is not _MISSING:
            updates["columns"] = columns
        if use_threads is not _MISSING:
            updates["use_threads"] = use_threads
        if ignore_empty is not _MISSING:
            updates["ignore_empty"] = ignore_empty
        if lazy is not _MISSING:
            updates["lazy"] = lazy
        if raise_error is not _MISSING:
            updates["raise_error"] = raise_error
        if batch_size is not _MISSING:
            updates["batch_size"] = batch_size

        updates.update(kwargs)

        unknown = [k for k in updates.keys() if k not in allowed]
        if unknown:
            raise TypeError(
                f"{cls.__name__}.check_parameters got unexpected parameter(s): {', '.join(sorted(unknown))}"
            )

        # ---- validations / normalizations ----

        if "columns" in updates:
            cols = updates["columns"]
            if cols is None:
                pass
            elif isinstance(cols, (str, bytes)):
                raise TypeError("columns must be a sequence of strings, not a single string/bytes")
            else:
                try:
                    cols_list = list(cols)
                except TypeError as e:
                    raise TypeError("columns must be a sequence of strings") from e
                if not all(isinstance(c, str) for c in cols_list):
                    bad = [type(c).__name__ for c in cols_list if not isinstance(c, str)]
                    raise TypeError(f"columns must contain only str, found: {bad[:3]}")
                updates["columns"] = list(cols_list)

        for b in ("use_threads", "ignore_empty", "lazy", "raise_error"):
            if b in updates and not isinstance(updates[b], bool):
                raise TypeError(f"{b} must be bool, got {type(updates[b]).__name__}")

        if "batch_size" in updates:
            bs = updates["batch_size"]
            if bs is None:
                updates["batch_size"] = 0
            elif not isinstance(bs, int):
                raise TypeError(f"batch_size must be int or None, got {type(bs).__name__}")
            elif bs < 0:
                updates["batch_size"] = 0

        if "mode" in updates:
            updates["mode"] = SaveMode.parse(updates["mode"], default=SaveMode.AUTO)

        if "match_by" in updates:
            mb = updates["match_by"]
            if mb is None:
                pass
            elif isinstance(mb, (str, bytes)):
                if isinstance(mb, bytes):
                    raise TypeError("match_by must be str or a sequence of str, not bytes")
                updates["match_by"] = (mb,)
            else:
                try:
                    mb_list = list(mb)
                except TypeError as e:
                    raise TypeError("match_by must be a string or a sequence of strings") from e
                if not mb_list:
                    updates["match_by"] = None
                else:
                    if not all(isinstance(c, str) for c in mb_list):
                        bad = [type(c).__name__ for c in mb_list if not isinstance(c, str)]
                        raise TypeError(f"match_by must contain only str, found: {bad[:3]}")
                    updates["match_by"] = tuple(mb_list)

        # common subclass validations
        if "compression" in updates:
            comp = updates["compression"]
            if comp is not None:
                if not isinstance(comp, str):
                    raise TypeError(f"compression must be str|None, got {type(comp).__name__}")
                if comp not in _ALLOWED_COMPRESSION:
                    raise ValueError(f"compression must be one of {sorted(_ALLOWED_COMPRESSION)}, got {comp!r}")

        if "compression_level" in updates:
            lvl = updates["compression_level"]
            if lvl is not None and not isinstance(lvl, int):
                raise TypeError(f"compression_level must be int|None, got {type(lvl).__name__}")

        if "skip_rows" in updates:
            sr = updates["skip_rows"]
            if not isinstance(sr, int) or sr < 0:
                raise ValueError("skip_rows must be a non-negative int")

        if "zip_compression" in updates:
            zc = updates["zip_compression"]
            if not isinstance(zc, int):
                raise TypeError(f"zip_compression must be int, got {type(zc).__name__}")

        # apply
        for k, v in updates.items():
            setattr(out, k, v)

        return out
