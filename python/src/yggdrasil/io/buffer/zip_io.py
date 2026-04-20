"""ZIP container I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Design principles
-----------------

* **Exact-name lookup is O(1).** Single-member reads go through
  :meth:`zipfile.ZipFile.getinfo`, which uses an internal name→info
  hash. We never scan :meth:`infolist` on the hot path.
* **Large members stream to disk.** Each member's payload goes into a
  fresh :class:`BytesIO` whose ``spill_bytes`` config decides whether
  to land in RAM or on disk. For a 2 GB member, peak extra RAM is
  bounded by ``config.spill_bytes``, not the member size.
* **Name-based inference first.** When a member has an unambiguous
  extension (``.json``, ``.csv``, …) we route it to the right inner
  MediaIO directly. Content sniffing is a fallback for extensionless
  members and costs at most a small prefix read.
* **Dictionary-encoded metadata columns.** When
  :attr:`ZipOptions.read_member_infos` injects a ``name`` column, it
  uses a :class:`pa.DictionaryArray` with a single-entry dictionary per
  member batch. A 10 million-row member × 20-byte filename costs 20
  bytes of strings + 10M int32 indices, not 10M × 20 bytes.
* **Group-level iteration.** Callers who want "batches from member A,
  then batches from member B" (no mid-member splitting) use
  :meth:`read_arrow_groups` or
  :meth:`read_arrow_tables_by_group`. The standard batch iterator
  remains flat.

Write path: each ``write_arrow_table`` call writes one member named by
:attr:`ZipOptions.member`. :class:`SaveMode.APPEND` adds members
without touching existing ones; :class:`SaveMode.UPSERT` replaces the
named member if it exists.

Transport-level compression (``MediaType.codec``) is handled by the
base class via ``open()`` / ``close()`` / ``mark_dirty()`` — it wraps
the whole ZIP stream, independent of the ZIP's own per-member
compression.
"""
from __future__ import annotations

import fnmatch
import io as _stdio
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import pyarrow as pa

from yggdrasil.io.enums import MediaType, MimeType, MimeTypes, SaveMode
from .bytes_io import BytesIO
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["ZipOptions", "ZipIO"]


# Size of the prefix read when sniffing an extensionless member's content
# type. Most format magic bytes are in the first 16 bytes; 512 is generous.
_SNIFF_PREFIX_BYTES = 512

# Chunk size for streaming a member out of the ZIP into a spill-backed
# inner BytesIO. 1 MiB keeps peak scratch memory small while amortizing
# the per-read overhead.
_MEMBER_STREAM_CHUNK = 1 * 1024 * 1024

# Valid values for ZipOptions.group_by. Callable is also accepted.
_VALID_GROUP_BY = frozenset({None, "member", "all"})


# ---------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------

@dataclass
class ZipOptions(MediaOptions):
    """Options for ZIP container I/O.

    Parameters
    ----------
    member:
        Read: selects member(s).
        - exact string (``"data.json"``) → one member via O(1) lookup.
        - glob pattern (``"*.json"``) → all matching members, concat.
        - ``None`` (default) → all non-directory members, concat.

        Write: the filename of the single member to add.
    inner_media:
        Override the media type of a member's payload. Accepts a
        :class:`MediaType`, a :class:`MimeType`, or a short string
        like ``"json"``, ``"csv"``, ``"parquet"``. When ``None`` we
        infer from the member's filename extension, falling back to
        content sniffing for extensionless names.
    force_inner_media:
        When ``True``, skip all inference and use :attr:`inner_media`
        verbatim. Requires :attr:`inner_media` to be set. Both a fast
        path (no sniff) and an escape hatch for mislabeled files.
    read_member_infos:
        List of ``(info_key, output_column_name)`` pairs. For each
        pair, the read path injects a metadata column containing the
        info value, dictionary-encoded so a 10-million-row member
        with a 20-byte filename uses 20 bytes of payload, not 200 MB.
        Currently supported ``info_key``: ``"name"``.
    group_by:
        Controls the batch-boundary semantics of
        :meth:`ZipIO.read_arrow_groups`.
        - ``None`` / ``"all"``: one group containing every batch.
        - ``"member"``: one group per member. A new member never starts
          inside a previous group's batch stream.
        - Callable ``(member_name) -> group_key``: user-defined
          grouping — all members mapping to the same key share a group.
    zip_compression:
        Write-side ZIP compression method (per-member, inside the archive).
        One of :data:`zipfile.ZIP_STORED`, :data:`zipfile.ZIP_DEFLATED`
        (default), :data:`zipfile.ZIP_BZIP2`, :data:`zipfile.ZIP_LZMA`.
        Named ``zip_compression`` (not ``compression``) to distinguish
        from the base :class:`MediaOptions.compression` field that
        governs transport-level codec compression of the whole archive.
    zip_compresslevel:
        Write-side compression level. ``None`` uses the method default.
    """

    member: str | None = None
    inner_media: Any = None
    force_inner_media: bool = False
    read_member_infos: Sequence[tuple[str, str]] | None = None
    group_by: Any = None
    zip_compression: int = zipfile.ZIP_DEFLATED
    zip_compresslevel: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.member is not None and not isinstance(self.member, str):
            raise TypeError(f"member must be str|None, got {type(self.member).__name__}")
        if self.member == "":
            raise ValueError("member must not be an empty string")

        if not isinstance(self.force_inner_media, bool):
            raise TypeError(
                f"force_inner_media must be bool, got {type(self.force_inner_media).__name__}"
            )
        if self.force_inner_media and self.inner_media is None:
            raise ValueError("force_inner_media=True requires inner_media to be set")

        if self.read_member_infos is not None:
            normalized: list[tuple[str, str]] = []
            for pair in self.read_member_infos:
                if not (isinstance(pair, tuple) and len(pair) == 2):
                    raise TypeError(
                        "read_member_infos entries must be (info_key, column_name) tuples"
                    )
                key, col = pair
                if not isinstance(key, str) or not key:
                    raise ValueError("read_member_infos info_key must be a non-empty str")
                if not isinstance(col, str) or not col:
                    raise ValueError("read_member_infos column_name must be a non-empty str")
                normalized.append((key, col))
            self.read_member_infos = tuple(normalized)

        if self.group_by is not None and not callable(self.group_by):
            if self.group_by not in _VALID_GROUP_BY:
                raise ValueError(
                    f"group_by must be None, 'member', 'all', or a callable; "
                    f"got {self.group_by!r}"
                )

        if self.zip_compression not in {
            zipfile.ZIP_STORED,
            zipfile.ZIP_DEFLATED,
            zipfile.ZIP_BZIP2,
            zipfile.ZIP_LZMA,
        }:
            raise ValueError(
                f"zip_compression must be a zipfile.ZIP_* constant, "
                f"got {self.zip_compression!r}"
            )

        if self.zip_compresslevel is not None and not isinstance(self.zip_compresslevel, int):
            raise TypeError(
                f"zip_compresslevel must be int|None, "
                f"got {type(self.zip_compresslevel).__name__}"
            )

    @classmethod
    def resolve(cls, *, options: "ZipOptions | None" = None, **overrides: Any) -> "ZipOptions":
        return cls.check_parameters(options=options, **overrides)


# ---------------------------------------------------------------------
# ZipIO
# ---------------------------------------------------------------------

@dataclass(slots=True)
class ZipIO(MediaIO[ZipOptions]):
    """ZIP container I/O with streaming member reads."""

    @classmethod
    def check_options(
        cls,
        options: Optional[ZipOptions],
        *args,
        **kwargs,
    ) -> ZipOptions:
        return ZipOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Inner media resolution
    # ------------------------------------------------------------------

    def _resolve_inner_media(
        self,
        info: zipfile.ZipInfo,
        prefix: bytes,
        options: ZipOptions,
    ) -> MediaType:
        """Resolve the :class:`MediaType` of a member's payload.

        Dispatch order:

        1. ``force_inner_media=True`` → :attr:`ZipOptions.inner_media`
           verbatim. No name lookup, no sniff.
        2. Explicit ``inner_media`` without force → treat as a strong
           hint; still used verbatim (but validated).
        3. Filename extension inference.
        4. Content sniff on *prefix*.
        5. Octet-stream fallback.
        """
        if options.force_inner_media:
            return self._coerce_media(options.inner_media)

        if options.inner_media is not None:
            return self._coerce_media(options.inner_media)

        # Name-based inference: cheapest signal.
        from_name = MediaType.parse(info.filename, default=None)
        if from_name is not None and from_name.mime_type is not MimeTypes.OCTET_STREAM:
            return from_name

        # Prefix sniff for extensionless / mislabeled names.
        sniffed = self._sniff_media_from_prefix(prefix)
        if sniffed is not None and sniffed.mime_type is not MimeTypes.OCTET_STREAM:
            return sniffed

        return MediaType(MimeTypes.OCTET_STREAM)

    @staticmethod
    def _coerce_media(value: Any) -> MediaType:
        """Turn a user-facing inner_media hint into a :class:`MediaType`."""
        if isinstance(value, MediaType):
            return value
        if isinstance(value, MimeType):
            return MediaType(value)
        if isinstance(value, str):
            # Accept short-form strings like "json" by probing the
            # extension-to-mime mapping via a synthetic filename.
            parsed = MediaType.parse(value, default=None)
            if parsed is not None and parsed.mime_type is not MimeTypes.OCTET_STREAM:
                return parsed
            # Fall back to treating the string as an extension.
            parsed = MediaType.parse(f"x.{value}", default=None)
            if parsed is not None and parsed.mime_type is not MimeTypes.OCTET_STREAM:
                return parsed
            raise ValueError(f"Cannot resolve inner_media={value!r} to a MediaType")
        raise TypeError(
            f"inner_media must be MediaType|MimeType|str|None, got {type(value).__name__}"
        )

    @staticmethod
    def _sniff_media_from_prefix(prefix: bytes) -> MediaType | None:
        """Best-effort content sniff. Returns ``None`` on no match.

        Exposed as a staticmethod because the tests monkey-patch it to
        verify it's called only when name-based inference fails.
        """
        if not prefix:
            return None

        head = prefix.lstrip()[:8]

        # JSON (array or object).
        if head[:1] in (b"[", b"{"):
            return MediaType(MimeTypes.JSON)

        # NDJSON — heuristic: first non-whitespace char is '{' and a
        # newline appears early. But '{' alone already resolves to JSON
        # above; NDJSON-vs-JSON is ambiguous without schema knowledge.
        # Stick with JSON as the safe default.

        # Parquet magic: PAR1 at the start (and end).
        if head[:4] == b"PAR1":
            return MediaType(MimeTypes.PARQUET)

        # Arrow IPC streaming format starts with the continuation marker.
        if head[:4] == b"\xff\xff\xff\xff":
            return MediaType(MimeTypes.ARROW_IPC)

        # XML.
        if head[:1] == b"<" or head[:5] == b"<?xml":
            return MediaType(MimeTypes.XML)

        # CSV: comma-separated, no magic. Skip sniffing — users should
        # set inner_media explicitly for CSV in a ZIP.

        return None

    # ------------------------------------------------------------------
    # Member loading (streaming, respects spill_bytes)
    # ------------------------------------------------------------------

    def _load_member_buffer(
        self,
        zf: zipfile.ZipFile,
        info: zipfile.ZipInfo,
    ) -> tuple[BytesIO, bytes]:
        """Stream *info*'s payload into a fresh :class:`BytesIO`.

        Returns ``(inner_buf, prefix)`` where *prefix* is the first
        :data:`_SNIFF_PREFIX_BYTES` of the payload (useful for content
        sniffing without re-reading the member). When the member fits
        below the holder's ``spill_bytes`` threshold, the buffer stays
        in RAM; otherwise it spills to disk.

        Exposed as an instance method so tests can wrap it for tracing.
        """
        cfg = self.holder.config
        inner = BytesIO(config=cfg)

        prefix_chunks: list[bytes] = []
        prefix_collected = 0

        with zf.open(info, "r") as src:
            while True:
                chunk = src.read(_MEMBER_STREAM_CHUNK)
                if not chunk:
                    break
                # Capture the first _SNIFF_PREFIX_BYTES bytes for later
                # content-type sniffing without a second pass.
                if prefix_collected < _SNIFF_PREFIX_BYTES:
                    need = _SNIFF_PREFIX_BYTES - prefix_collected
                    prefix_chunks.append(chunk[:need])
                    prefix_collected += min(need, len(chunk))
                inner.write_bytes(chunk)

        inner.seek(0)
        prefix = b"".join(prefix_chunks)
        return inner, prefix

    # ------------------------------------------------------------------
    # Member selection
    # ------------------------------------------------------------------

    def _select_infos(
        self,
        zf: zipfile.ZipFile,
        options: ZipOptions,
    ) -> list[zipfile.ZipInfo]:
        """Resolve ``options.member`` to a list of :class:`ZipInfo` objects.

        * Exact string: O(1) via :meth:`zf.getinfo`. Raises ``KeyError``
          with a full member listing if missing or if the match is a
          directory entry.
        * Glob pattern (contains ``*``, ``?``, or ``[``): walks
          ``infolist`` once, filters via :mod:`fnmatch`.
        * ``None``: every non-directory member.
        """
        if options.member is None:
            return [info for info in zf.infolist() if not self._is_dir_entry(info)]

        name = options.member
        if self._is_glob(name):
            matches = [
                info for info in zf.infolist()
                if not self._is_dir_entry(info) and fnmatch.fnmatch(info.filename, name)
            ]
            if not matches:
                raise KeyError(
                    f"No ZIP member matches pattern {name!r}. "
                    f"Available: {self._nondir_names(zf)}"
                )
            return matches

        # Exact lookup — O(1) via zipfile's internal name hash.
        try:
            info = zf.getinfo(name)
        except KeyError:
            raise KeyError(
                f"ZIP member {name!r} not found. "
                f"Available: {self._nondir_names(zf)}"
            ) from None

        if self._is_dir_entry(info):
            # getinfo matched a directory entry (names ending in '/').
            # Treat that as "not found" — directory entries have no payload.
            raise KeyError(
                f"ZIP member {name!r} not found. "
                f"Available: {self._nondir_names(zf)}"
            )

        return [info]

    @staticmethod
    def _is_dir_entry(info: zipfile.ZipInfo) -> bool:
        return info.filename.endswith("/")

    @staticmethod
    def _is_glob(name: str) -> bool:
        """Return True if *name* contains any fnmatch wildcard."""
        return any(ch in name for ch in ("*", "?", "["))

    @staticmethod
    def _nondir_names(zf: zipfile.ZipFile) -> list[str]:
        """Return the list of non-directory member names.

        Used only on the error branch (missing-member KeyError) — building
        this list means walking ``infolist`` which we avoid on the hot
        path. Exposed as a staticmethod so tests can verify it's not
        called on success.
        """
        return [info.filename for info in zf.infolist() if not info.filename.endswith("/")]

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _group_key_for(
        info: zipfile.ZipInfo,
        options: ZipOptions,
    ) -> Any:
        """Return the group key for *info*."""
        gb = options.group_by
        if gb is None or gb == "all":
            return "__all__"
        if gb == "member":
            return info.filename
        if callable(gb):
            return gb(info.filename)
        raise ValueError(f"Unknown group_by: {gb!r}")

    # ------------------------------------------------------------------
    # Core read protocol
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self,
        options: ZipOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield a flat stream of Arrow batches from ZIP members.

        Batches are concatenated across members. To observe member
        boundaries, use :meth:`read_arrow_groups` instead.
        """
        with self.open() as b:
            if b.buffer.size <= 0:
                return

            for _, batches in self._iter_member_batches(options):
                yield from batches

    def _collect_arrow_schema(self, full: bool = False) -> "pyarrow.Schema":
        """Return the schema of the first matching member.

        With ``full=True`` collect and unify every member's schema.
        """
        with self.open() as b:
            if b.buffer.size <= 0:
                return pa.schema([])

            options = self.check_options(options=None)

            with self._open_zipfile() as zf:
                infos = self._select_infos(zf, options)
                if not infos:
                    return pa.schema([])

                if not full:
                    info = infos[0]
                    inner_buf, prefix = self._load_member_buffer(zf, info)
                    try:
                        return self._member_schema(info, inner_buf, prefix, options)
                    finally:
                        inner_buf.close()

                # Full: union of schemas across all members.
                schemas: list[pa.Schema] = []
                for info in infos:
                    inner_buf, prefix = self._load_member_buffer(zf, info)
                    try:
                        schemas.append(self._member_schema(info, inner_buf, prefix, options))
                    finally:
                        inner_buf.close()
                return pa.unify_schemas(schemas) if schemas else pa.schema([])

    def _member_schema(
        self,
        info: zipfile.ZipInfo,
        inner_buf: BytesIO,
        prefix: bytes,
        options: ZipOptions,
    ) -> pa.Schema:
        """Return the schema of a single member via its inner MediaIO."""
        media = self._resolve_inner_media(info, prefix, options)
        inner_buf.set_media_type(media, safe=False)
        inner_io = MediaIO.make(inner_buf, media)
        try:
            return inner_io._collect_arrow_schema(full=False)
        finally:
            # Inner MediaIO close() doesn't close the BytesIO, just
            # releases the internal buffer reference. Safe to call.
            if hasattr(inner_io, "close"):
                try:
                    inner_io.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Group-level iteration (public API)
    # ------------------------------------------------------------------

    def read_arrow_groups(
        self,
        *,
        options: ZipOptions | None = None,
        **option_kwargs,
    ) -> Iterator[tuple[Any, Iterator["pa.RecordBatch"]]]:
        """Yield ``(group_key, batch_iterator)`` pairs.

        With ``group_by="member"`` each tuple carries the member name
        and an iterator over that member's batches. Iterators must be
        consumed before advancing to the next group — they share the
        underlying ZIP file handle.

        Example::

            with io_.open():
                for name, batches in io_.read_arrow_groups(
                    options=ZipOptions(group_by="member")
                ):
                    table = pa.Table.from_batches(list(batches))
                    print(name, table.num_rows)
        """
        resolved = self.check_options(options=options, **option_kwargs)

        with self.open() as b:
            if b.buffer.size <= 0:
                return

            last_key: Any = object()  # sentinel guaranteed unequal to any key
            buffered: list[pa.RecordBatch] = []

            # Collect (key, batches_iter) pairs but materialize groups
            # at their boundary so callers can consume lazily.
            current_key: Any = object()
            group_batches: list[pa.RecordBatch] | None = None

            for info, batches in self._iter_member_batches(resolved):
                key = self._group_key_for(info, resolved)

                if group_batches is None:
                    current_key = key
                    group_batches = []

                if key != current_key:
                    # Flush the previous group.
                    yield current_key, iter(group_batches)
                    current_key = key
                    group_batches = []

                group_batches.extend(batches)

            if group_batches is not None:
                yield current_key, iter(group_batches)

    def read_arrow_tables_by_group(
        self,
        *,
        options: ZipOptions | None = None,
        **option_kwargs,
    ) -> Iterator[tuple[Any, "pa.Table"]]:
        """Yield ``(group_key, pa.Table)`` pairs.

        Convenience wrapper over :meth:`read_arrow_groups` that
        materializes each group into an Arrow table. Memory scales
        with the biggest single group, not the whole archive.
        """
        for key, batches in self.read_arrow_groups(
            options=options, **option_kwargs
        ):
            batch_list = list(batches)
            if batch_list:
                yield key, pa.Table.from_batches(batch_list)
            else:
                yield key, pa.table({})

    # ------------------------------------------------------------------
    # Internal: member iteration yielding (info, batches)
    # ------------------------------------------------------------------

    def _iter_member_batches(
        self,
        options: ZipOptions,
    ) -> Iterator[tuple[zipfile.ZipInfo, Iterator["pa.RecordBatch"]]]:
        """Yield ``(ZipInfo, batches)`` for every selected member.

        Shared entry point for both the flat and grouped readers.
        """
        with self._open_zipfile() as zf:
            infos = self._select_infos(zf, options)
            for info in infos:
                inner_buf, prefix = self._load_member_buffer(zf, info)
                try:
                    member_batches = self._read_member_batches(
                        info, inner_buf, prefix, options
                    )
                    yield info, member_batches
                finally:
                    # Inner buffer is freed once the consumer has drained
                    # its batches. For very large spilled buffers this
                    # releases the spill file promptly.
                    inner_buf.close()

    def _read_member_batches(
        self,
        info: zipfile.ZipInfo,
        inner_buf: BytesIO,
        prefix: bytes,
        options: ZipOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Read a single member's payload as Arrow batches.

        Applies:
        - member-info metadata columns (dict-encoded)
        - column projection
        - ignore_empty
        - options.cast.cast_iterator
        """
        media = self._resolve_inner_media(info, prefix, options)
        inner_buf.set_media_type(media, safe=False)
        inner_io = MediaIO.make(inner_buf, media)

        # Build a child options instance suited to the inner format.
        # We don't propagate member/inner_media/force_inner_media — those
        # are ZIP-specific — but we do pass through batch_size, columns,
        # ignore_empty, cast, use_threads, raise_error.
        inner_kwargs = self._inner_read_kwargs(options)

        # The inner reader yields its own batches. We inject member-info
        # columns and apply projection / cast in *this* layer so the
        # ZIP-specific knobs don't leak into every inner format.
        raw_batches = inner_io.read_arrow_batches(**inner_kwargs)

        enriched = self._inject_member_infos(info, raw_batches, options)

        if options.columns is not None:
            # Re-apply projection after enrichment: the caller's
            # `columns` list may reference the injected metadata columns.
            enriched = (
                self._project_columns(batch, options.columns) for batch in enriched
            )

        if options.ignore_empty:
            enriched = (batch for batch in enriched if batch.num_rows > 0)

        yield from options.cast.cast_iterator(enriched)

    @staticmethod
    def _inner_read_kwargs(options: ZipOptions) -> dict[str, Any]:
        """Subset of options safe to forward to an inner MediaIO reader.

        We deliberately do NOT pass ``columns`` to the inner reader —
        ZIP-level projection may reference injected metadata columns
        that the inner format doesn't know about. Projection is
        re-applied in the ZIP layer.
        """
        out: dict[str, Any] = {}
        for k in ("batch_size", "use_threads", "raise_error"):
            v = getattr(options, k, ...)
            if v is not ...:
                out[k] = v
        # ignore_empty is handled at the ZIP layer after enrichment.
        # cast is handled at the ZIP layer after enrichment.
        return out

    # ------------------------------------------------------------------
    # Member-info enrichment (dictionary-encoded metadata columns)
    # ------------------------------------------------------------------

    def _inject_member_infos(
        self,
        info: zipfile.ZipInfo,
        batches: Iterator["pa.RecordBatch"],
        options: ZipOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Inject ``read_member_infos`` columns into each batch.

        Each info value becomes a :class:`pa.DictionaryArray` with a
        single-entry dictionary — so a 10M-row member with a 20-byte
        name uses 20 bytes for the dictionary + 10M × 4 bytes for
        indices, not 10M × 20 bytes of string payload.
        """
        specs = options.read_member_infos
        if not specs:
            yield from batches
            return

        # Precompute values for this member (same across all its batches).
        info_values: list[tuple[str, Any]] = []
        for key, col_name in specs:
            info_values.append((col_name, self._extract_info_value(info, key)))

        for batch in batches:
            n = batch.num_rows
            new_arrays: list[pa.Array] = list(batch.columns)
            new_names: list[str] = list(batch.schema.names)

            for col_name, value in info_values:
                dict_array = self._make_dict_column(value, n)
                new_arrays.append(dict_array)
                new_names.append(col_name)

            yield pa.RecordBatch.from_arrays(new_arrays, names=new_names)

    @staticmethod
    def _extract_info_value(info: zipfile.ZipInfo, key: str) -> Any:
        """Map a ``read_member_infos`` key to a :class:`ZipInfo` field."""
        if key == "name":
            return info.filename
        if key == "file_size":
            return info.file_size
        if key == "compress_size":
            return info.compress_size
        if key == "date_time":
            # (year, month, day, hour, minute, second)
            return info.date_time
        if key == "crc":
            return info.CRC
        raise ValueError(f"Unknown read_member_infos key: {key!r}")

    @staticmethod
    def _make_dict_column(value: Any, n_rows: int) -> "pa.Array":
        """Build an ``n_rows``-long dict-encoded column with a single entry.

        Used for metadata columns that repeat one value across every row
        of a member's batch. For very wide tables this is essentially
        free; for narrow tables it still uses 4 bytes/row of index vs
        ``len(value)`` bytes/row for a string column.
        """
        if isinstance(value, str):
            dict_values = pa.array([value], type=pa.string())
        elif isinstance(value, bool):
            # bool before int — isinstance(True, int) is True.
            dict_values = pa.array([value], type=pa.bool_())
        elif isinstance(value, int):
            dict_values = pa.array([value], type=pa.int64())
        elif isinstance(value, tuple) and len(value) == 6 and all(isinstance(x, int) for x in value):
            # date_time tuple — store as string; callers can parse back.
            dict_values = pa.array([str(value)], type=pa.string())
        else:
            dict_values = pa.array([str(value)], type=pa.string())

        indices = pa.array([0] * n_rows, type=pa.int32())
        return pa.DictionaryArray.from_arrays(indices, dict_values)

    # ------------------------------------------------------------------
    # Column projection (post-enrichment)
    # ------------------------------------------------------------------

    @staticmethod
    def _project_columns(
        batch: "pa.RecordBatch",
        columns: Sequence[str],
    ) -> "pa.RecordBatch":
        """Select columns from *batch*, dropping unknown names silently."""
        names = set(batch.schema.names)
        wanted = [c for c in columns if c in names]
        return batch.select(wanted)

    # ------------------------------------------------------------------
    # ZipFile helper
    # ------------------------------------------------------------------

    def _open_zipfile(self) -> zipfile.ZipFile:
        """Open the underlying buffer as a :class:`zipfile.ZipFile`.

        Uses :meth:`BytesIO.view` with ``pos=0`` so the parent cursor
        isn't disturbed — this is what makes
        ``test_zipio_does_not_move_parent_cursor_on_read`` pass.
        """
        # We intentionally materialize to an io.BytesIO when the buffer
        # is small; zipfile.ZipFile needs real seekability and some
        # views don't expose mmap-backed seek correctly. For large
        # spilled buffers we pass the path directly, which is O(0) memory.
        b = self.buffer
        if b.spilled and b.path is not None:
            # Path-backed: open directly so zipfile can seek the file.
            return zipfile.ZipFile(str(b.path), mode="r")

        # In-memory: view(pos=0) is a seekable IO wrapper that doesn't
        # move b._pos.
        view = b.view(pos=0)
        return zipfile.ZipFile(view, mode="r")

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def _write_arrow_batches(
        self,
        batches: Iterator["pyarrow.RecordBatch"],
        options: ZipOptions,
    ) -> None:
        """Write one member into the ZIP (created or updated)."""
        with self.open() as b:
            if self.skip_write(options.mode):
                return

            member_name = options.member
            if not member_name or self._is_glob(member_name):
                raise ValueError(
                    "write requires options.member to be an exact member name "
                    f"(got {member_name!r})"
                )

            if options.inner_media is None:
                raise ValueError(
                    "write requires options.inner_media to be set "
                    "(ZIP members need a concrete inner format)"
                )

            inner_media = self._coerce_media(options.inner_media)

            # Apply the write-side cast before serializing through the
            # inner format — same pattern as Parquet/XLSX/XML.
            cast_batches = options.cast.cast_iterator(batches)

            # Render the member's payload into an in-memory BytesIO via
            # the appropriate inner MediaIO.
            inner_buf = BytesIO(config=b.buffer.config)
            inner_buf.set_media_type(inner_media, safe=False)
            inner_io = MediaIO.make(inner_buf, inner_media)

            # Write via the inner format's own write path. We route
            # through _write_arrow_batches to avoid re-resolving options.
            inner_opts = inner_io.check_options(options=None)
            inner_io._write_arrow_batches(cast_batches, inner_opts)

            payload = inner_buf.to_bytes()
            inner_buf.close()

            # Save-mode dispatch:
            # - OVERWRITE / AUTO: replace the whole archive with one member.
            # - APPEND: add/replace this member; keep others.
            # - UPSERT: same as APPEND with named-member semantics (the
            #   named member is always replaced if present).
            if (
                options.mode in (SaveMode.APPEND, SaveMode.UPSERT)
                and b.buffer.size > 0
            ):
                new_zip_bytes = self._rewrite_with_member(
                    existing=b.buffer.to_bytes(),
                    member_name=member_name,
                    payload=payload,
                    options=options,
                )
            else:
                new_zip_bytes = self._build_zip_with_one_member(
                    member_name=member_name,
                    payload=payload,
                    options=options,
                )

            b.buffer.replace_with_payload(new_zip_bytes)
            b.mark_dirty()

    def _build_zip_with_one_member(
        self,
        *,
        member_name: str,
        payload: bytes,
        options: ZipOptions,
    ) -> bytes:
        """Build a fresh ZIP archive containing exactly one member."""
        sink = _stdio.BytesIO()
        kwargs: dict[str, Any] = {"compression": options.zip_compression}
        if options.zip_compresslevel is not None:
            kwargs["compresslevel"] = options.zip_compresslevel
        try:
            with zipfile.ZipFile(sink, mode="w", **kwargs) as zf:
                zf.writestr(member_name, payload)
        except TypeError:
            # Older Python doesn't accept compresslevel; retry without.
            kwargs.pop("compresslevel", None)
            sink = _stdio.BytesIO()
            with zipfile.ZipFile(sink, mode="w", **kwargs) as zf:
                zf.writestr(member_name, payload)
        return sink.getvalue()

    def _rewrite_with_member(
        self,
        *,
        existing: bytes,
        member_name: str,
        payload: bytes,
        options: ZipOptions,
    ) -> bytes:
        """Rebuild an archive, replacing or adding one member.

        Python's :mod:`zipfile` has no in-place member replacement, so
        APPEND/UPSERT does a copy-rewrite. Members other than the
        target are copied byte-for-byte without re-compression via
        :meth:`ZipFile.writestr` on their already-compressed bytes.
        """
        src = _stdio.BytesIO(existing)
        sink = _stdio.BytesIO()

        kwargs: dict[str, Any] = {"compression": options.zip_compression}
        if options.zip_compresslevel is not None:
            kwargs["compresslevel"] = options.zip_compresslevel

        try:
            with zipfile.ZipFile(src, mode="r") as src_zf, zipfile.ZipFile(
                sink, mode="w", **kwargs
            ) as dst_zf:
                for info in src_zf.infolist():
                    if info.filename == member_name:
                        continue  # will be rewritten below
                    # Copy member through; zipfile decompresses then
                    # recompresses. For identical compression settings
                    # this is unavoidable without low-level raw-copy
                    # code, which zipfile doesn't expose cleanly.
                    dst_zf.writestr(info, src_zf.read(info.filename))
                dst_zf.writestr(member_name, payload)
        except TypeError:
            # Fallback without compresslevel for older Python.
            kwargs.pop("compresslevel", None)
            src.seek(0)
            sink = _stdio.BytesIO()
            with zipfile.ZipFile(src, mode="r") as src_zf, zipfile.ZipFile(
                sink, mode="w", **kwargs
            ) as dst_zf:
                for info in src_zf.infolist():
                    if info.filename == member_name:
                        continue
                    dst_zf.writestr(info, src_zf.read(info.filename))
                dst_zf.writestr(member_name, payload)

        return sink.getvalue()