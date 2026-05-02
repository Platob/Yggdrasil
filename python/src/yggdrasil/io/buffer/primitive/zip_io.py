"""Zip I/O for :class:`PrimitiveIO`.

:class:`ZipIO` treats a zip archive as a tabular container by
storing each :class:`pa.RecordBatch` as a separate entry encoded
in Arrow IPC streaming format. The zip itself does no compression
on the entries by default — Arrow IPC's framing is more efficient
than deflate over already-binary columnar data.

Save modes: OVERWRITE replaces the archive. APPEND adds new
entries (zip's central-directory layout makes this cheap).

Per-entry I/O via the Fragment surface
--------------------------------------

The archive is also navigable as a flat namespace of entries.
:meth:`ZipIO.read_fragments` yields one :class:`Fragment` per zip
entry, with:

* ``url`` = holder URL extended with ``#<entry-name>`` (the URL
  fragment selector captures the sub-location in one string);
* ``io`` = a live :class:`ZipEntryIO`, ready to read or write;
* ``parent`` = the holder's own root :class:`Fragment`, so callers
  can walk back up via :attr:`Fragment.ancestors` to the archive
  itself.

A :class:`ZipEntryIO` IS-A :class:`PrimitiveIO` whose backing buffer
is the entry's payload bytes — it supports the full
:class:`BytesIO` surface (read, write, seek, truncate, spill,
``memoryview``). On commit, the holder rewrites the archive with
the entry's new bytes substituted in. Multiple live handles to
the same entry name share buffer state through a per-holder live
map keyed by name.

Dirty tracking
--------------

:class:`BytesIO` itself has no dirty/clean concept — its contract
is "you wrote bytes, those bytes are there." But :class:`ZipEntryIO`
needs to know whether to bother committing back to the holder, so
it tracks ``_dirty`` itself by overriding the two write primitives
(``_write_at`` and ``_set_size``). ``_acquire`` clears the flag
after pulling the entry's bytes in (acquire is not a user-driven
mutation); ``_release`` checks the flag and runs the commit only
when it's True.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import io as _io
import time
import zipfile
from typing import Any, ClassVar, Iterable, Iterator
from weakref import WeakValueDictionary

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.enums import MimeTypes, Mode, MediaType, MediaTypes
from yggdrasil.io.fragment import Fragment, FragmentInfos
from .base import PrimitiveIO

__all__ = ["ZipIO", "ZipOptions", "ZipEntryIO", "ZipEntryOptions"]


_ENTRY_NAME = "batch-{:06d}.arrow"
_ENTRY_PREFIX = "batch-"


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ZipOptions(CastOptions):
    """:class:`CastOptions` extended with Zip-specific knobs."""

    compression: int = zipfile.ZIP_STORED
    compresslevel: "int | None" = None
    entry_name_template: str = _ENTRY_NAME


@dataclasses.dataclass(frozen=True, slots=True)
class ZipEntryOptions(CastOptions):
    """:class:`CastOptions` for per-entry tabular I/O.

    A :class:`ZipEntryIO` stores one logical Arrow IPC stream per
    entry; OVERWRITE rewrites the entry's payload, APPEND concatenates
    batches into the same entry. The archive's framing knobs (zip
    compression, name template) live on the holder's :class:`ZipOptions`
    and are not duplicated here.
    """


# ---------------------------------------------------------------------------
# ZipIO
# ---------------------------------------------------------------------------


class ZipIO(PrimitiveIO):
    """:class:`PrimitiveIO` for zip-archived Arrow IPC batch sequences."""

    __slots__ = ("_live_entries",)

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # name -> live ZipEntryIO (weak so closed handles drop out).
        # Two opens of the same name share the SAME ZipEntryIO, giving
        # POSIX-shared-fd semantics across handles.
        self._live_entries: "WeakValueDictionary[str, ZipEntryIO]" = (
            WeakValueDictionary()
        )

    @classmethod
    def default_mime_type(cls):
        return MimeTypes.ZIP

    @classmethod
    def options_class(cls):
        return ZipOptions

    # ==================================================================
    # Schema
    # ==================================================================

    def _collect_schema(self, options: ZipOptions) -> Schema:
        if self.is_empty():
            return Schema.empty()
        first = next(iter(self._read_arrow_batches(options)), None)
        if first is None:
            return Schema.empty()
        return Schema.from_arrow(first.schema)

    # ==================================================================
    # Read path — Arrow batches across all batch-* entries
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: ZipOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield record batches by enumerating zip entries in name order."""
        for fragment in self.read_fragments(open_io=True):
            yield from fragment.io.read_arrow_batches(options)

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ZipOptions,
    ) -> None:
        """Persist batches as zip entries.

        OVERWRITE truncates and writes a fresh archive. APPEND opens
        the existing archive in ``"a"`` mode and adds entries with
        names continuing from the last existing index.
        """
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action not in (Mode.OVERWRITE, Mode.APPEND):
            raise NotImplementedError(
                f"{type(self).__name__}._write_arrow_batches handles "
                f"OVERWRITE and APPEND; got {action!r}."
            )

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None:
            return

        if options.target_field is not None:
            first = options.cast_arrow_tabular(first)

        is_append = action is Mode.APPEND and not self.is_empty()

        if is_append:
            start_index = self._next_entry_index(options)
            zip_mode = "a"
        else:
            start_index = 0
            zip_mode = "w"

        lifecycle = options.copy(
            truncate_before_write=not is_append,
            # zipfile in append mode reads the central directory
            # from the end of the file, then seeks itself; no need
            # for write_seek=-1 here.
        )

        zf_kwargs: dict[str, Any] = {
            "mode": zip_mode,
            "compression": options.compression,
        }
        if options.compresslevel is not None:
            zf_kwargs["compresslevel"] = options.compresslevel

        with self._writing_context(lifecycle) as io:
            with zipfile.ZipFile(io, **zf_kwargs) as zf:
                self._write_entry(zf, start_index, first, options)
                for i, batch in enumerate(iterator, start=start_index + 1):
                    if options.target_field is not None:
                        batch = options.cast_arrow_tabular(batch)
                    self._write_entry(zf, i, batch, options)

    def _next_entry_index(self, options: ZipOptions) -> int:
        """Return one past the highest existing batch-index in the zip."""
        with self._reading_context(options) as io:
            with zipfile.ZipFile(io, mode="r") as zf:
                indexes = []
                for name in zf.namelist():
                    if not name.startswith(_ENTRY_PREFIX):
                        continue
                    stem = name[len(_ENTRY_PREFIX):].split(".", 1)[0]
                    try:
                        indexes.append(int(stem))
                    except ValueError:
                        continue
                return max(indexes) + 1 if indexes else 0

    @staticmethod
    def _write_entry(
        zf: zipfile.ZipFile,
        index: int,
        batch: pa.RecordBatch,
        options: ZipOptions,
    ) -> None:
        buf = _io.BytesIO()
        writer = ipc.RecordBatchStreamWriter(buf, batch.schema)
        try:
            writer.write_batch(batch)
        finally:
            writer.close()
        name = options.entry_name_template.format(index)
        zf.writestr(name, buf.getvalue())

    # ==================================================================
    # Fragment surface — the zip archive viewed as a flat namespace
    # ==================================================================

    def read_fragments(
        self,
        *,
        key: "str | None" = None,
        open_io: bool = True,
    ) -> Iterator[Fragment]:
        """Yield one :class:`Fragment` per zip entry.

        Each fragment carries:

        * ``url`` = the holder's URL with the entry name set as the
          URL fragment (``zip://archive.zip#batch-000000.arrow``).
          One string captures both the parent file and the
          intra-file location.
        * ``io`` = a live :class:`ZipEntryIO` opened on this entry,
          ready for ``read_arrow_batches`` / direct buffer reads.
        * ``parent`` = the holder's own root fragment (built once
          per call), so :attr:`Fragment.ancestors` walks back up
          to the archive itself.
        * ``mtime`` = the entry's stored modification time encoded
          to seconds-since-epoch, falling back to the holder's
          ``mtime`` when the zip entry's date_time is unset.
        * ``schema`` = ``None`` (lazy — read-on-demand via the
          attached IO; populating eagerly would force-decode every
          entry just to enumerate).

        :param key: optional :func:`fnmatch` pattern against entry
            names — only matching entries are yielded.
        :param open_io: when ``True`` (default) every yielded fragment
            has a live :class:`ZipEntryIO` attached. When ``False``,
            the IO is detached via :meth:`Fragment.without_io` —
            useful for cheap directory walks where the caller will
            attach IO selectively later.

        Note: there is no ``recursive`` knob. Zip's namespace is
        flat by design — name segments (``"a/b/c.json"``) are part
        of the key, not a nested IO surface.
        """
        # Build the holder's root fragment once — every yielded
        # entry-fragment points at it as parent.
        root = self._self_fragment()

        if self.is_empty():
            return

        with self._reading_context(self._default_options()) as io:
            with zipfile.ZipFile(io, mode="r") as zf:
                infos = list(zf.infolist())

        for info in infos:
            if key is not None and not fnmatch.fnmatch(info.filename, key):
                continue
            frag = self._fragment_for(info, parent=root)
            if not open_io:
                frag = frag.without_io()
            yield frag

    def fragment_for(self, name: str) -> Fragment:
        """Return a :class:`Fragment` for ``name`` (need not exist).

        Useful when constructing a write target: open the returned
        fragment's IO, write into it, and the holder records the new
        entry on commit.
        """
        info = self._entry_info(name)
        return self._fragment_for(
            info if info is not None else _synthetic_info(name),
            parent=self._self_fragment(),
        )

    # Backwards-compatible alias for callers wanting the holder's
    # name-keyed entry surface without the Fragment apparatus.
    entry = fragment_for

    def __iter__(self) -> Iterator[Fragment]:
        return self.read_fragments()

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self._entry_exists(name)

    def list_entries(self) -> list[str]:
        """Return all entry names currently in the archive, sorted."""
        if self.is_empty():
            return []
        with self._reading_context(self._default_options()) as io:
            with zipfile.ZipFile(io, mode="r") as zf:
                return sorted(zf.namelist())

    # ------------------------------------------------------------------
    # Fragment construction internals
    # ------------------------------------------------------------------

    def _self_fragment(self) -> Fragment:
        """Build the holder's own root :class:`Fragment` for parent-linking."""
        url = getattr(self, "url", None)
        infos = FragmentInfos(
            url=url,
            mtime=float(getattr(self, "mtime", 0.0) or 0.0),
            schema=None,
        )
        # The holder fragment carries `self` as its IO so consumers
        # walking up via `ancestors` reach a usable archive handle.
        return Fragment(infos=infos, io=self, parent=None)

    def _fragment_for(
        self,
        info: zipfile.ZipInfo,
        parent: Fragment,
    ) -> Fragment:
        """Build a per-entry :class:`Fragment` with attached IO."""
        entry_url = self._entry_url(info.filename)
        mtime = _zipinfo_mtime(info, fallback=parent.infos.mtime)
        infos = FragmentInfos(url=entry_url, mtime=mtime, schema=None)
        entry_io = self._open_entry_io(info.filename, auto_open=True)
        return Fragment(infos=infos, io=entry_io, parent=parent)

    def _entry_url(self, name: str):
        """Compose ``holder_url#entry-name`` for a Fragment URL."""
        base = getattr(self, "url", None)
        if base is None:
            return None
        # URL is expected to expose with_fragment(); fall back to a
        # raw replace if the implementation differs.
        with_fragment = getattr(base, "with_fragment", None)
        if with_fragment is not None:
            return with_fragment(name)
        return dataclasses.replace(base, fragment=name)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Entry-namespace internals (used by Fragment construction & ZipEntryIO)
    # ------------------------------------------------------------------

    def _default_options(self) -> ZipOptions:
        """Return a :class:`ZipOptions` with this class's defaults."""
        return ZipOptions()

    def _entry_info(self, name: str) -> "zipfile.ZipInfo | None":
        if self.is_empty():
            return None
        with self._reading_context(self._default_options()) as io:
            with zipfile.ZipFile(io, mode="r") as zf:
                try:
                    return zf.getinfo(name)
                except KeyError:
                    return None

    def _entry_exists(self, name: str) -> bool:
        return self._entry_info(name) is not None

    def _read_entry_bytes(self, name: str) -> bytes:
        """Return the entry's uncompressed payload, or empty bytes."""
        if self.is_empty():
            return b""
        with self._reading_context(self._default_options()) as io:
            with zipfile.ZipFile(io, mode="r") as zf:
                try:
                    return zf.read(name)
                except KeyError:
                    return b""

    def _open_entry_io(self, name: str, *, auto_open: bool) -> "ZipEntryIO":
        existing = self._live_entries.get(name)
        if existing is not None:
            return existing
        entry_io = ZipEntryIO(holder=self, name=name, auto_open=auto_open)
        self._live_entries[name] = entry_io
        return entry_io

    def _commit_entry_payload(self, name: str, payload: bytes | bytearray) -> None:
        """Persist ``payload`` as the contents of entry ``name``.

        Implements per-entry write-back from a :class:`ZipEntryIO`
        commit. Because zip's central directory is at the end of the
        file, replacing one entry requires rewriting the archive; we
        do so by streaming all other entries through a fresh
        :class:`zipfile.ZipFile` and substituting ``payload`` for
        ``name``. If ``name`` does not yet exist, it is appended via
        the cheap APPEND path.

        After this returns, the holder's underlying buffer reflects
        the new archive bytes. The holder's own ``_release`` is
        responsible for propagating those bytes to its bound path,
        if any.
        """
        opts = self._default_options()
        payload_bytes = bytes(payload)

        if self.is_empty() or not self._entry_exists(name):
            # No rewrite needed — append into existing (or fresh) archive.
            zip_mode = "a" if not self.is_empty() else "w"
            lifecycle = opts.copy(truncate_before_write=self.is_empty())
            zf_kwargs: dict[str, Any] = {
                "mode": zip_mode,
                "compression": opts.compression,
            }
            if opts.compresslevel is not None:
                zf_kwargs["compresslevel"] = opts.compresslevel
            with self._writing_context(lifecycle) as io:
                with zipfile.ZipFile(io, **zf_kwargs) as zf:
                    zf.writestr(name, payload_bytes)
            return

        # Slow path: rewrite archive in place, swapping one entry.
        # Stage to a fresh BytesIO, then atomically replace the holder's
        # buffer payload via the BytesIO base helper.
        staged = _io.BytesIO()
        with self._reading_context(opts) as src_io:
            with zipfile.ZipFile(src_io, mode="r") as src_zf:
                zf_kwargs = {
                    "mode": "w",
                    "compression": opts.compression,
                }
                if opts.compresslevel is not None:
                    zf_kwargs["compresslevel"] = opts.compresslevel
                with zipfile.ZipFile(staged, **zf_kwargs) as dst_zf:
                    for info in src_zf.infolist():
                        if info.filename == name:
                            continue
                        dst_zf.writestr(info, src_zf.read(info.filename))
                    dst_zf.writestr(name, payload_bytes)

        self.replace_with_payload(staged.getvalue())

    def _delete_entry(self, name: str) -> None:
        """Remove ``name`` from the archive. No-op if not present."""
        if self.is_empty() or not self._entry_exists(name):
            return
        opts = self._default_options()
        staged = _io.BytesIO()
        with self._reading_context(opts) as src_io:
            with zipfile.ZipFile(src_io, mode="r") as src_zf:
                zf_kwargs: dict[str, Any] = {
                    "mode": "w",
                    "compression": opts.compression,
                }
                if opts.compresslevel is not None:
                    zf_kwargs["compresslevel"] = opts.compresslevel
                with zipfile.ZipFile(staged, **zf_kwargs) as dst_zf:
                    for info in src_zf.infolist():
                        if info.filename == name:
                            continue
                        dst_zf.writestr(info, src_zf.read(info.filename))
        self.replace_with_payload(staged.getvalue())


# ---------------------------------------------------------------------------
# ZipEntryIO
# ---------------------------------------------------------------------------


class ZipEntryIO(PrimitiveIO):
    """A :class:`PrimitiveIO` view of a single entry inside a :class:`ZipIO`.

    The entry's uncompressed payload IS this object's backing
    :class:`BytesIO` — read/write/seek/truncate behave exactly as on
    a free-standing buffer. Lifecycle:

    * ``_acquire`` pulls the entry's payload out of the holder and
      installs it as this buffer's bytes (or starts empty if the
      entry does not yet exist), then clears the dirty flag.
    * Any user-driven mutation (write, truncate, etc.) flows
      through ``_write_at`` / ``_set_size`` and flips ``_dirty``
      to True.
    * ``_release`` writes the buffer's payload back into the
      holder via :meth:`ZipIO._commit_entry_payload` if ``_dirty``
      is True. The holder rewrites its archive bytes in place; its
      own ``_release`` propagates the change to the underlying path.

    Tabular reads/writes use Arrow IPC streaming format — one logical
    stream per entry — so the entry can hold many record batches with
    OVERWRITE and APPEND save modes for in-entry concatenation.

    Notes
    -----
    Multiple ``open_entry`` / fragment-resolution calls for the same
    name on the same holder return the *same* :class:`ZipEntryIO`
    instance (POSIX-shared-fd semantics): two readers see each
    other's writes through one buffer. Independent ``ZipIO`` holders
    backed by the same path do not share live-entry state.

    The fragment surface attaches a :class:`ZipEntryIO` to each
    yielded :class:`Fragment` as its ``io``; callers who want the
    enclosing archive walk back up via :attr:`Fragment.parent` /
    :attr:`Fragment.ancestors`.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    def __init__(
        self,
        *,
        holder: ZipIO,
        name: str,
        auto_open: bool = True,
        **kwargs: Any,
    ) -> None:
        # Forward kwargs to BytesIO/PrimitiveIO so spill_bytes,
        # spill_ttl, etc. flow through normally. We want acquire to
        # run AFTER our slots are in place so _acquire can clear
        # _dirty without AttributeError; pass auto_open=False to
        # super and open ourselves below.
        super().__init__(auto_open=False, **kwargs)
        self._holder = holder
        self._entry_name = name
        self._dirty = False
        if auto_open:
            self.open()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def holder(self) -> ZipIO:
        return self._holder

    @property
    def entry_name(self) -> str:
        return self._entry_name

    @property
    def url(self):
        """Return ``holder.url#entry-name`` so the IO has a self-locator."""
        return self._holder._entry_url(self._entry_name)

    # ------------------------------------------------------------------
    # MimeType / options
    # ------------------------------------------------------------------

    @classmethod
    def default_mime_type(cls):
        # An entry is opaque-by-default; tabular paths use the per-entry
        # ZipEntryOptions and Arrow IPC framing internally. Callers that
        # want a richer mime can pass `with_media_type` after open.
        return MimeTypes.ZIP_ENTRY

    @classmethod
    def options_class(cls):
        return ZipEntryOptions

    # ------------------------------------------------------------------
    # Dirty tracking — override the two write primitives
    # ------------------------------------------------------------------
    #
    # BytesIO's contract is "if you wrote, the bytes are there" — it
    # has no dirty/clean concept. ZipEntryIO needs one because the
    # holder commit is OUR concern, not the buffer's. Per BytesIO's
    # own docstring, all public mutation flows through _write_at and
    # _set_size, so hooking those two is sufficient.
    #
    # _acquire's `replace_with_payload` (when populating from holder
    # bytes) goes through neither — it tears down + reinits. We
    # explicitly clear _dirty after `_acquire` to keep the flag
    # honest. delete() is similar: it calls replace_with_payload(b"")
    # then clears the flag, since the holder is already in sync.

    def _write_at(self, data, pos: int) -> int:
        n = super()._write_at(data, pos)
        if n > 0:
            self._dirty = True
        return n

    def _set_size(self, n: int) -> int:
        # Even a no-op truncate to the same size we'll skip dirtying;
        # but a truncate that changes size (grow or shrink) IS a
        # mutation that needs to commit back.
        before = self._size
        result = super()._set_size(n)
        if result != before:
            self._dirty = True
        return result

    # ------------------------------------------------------------------
    # Lifecycle — bridge buffer <-> holder entry
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        # Pull current entry payload (if any) into self before any
        # BytesIO read/write sees the buffer.
        super()._acquire()
        payload = self._holder._read_entry_bytes(self._entry_name)
        if payload:
            # replace_with_payload swaps the backing bytes wholesale
            # and resets the cursor to 0. It tears down + reinits, so
            # it doesn't go through _write_at — _dirty stays as we
            # set it.
            self.replace_with_payload(payload)
        # Acquire is not, by itself, a mutation of the entry.
        self._dirty = False

    def _release(self, committed: bool = False) -> None:
        try:
            if self._dirty:
                # Snapshot bytes through memoryview to avoid an extra
                # copy on the memory path; bytes() materializes for
                # zipfile's API.
                with self.memoryview() as mv:
                    payload = bytes(mv)
                self._holder._commit_entry_payload(self._entry_name, payload)
                self._dirty = False
        finally:
            # Drop ourselves from the holder's live map so a fresh
            # open returns a fresh handle. WeakValueDictionary will
            # also clear us out on GC, but explicit is friendlier.
            live = self._holder._live_entries
            if live.get(self._entry_name) is self:
                try:
                    del live[self._entry_name]
                except KeyError:  # pragma: no cover - benign race
                    pass
            super()._release(committed=committed)

    # ------------------------------------------------------------------
    # Entry-level mutation helpers
    # ------------------------------------------------------------------

    def delete(self) -> None:
        """Remove this entry from the holder archive.

        After ``delete``, the buffer is reset to empty and the entry
        is gone from the archive — both sides consistent, so we
        clear ``_dirty``. Further writes will recreate the entry on
        the next commit.
        """
        self._holder._delete_entry(self._entry_name)
        self.replace_with_payload(b"")
        self._dirty = False

    # ------------------------------------------------------------------
    # Tabular I/O — Arrow IPC stream per entry
    # ------------------------------------------------------------------

    def _collect_schema(self, options: ZipEntryOptions) -> Schema:
        if self.is_empty():
            return Schema.empty()
        first = next(iter(self._read_arrow_batches(options)), None)
        if first is None:
            return Schema.empty()
        return Schema.from_arrow(first.schema)

    def _read_arrow_batches(
        self,
        options: ZipEntryOptions,
    ) -> Iterator[pa.RecordBatch]:
        if self.is_empty():
            return

        mt = MediaType.from_(self.entry_name, default=MediaTypes.OCTET_STREAM)

        if mt.is_octet:
            return

        with self.as_media(media_type=mt) as media:
            yield from media.read_arrow_batches(options)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ZipEntryOptions,
    ) -> None:
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action not in (Mode.OVERWRITE, Mode.APPEND):
            raise NotImplementedError(
                f"{type(self).__name__}._write_arrow_batches handles "
                f"OVERWRITE and APPEND; got {action!r}."
            )

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None:
            return
        if options.target_field is not None:
            first = options.cast_arrow_tabular(first)

        is_append = action is Mode.APPEND and not self.is_empty()

        if is_append:
            # Arrow IPC streams concat cleanly: open existing payload,
            # seek to end, write more batches with the same schema.
            lifecycle = options.copy(
                truncate_before_write=False,
                write_seek=-1,  # append at EOF
            )
        else:
            lifecycle = options.copy(truncate_before_write=True)

        with self._writing_context(lifecycle) as io:
            writer = ipc.RecordBatchStreamWriter(io, first.schema)
            try:
                writer.write_batch(first)
                for batch in iterator:
                    if options.target_field is not None:
                        batch = options.cast_arrow_tabular(batch)
                    writer.write_batch(batch)
            finally:
                writer.close()
        # The writer wrote bytes through io's BytesIO surface, which
        # ultimately called our _write_at — so _dirty is already True.
        # No explicit set needed.

    # ------------------------------------------------------------------
    # Fragment helper — promote this IO into a Fragment
    # ------------------------------------------------------------------

    def as_fragment(self) -> Fragment:
        """Build a :class:`Fragment` describing this entry.

        Parent is the holder's root fragment. Useful for callers
        that materialize a :class:`ZipEntryIO` directly (without
        going through :meth:`ZipIO.read_fragments`) and want to feed
        it into Fragment-shaped pipelines.
        """
        info = self._holder._entry_info(self._entry_name)
        zinfo = info if info is not None else _synthetic_info(self._entry_name)
        parent = self._holder._self_fragment()
        infos = FragmentInfos(
            url=self.url,
            mtime=_zipinfo_mtime(zinfo, fallback=parent.infos.mtime),
            schema=None,
        )
        return Fragment(infos=infos, io=self, parent=parent)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}("
            f"holder=<{type(self._holder).__name__}>, "
            f"name={self._entry_name!r}, "
            f"dirty={self._dirty})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zipinfo_mtime(info: zipfile.ZipInfo, *, fallback: float) -> float:
    """Convert a ZipInfo's date_time tuple to seconds-since-epoch.

    Zip's ``date_time`` defaults to ``(1980, 1, 1, 0, 0, 0)`` for
    entries that never had a real timestamp; we treat that sentinel
    as "no useful mtime" and return the holder's fallback instead.
    """
    dt = getattr(info, "date_time", None)
    if not dt or dt == (1980, 1, 1, 0, 0, 0):
        return fallback
    try:
        return time.mktime((*dt, 0, 0, -1))
    except (ValueError, OverflowError):  # pragma: no cover - degenerate input
        return fallback


def _synthetic_info(name: str) -> zipfile.ZipInfo:
    """A ZipInfo placeholder for entries that don't yet exist in the archive.

    Used by ``fragment_for`` so callers can build a Fragment for a
    write target before the entry is committed.
    """
    return zipfile.ZipInfo(filename=name)