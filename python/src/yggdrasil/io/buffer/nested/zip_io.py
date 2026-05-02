"""Zip archive as a :class:`NestedIO` with single-file storage.

:class:`ZipIO` models a single zip archive on disk as a folder-shaped
:class:`NestedIO`: its children are zip entries (each yielded as a
:class:`ZipEntryIO`), reads enumerate the archive's central directory,
writes mint new entries by name. Unlike :class:`FolderIO` whose
backing store is a directory tree, :class:`ZipIO`'s backing store is
one zip file — the children-as-IOs surface matches :class:`NestedIO`'s
contract, so the same read/write derivations flow through it.

The contract
------------

A :class:`ZipIO` holds a single :class:`Path` (the archive file).
:meth:`iter_children` opens the archive and yields one
:class:`ZipEntryIO` per entry; :meth:`make_child` mints a fresh entry
handle bound to a name. Each yielded child carries ``parent = self``
so consumers can walk back up to the archive.

Per-entry I/O — :class:`ZipEntryIO` IS-A :class:`BytesIO`
---------------------------------------------------------

A :class:`ZipEntryIO` is a :class:`BytesIO` that knows how to read
and write itself autonomously inside its parent archive. Its
backing buffer is the entry's payload bytes — full
:class:`BytesIO` surface (read, write, seek, truncate, spill,
``memoryview``) plus a few extras:

* ``parent`` — the :class:`ZipIO` archive this entry belongs to.
* ``entry_name`` — the entry's filename inside the archive.
* ``zip_info`` — the :class:`zipfile.ZipInfo` carrying compression,
  CRC, mtime, size, etc., as recorded in the central directory.
  Lazily populated on acquire; ``None`` until the entry exists.
* ``compression`` / ``compresslevel`` — per-entry overrides, falling
  back to the parent's :class:`ZipOptions` defaults on commit.

Tabular reads/writes (Arrow IPC, Parquet, …) flow through the
:class:`BytesIO` tabular contract: setting a tabular ``media_type``
on the entry routes :meth:`read_arrow_batches` /
:meth:`write_arrow_batches` to the registered concrete leaf
(``ArrowIPCIO``, ``ParquetIO``, …) wrapping the entry's buffer. No
PrimitiveIO inheritance needed — the entry is a byte buffer with
parent-aware lifecycle and lets the format leaves do format work.

Dirty tracking
--------------

:class:`BytesIO` itself has no dirty/clean concept — its contract is
"you wrote bytes, those bytes are there." But :class:`ZipEntryIO`
needs to know whether to bother committing back to the parent, so it
tracks ``_dirty`` itself by overriding the two write primitives
(``_write_at`` and ``_set_size``). ``_acquire`` clears the flag after
pulling the entry's bytes in (acquire is not a user-driven mutation);
``_release`` checks the flag and runs the commit only when it's True.

Optimized commit path
---------------------

:meth:`ZipIO._commit_entry_payload` picks the cheapest write:

* New entry, archive missing — create the archive with one entry.
* New entry, archive non-empty — append to the central directory in
  place (zip-native append), no rewrite.
* Existing entry, payload unchanged — skipped via dirty tracking.
* Existing entry, payload changed — stream all other entries through
  a fresh archive with the swapped payload, then atomically replace
  the file. Required because zip's central directory sits at EOF and
  per-entry sizes change on rewrite.
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

from yggdrasil.data.options import CastOptions
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.enums import MediaType, MediaTypes, MimeTypes
from .base import NestedIO, NestedOptions

__all__ = ["ZipIO", "ZipOptions", "ZipEntryIO", "ZipEntryOptions"]


_ENTRY_PREFIX = "batch-"
_ENTRY_NAME = "batch-{:06d}.arrow"


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ZipOptions(NestedOptions):
    """:class:`NestedOptions` extended with zip-archive knobs.

    Inherits child-routing (``child_media_type``, ``child_row_size``,
    ``child_byte_size``) from :class:`NestedOptions` and adds the
    archive's framing knobs.
    """

    compression: int = zipfile.ZIP_STORED
    compresslevel: "int | None" = None
    entry_name_template: str = _ENTRY_NAME


@dataclasses.dataclass(frozen=True, slots=True)
class ZipEntryOptions(CastOptions):
    """Options for per-entry tabular I/O on a :class:`ZipEntryIO`.

    The entry stores one logical Arrow IPC stream; OVERWRITE rewrites
    the entry's payload, APPEND concatenates batches into the same
    entry. Archive-level framing (compression, name template) lives
    on the parent's :class:`ZipOptions`.
    """


# ---------------------------------------------------------------------------
# ZipIO
# ---------------------------------------------------------------------------


class ZipIO(NestedIO[ZipOptions]):
    """A :class:`NestedIO` whose storage is a single zip archive file.

    Children are zip entries, each yielded as a :class:`ZipEntryIO`.
    The archive itself is one path on disk — there's no folder tree.
    Reading drains every entry's Arrow IPC stream into one batch
    iterator; writing splits the input into one or more entries
    according to :attr:`NestedOptions.child_row_size` /
    ``child_byte_size`` (default: one entry per write call).
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_mime_type(cls):
        return MimeTypes.ZIP

    @classmethod
    def options_class(cls):
        return ZipOptions

    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        parent: "NestedIO | None" = None,
        auto_open: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data, path=path, parent=parent, auto_open=auto_open, **kwargs
        )
        # name -> live ZipEntryIO (weak so closed handles drop out).
        # Two opens of the same name share the SAME ZipEntryIO, giving
        # POSIX-shared-fd semantics across handles.
        self._live_entries: "WeakValueDictionary[str, ZipEntryIO]" = (
            WeakValueDictionary()
        )

    def _default_child_media_type(self) -> Any:
        # Arrow IPC stream is the canonical entry payload for tabular
        # writes. Callers that want a different format pass
        # ``options.child_media_type`` explicitly.
        return MimeTypes.ARROW_IPC

    # ==================================================================
    # Children — enumerate the archive's central directory
    # ==================================================================

    def _iter_children(
        self,
        options: ZipOptions,
    ) -> "Iterator[ZipEntryIO]":
        """Yield one :class:`ZipEntryIO` per entry in the archive.

        Hidden entries (name starting with ``.``) are filtered out by
        :meth:`_is_ignored_name`. Each yielded child has its
        ``parent`` attribute stamped to ``self``; multiple yields for
        the same name share one backing :class:`ZipEntryIO`.
        """
        if not self.path.exists():
            return
        for name in self._list_entry_names():
            if self._is_ignored_name(name):
                continue
            yield self._attach(self._open_entry_io(name, auto_open=False))

    def make_child(
        self,
        name: str,
        *,
        media_type: Any = None,
    ) -> "ZipEntryIO":
        """Mint a closed :class:`ZipEntryIO` bound to entry ``name``.

        The entry need not exist yet; opening the returned IO either
        pulls existing payload bytes or starts empty, and a dirty
        commit on close writes the entry into the archive. The
        ``media_type`` argument is accepted for parity with the
        :class:`NestedIO` contract but is ignored — entry framing is
        always Arrow IPC for the tabular surface; raw bytes pass
        through unchanged.
        """
        del media_type  # parent's framing is unchanged across mime types
        if "\\" in name:
            raise ValueError(
                f"Entry name must not contain backslashes; got {name!r}. "
                "Use forward slashes."
            )
        return self._attach(self._open_entry_io(name, auto_open=False))

    # Convenience alias — name-keyed access to entries without going
    # through the children iterator.
    def entry(self, name: str) -> "ZipEntryIO":
        return self.make_child(name)

    def __iter__(self) -> "Iterator[ZipEntryIO]":
        return self._iter_children(self._default_options())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self._entry_exists(name)

    def list_entries(self) -> list[str]:
        """Return all entry names currently in the archive, sorted."""
        return sorted(self._list_entry_names())

    # ==================================================================
    # Empty / clear
    # ==================================================================

    def is_empty(self) -> bool:
        """True if the archive is missing or has no non-ignored entries."""
        if not self.path.exists():
            return True
        try:
            with zipfile.ZipFile(self.path.full_path(), mode="r") as zf:
                for info in zf.infolist():
                    if not self._is_ignored_name(info.filename):
                        return False
                return True
        except (zipfile.BadZipFile, FileNotFoundError):
            return True

    def _is_ignored_name(self, name: str) -> bool:
        """Hide hidden entries (name starts with ``.``) from enumeration."""
        return name.startswith(".")

    def _clear_children(self) -> None:
        """Reset the archive to empty by writing a fresh empty zip."""
        if not self.path.exists():
            return
        with zipfile.ZipFile(
            self.path.full_path(), mode="w", compression=zipfile.ZIP_STORED
        ):
            pass

    # ==================================================================
    # Write derivation — entry-atomic commits, no on-disk staging
    # ==================================================================

    def _write_one_child(
        self,
        batches: "Iterable[pa.RecordBatch]",
        *,
        media_type: Any,
        options: ZipOptions,
    ) -> None:
        """Drain *batches* into one fresh zip entry.

        Overrides :meth:`NestedIO._write_one_child` to skip the
        staging-file dance: a zip entry is committed atomically to
        the central directory when the :class:`ZipEntryIO` releases,
        so there's nothing to rename. On exception we best-effort
        delete the half-written entry.
        """
        name = self._next_child_name(media_type=media_type, options=options)
        child = self.make_child(name, media_type=media_type)
        try:
            with child:
                child.write_arrow_batches(batches, options=options)
        except Exception:
            try:
                self._delete_entry(name)
            except Exception:
                pass
            raise

    def _next_child_name(
        self,
        *,
        media_type: Any = None,
        options: "ZipOptions | None" = None,
    ) -> str:
        """Mint the next ``batch-{N:06d}.arrow`` entry name.

        Scans existing entries with the configured prefix to find the
        highest index and adds one. Falls back to index 0 when the
        archive is missing or unreadable.
        """
        del media_type
        template = (
            options.entry_name_template if options is not None else _ENTRY_NAME
        )
        max_idx = -1
        if self.path.exists():
            try:
                with zipfile.ZipFile(self.path.full_path(), mode="r") as zf:
                    for n in zf.namelist():
                        if not n.startswith(_ENTRY_PREFIX):
                            continue
                        stem = n[len(_ENTRY_PREFIX):].split(".", 1)[0]
                        try:
                            idx = int(stem)
                        except ValueError:
                            continue
                        if idx > max_idx:
                            max_idx = idx
            except (zipfile.BadZipFile, FileNotFoundError):
                pass
        return template.format(max_idx + 1)

    # ==================================================================
    # Entry-namespace internals (used by ZipEntryIO)
    # ==================================================================

    def _list_entry_names(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            with zipfile.ZipFile(self.path.full_path(), mode="r") as zf:
                return list(zf.namelist())
        except (zipfile.BadZipFile, FileNotFoundError):
            return []

    def _entry_info(self, name: str) -> "zipfile.ZipInfo | None":
        if not self.path.exists():
            return None
        try:
            with zipfile.ZipFile(self.path.full_path(), mode="r") as zf:
                try:
                    return zf.getinfo(name)
                except KeyError:
                    return None
        except (zipfile.BadZipFile, FileNotFoundError):
            return None

    def _entry_exists(self, name: str) -> bool:
        return self._entry_info(name) is not None

    def _read_entry_bytes(self, name: str) -> bytes:
        """Return entry ``name``'s uncompressed payload, or empty bytes."""
        if not self.path.exists():
            return b""
        try:
            with zipfile.ZipFile(self.path.full_path(), mode="r") as zf:
                try:
                    return zf.read(name)
                except KeyError:
                    return b""
        except (zipfile.BadZipFile, FileNotFoundError):
            return b""

    def _commit_entry_payload(
        self,
        name: str,
        payload: bytes | bytearray,
        *,
        options: "ZipOptions | None" = None,
        compression: "int | None" = None,
        compresslevel: "int | None" = None,
    ) -> None:
        """Persist ``payload`` as the contents of entry ``name``.

        Implements per-entry write-back from a :class:`ZipEntryIO`
        commit. Two paths:

        * **Cheap append** — entry doesn't exist (or archive is
          missing). Open the archive in ``"a"`` mode (or ``"w"`` if
          missing) and append the entry to the central directory.
          No rewrite.
        * **Rewrite swap** — entry exists. Stream all other entries
          through a fresh in-memory archive with ``payload``
          substituted for ``name``, then atomically replace the
          file. Required because zip's central directory sits at
          EOF and per-entry sizes change on rewrite.

        Per-entry ``compression`` / ``compresslevel`` overrides take
        precedence over the parent's :class:`ZipOptions` defaults.
        """
        opts = options if options is not None else ZipOptions()
        payload_bytes = bytes(payload)
        effective_compression = (
            compression if compression is not None else opts.compression
        )
        effective_compresslevel = (
            compresslevel if compresslevel is not None else opts.compresslevel
        )
        zf_kwargs: dict[str, Any] = {"compression": effective_compression}
        if effective_compresslevel is not None:
            zf_kwargs["compresslevel"] = effective_compresslevel

        if not self.path.exists() or not self._entry_exists(name):
            mode = "a" if self.path.exists() else "w"
            with zipfile.ZipFile(self.path.full_path(), mode=mode, **zf_kwargs) as zf:
                zf.writestr(name, payload_bytes)
            return

        # Slow path: rewrite archive in place, swapping one entry.
        # Stage to a fresh BytesIO, then write back to the path.
        staged = _io.BytesIO()
        with zipfile.ZipFile(self.path.full_path(), mode="r") as src_zf:
            with zipfile.ZipFile(staged, mode="w", **zf_kwargs) as dst_zf:
                for info in src_zf.infolist():
                    if info.filename == name:
                        continue
                    dst_zf.writestr(info, src_zf.read(info.filename))
                dst_zf.writestr(name, payload_bytes)

        with open(self.path.full_path(), "wb") as fh:
            fh.write(staged.getvalue())

    def _delete_entry(self, name: str) -> None:
        """Remove ``name`` from the archive. No-op if not present."""
        if not self.path.exists() or not self._entry_exists(name):
            return
        opts = ZipOptions()
        zf_kwargs: dict[str, Any] = {"compression": opts.compression}
        if opts.compresslevel is not None:
            zf_kwargs["compresslevel"] = opts.compresslevel
        staged = _io.BytesIO()
        with zipfile.ZipFile(self.path.full_path(), mode="r") as src_zf:
            with zipfile.ZipFile(staged, mode="w", **zf_kwargs) as dst_zf:
                for info in src_zf.infolist():
                    if info.filename == name:
                        continue
                    dst_zf.writestr(info, src_zf.read(info.filename))
        with open(self.path.full_path(), "wb") as fh:
            fh.write(staged.getvalue())

    def _open_entry_io(self, name: str, *, auto_open: bool) -> "ZipEntryIO":
        existing = self._live_entries.get(name)
        if existing is not None:
            return existing
        entry_io = ZipEntryIO(parent=self, name=name, auto_open=auto_open)
        self._live_entries[name] = entry_io
        return entry_io


# ---------------------------------------------------------------------------
# ZipEntryIO
# ---------------------------------------------------------------------------


class ZipEntryIO(BytesIO):
    """A :class:`BytesIO` view of a single entry inside a :class:`ZipIO`.

    The entry's uncompressed payload IS this object's backing buffer
    — read/write/seek/truncate behave exactly as on a free-standing
    :class:`BytesIO`. The "ZipEntry" half adds three things:

    * **Parent linkage** — :attr:`parent` is the enclosing
      :class:`ZipIO`, :attr:`entry_name` is this entry's filename,
      and :attr:`zip_info` carries the central-directory metadata
      (compression / CRC / mtime / size) when the entry exists.
    * **Per-entry overrides** — :attr:`compression` /
      :attr:`compresslevel` let a caller pin an entry's framing
      independently of the parent's defaults.
    * **Parent-aware lifecycle** — ``_acquire`` pulls the entry's
      bytes from the archive's central directory so the buffer
      reads transparently; ``_release`` commits dirty bytes back
      via the parent's optimized commit path.

    Tabular I/O (Arrow IPC, Parquet, …) flows through the normal
    :class:`BytesIO` tabular contract: set a tabular ``media_type``
    on the entry (the entry's filename extension is enough), call
    :meth:`read_arrow_batches` / :meth:`write_arrow_batches`, and
    the request is redirected via :meth:`as_media` to the registered
    concrete leaf operating on this buffer.

    Notes
    -----
    Multiple :meth:`ZipIO.make_child` / :meth:`ZipIO.iter_children`
    calls for the same name on the same parent return the *same*
    :class:`ZipEntryIO` instance — two readers see each other's
    writes through one buffer. Independent :class:`ZipIO` parents
    backed by the same path do not share live-entry state.
    """

    def __init__(
        self,
        *,
        parent: ZipIO,
        name: str,
        zip_info: "zipfile.ZipInfo | None" = None,
        compression: "int | None" = None,
        compresslevel: "int | None" = None,
        auto_open: bool = True,
        media_type: Any = None,
        **kwargs: Any,
    ) -> None:
        # Default the entry's media type to the one inferred from
        # its filename extension — the common case (``batch-N.arrow``,
        # ``manifest.json``, etc.). Callers can pass ``media_type=``
        # to pin something else; ``MediaTypes.ZIP_ENTRY`` is used as
        # the opaque fallback when no extension matches.
        if media_type is None:
            media_type = MediaType.from_(name, default=MediaTypes.ZIP_ENTRY)

        # auto_open=False to super: we want our parent / dirty slots
        # in place before _acquire pulls bytes from the archive.
        super().__init__(auto_open=False, media_type=media_type, **kwargs)
        self.parent = parent
        self._entry_name = name
        self._zip_info: "zipfile.ZipInfo | None" = zip_info
        self.compression: "int | None" = compression
        self.compresslevel: "int | None" = compresslevel
        self._dirty = False
        if auto_open:
            self.open()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def entry_name(self) -> str:
        return self._entry_name

    @property
    def zip_info(self) -> "zipfile.ZipInfo | None":
        """Central-directory metadata for the entry, or ``None``.

        Populated lazily on :meth:`_acquire` from the parent's
        archive. Stays ``None`` for entries that haven't been
        committed yet (a fresh ``make_child(name)`` followed by no
        write returns ``None`` here).
        """
        return self._zip_info

    @property
    def mtime(self) -> float:
        """Entry mtime in seconds-since-epoch, or the parent's mtime.

        Falls back to the parent's mtime when the central-directory
        entry uses zip's ``(1980,1,1,0,0,0)`` "no useful timestamp"
        sentinel.
        """
        parent_mtime = float(getattr(self.parent, "mtime", 0.0) or 0.0)
        if self._zip_info is None:
            return parent_mtime
        return _zipinfo_mtime(self._zip_info, fallback=parent_mtime)

    # ------------------------------------------------------------------
    # MimeType — opaque-by-default; format leaves take over via as_media
    # ------------------------------------------------------------------

    @classmethod
    def default_mime_type(cls):
        # Per-instance default is overridden in __init__ from the
        # entry's filename. The class-level default just claims the
        # ZIP_ENTRY mime so the registry can hand back ZipEntryIO
        # for path-based dispatch (rare but consistent).
        return MimeTypes.ZIP_ENTRY

    # ------------------------------------------------------------------
    # Dirty tracking — override the two write primitives
    # ------------------------------------------------------------------
    #
    # BytesIO's contract is "if you wrote, the bytes are there" — it
    # has no dirty/clean concept. ZipEntryIO needs one because the
    # parent commit is OUR concern, not the buffer's. Per BytesIO's
    # own docstring, all public mutation flows through _write_at and
    # _set_size, so hooking those two is sufficient.
    #
    # _acquire's `replace_with_payload` (when populating from parent
    # bytes) goes through neither — it tears down + reinits. We
    # explicitly clear _dirty after `_acquire` to keep the flag
    # honest. delete() is similar: it calls replace_with_payload(b"")
    # then clears the flag, since the parent is already in sync.

    def _write_at(self, data, pos: int) -> int:
        n = super()._write_at(data, pos)
        if n > 0:
            self._dirty = True
        return n

    def _set_size(self, n: int) -> int:
        before = self._size
        result = super()._set_size(n)
        if result != before:
            self._dirty = True
        return result

    def _write_arrow_batches(
        self,
        batches: "Iterable[pa.RecordBatch]",
        options: CastOptions,
    ) -> None:
        # BytesIO routes the actual write through a registered leaf
        # (ArrowIPCIO, ParquetIO, …) wrapping our shared backing
        # buffer. The leaf's _write_at doesn't run through ours, so
        # mark dirty explicitly here so the entry commits on close.
        super()._write_arrow_batches(batches, options)
        self._dirty = True

    # ------------------------------------------------------------------
    # Lifecycle — bridge buffer <-> parent entry
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        # Pull current entry payload (if any) into self before any
        # BytesIO read/write sees the buffer.
        super()._acquire()
        # Refresh metadata cache — the central-directory entry may
        # have changed since the last open.
        self._zip_info = self.parent._entry_info(self._entry_name)
        payload = self.parent._read_entry_bytes(self._entry_name)
        if payload:
            self.replace_with_payload(payload)
        self._dirty = False

    def _commit(self) -> None:
        # Disposable.close() runs commit() before _release(), and
        # commit() clears _dirty after _commit succeeds. Routing the
        # payload-write here (rather than _release) keeps that
        # contract intact; otherwise _release would observe an
        # already-cleared dirty flag and silently skip the commit.
        with self.memoryview() as mv:
            payload = bytes(mv)
        self.parent._commit_entry_payload(
            self._entry_name,
            payload,
            compression=self.compression,
            compresslevel=self.compresslevel,
        )
        # Refresh zip_info so a re-open on this same instance sees
        # the just-committed metadata.
        self._zip_info = self.parent._entry_info(self._entry_name)

    def _release(self) -> None:
        try:
            # Drop ourselves from the parent's live map so a fresh
            # open returns a fresh handle.
            live = self.parent._live_entries
            if live.get(self._entry_name) is self:
                try:
                    del live[self._entry_name]
                except KeyError:  # pragma: no cover - benign race
                    pass
        finally:
            super()._release()

    # ------------------------------------------------------------------
    # Entry-level mutation helpers
    # ------------------------------------------------------------------

    def delete(self) -> None:
        """Remove this entry from the parent archive.

        After ``delete``, the buffer is reset to empty and the entry
        is gone from the archive — both sides consistent, so we
        clear ``_dirty``. Further writes will recreate the entry on
        the next commit.
        """
        self.parent._delete_entry(self._entry_name)
        self.replace_with_payload(b"")
        self._zip_info = None
        self._dirty = False

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        parent_name = type(self.parent).__name__ if self.parent is not None else "?"
        return (
            f"{type(self).__name__}("
            f"parent=<{parent_name}>, "
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
    as "no useful mtime" and return the caller's fallback instead.
    """
    dt = getattr(info, "date_time", None)
    if not dt or dt == (1980, 1, 1, 0, 0, 0):
        return fallback
    try:
        return time.mktime((*dt, 0, 0, -1))
    except (ValueError, OverflowError):  # pragma: no cover - degenerate input
        return fallback
