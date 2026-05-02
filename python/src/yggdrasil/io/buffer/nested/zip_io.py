"""Zip archive as a :class:`NestedIO` with single-file storage.

:class:`ZipIO` models a single zip archive on disk as a folder-shaped
:class:`NestedIO`: its children are zip entries (each yielded as a
:class:`ZipEntryIO`), reads enumerate the archive's central directory,
writes mint new entries by name. Unlike :class:`FolderIO` whose
backing store is a directory tree, :class:`ZipIO`'s backing store is
one zip file — the children-as-IOs surface matches :class:`NestedIO`'s
contract, so the same read/write derivations flow through it without
any :class:`Fragment` indirection.

The contract
------------

A :class:`ZipIO` holds a single :class:`Path` (the archive file).
:meth:`iter_children` opens the archive and yields one
:class:`ZipEntryIO` per entry; :meth:`make_child` mints a fresh entry
handle bound to a name. Each yielded child carries ``parent = self``
so consumers can walk back up to the archive.

Per-entry I/O
-------------

A :class:`ZipEntryIO` IS-A :class:`PrimitiveIO` whose backing buffer
is the entry's payload bytes — it supports the full :class:`BytesIO`
surface (read, write, seek, truncate, spill, ``memoryview``). On
commit, the parent rewrites the archive with the entry's new bytes
substituted in. Multiple live handles to the same entry name share
buffer state through a per-parent live map keyed by name.

Dirty tracking
--------------

:class:`BytesIO` itself has no dirty/clean concept — its contract is
"you wrote bytes, those bytes are there." But :class:`ZipEntryIO`
needs to know whether to bother committing back to the parent, so it
tracks ``_dirty`` itself by overriding the two write primitives
(``_write_at`` and ``_set_size``). ``_acquire`` clears the flag after
pulling the entry's bytes in (acquire is not a user-driven mutation);
``_release`` checks the flag and runs the commit only when it's True.
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
from yggdrasil.io.buffer.primitive import PrimitiveIO
from yggdrasil.io.enums import MediaType, MediaTypes, MimeTypes, Mode
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
    ) -> None:
        """Persist ``payload`` as the contents of entry ``name``.

        Implements per-entry write-back from a :class:`ZipEntryIO`
        commit. Because zip's central directory is at the end of the
        file, replacing one entry requires rewriting the archive; we
        do so by streaming all other entries through a fresh
        :class:`zipfile.ZipFile` and substituting ``payload`` for
        ``name``. If ``name`` does not yet exist, it is appended
        (cheap) into the existing or fresh archive.
        """
        opts = options if options is not None else ZipOptions()
        payload_bytes = bytes(payload)
        zf_kwargs: dict[str, Any] = {"compression": opts.compression}
        if opts.compresslevel is not None:
            zf_kwargs["compresslevel"] = opts.compresslevel

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


class ZipEntryIO(PrimitiveIO):
    """A :class:`PrimitiveIO` view of a single entry inside a :class:`ZipIO`.

    The entry's uncompressed payload IS this object's backing
    :class:`BytesIO` — read/write/seek/truncate behave exactly as on
    a free-standing buffer. Lifecycle:

    * ``_acquire`` pulls the entry's payload out of the parent and
      installs it as this buffer's bytes (or starts empty if the
      entry does not yet exist), then clears the dirty flag.
    * Any user-driven mutation (write, truncate, etc.) flows through
      ``_write_at`` / ``_set_size`` and flips ``_dirty`` to True.
    * ``_release`` writes the buffer's payload back into the parent
      via :meth:`ZipIO._commit_entry_payload` if ``_dirty`` is True.

    Tabular reads/writes use Arrow IPC streaming format — one logical
    stream per entry — so the entry can hold many record batches with
    OVERWRITE and APPEND save modes for in-entry concatenation.

    Notes
    -----
    Multiple :meth:`ZipIO.make_child` / :meth:`ZipIO.iter_children`
    calls for the same name on the same parent return the *same*
    :class:`ZipEntryIO` instance (POSIX-shared-fd semantics): two
    readers see each other's writes through one buffer. Independent
    :class:`ZipIO` parents backed by the same path do not share
    live-entry state.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    def __init__(
        self,
        *,
        parent: ZipIO,
        name: str,
        auto_open: bool = True,
        **kwargs: Any,
    ) -> None:
        # Forward kwargs to BytesIO/PrimitiveIO so spill_bytes,
        # spill_ttl, etc. flow through normally. Pass auto_open=False
        # to super so our slots are in place before _acquire runs.
        super().__init__(auto_open=False, **kwargs)
        self.parent = parent
        self._entry_name = name
        self._dirty = False
        if auto_open:
            self.open()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def entry_name(self) -> str:
        return self._entry_name

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

    # ------------------------------------------------------------------
    # Lifecycle — bridge buffer <-> parent entry
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        # Pull current entry payload (if any) into self before any
        # BytesIO read/write sees the buffer.
        super()._acquire()
        payload = self.parent._read_entry_bytes(self._entry_name)
        if payload:
            self.replace_with_payload(payload)
        self._dirty = False

    def _release(self) -> None:
        try:
            if self._dirty:
                with self.memoryview() as mv:
                    payload = bytes(mv)
                self.parent._commit_entry_payload(self._entry_name, payload)
                self._dirty = False
        finally:
            # Drop ourselves from the parent's live map so a fresh
            # open returns a fresh handle.
            live = self.parent._live_entries
            if live.get(self._entry_name) is self:
                try:
                    del live[self._entry_name]
                except KeyError:  # pragma: no cover - benign race
                    pass
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

        mt = MediaType.from_(self._entry_name, default=MediaTypes.OCTET_STREAM)
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
