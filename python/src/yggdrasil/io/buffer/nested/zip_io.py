"""Zip archive as a :class:`BytesIO` with a children surface.

:class:`ZipIO` IS-A :class:`BytesIO` whose backing bytes are a zip
archive. On top of the normal :class:`BytesIO` byte surface (read /
write / seek / spill against the ``.zip`` file), it exposes a children
surface for the archive's entries:

- :meth:`iter_children` (and ``for entry in zio``) enumerates every
  archive entry as a :class:`ZipEntryIO`. Zip's namespace is flat, so
  path-shaped names like ``dir/a.txt`` are yielded verbatim — they do
  not auto-collapse into folder views.
- :meth:`make_child` mints a fresh entry handle bound to a name.
- :meth:`folder` returns a :class:`ZipEntryFolderIO` view scoped to a
  prefix (so consumers can treat ``dir/a.txt`` + ``dir/b.txt`` as a
  single virtual folder when they want that grouping).

The byte surface and the children surface stay consistent because
entry commits go through :meth:`_commit_entry_payload`, which mutates
the underlying ``.zip`` file directly via :mod:`zipfile`. A caller
that only ever reads the archive bytes sees a normal
:class:`BytesIO`; one that walks the children surface sees the
entries; one that does both gets a coherent view.

Children operations preserve the parent's byte cursor: iterating
children, opening / committing / deleting an entry, and clearing the
archive all snapshot ``self._pos`` and restore it on exit, so a
caller mid-stream over the archive bytes can interleave child access
without losing its place. Iterating with ``for entry in zio:``
walks children; ``next(zio)`` is rejected to avoid the inherited
``BytesIO`` line-iteration semantics quietly returning archive bytes
instead of an entry.

In-memory archives
------------------

:class:`ZipIO` supports two backings: a path-bound archive (the
``.zip`` lives on disk and the bound file IS the archive), and an
in-memory archive (no path; the parent's ``BytesIO`` buffer IS the
archive bytes). The children surface works identically in both
modes — iteration, ``make_child``, commit, and delete all go through
:meth:`_open_archive_reader` and :meth:`_replace_archive_bytes`,
which dispatch on whether ``self.path`` is bound. The path-bound
case still uses zip-native append (``mode="a"``) on cheap commits;
the in-memory case rewrites the buffer wholesale (zip's central
directory sits at EOF, so an in-memory append is just a rewrite).

Per-entry I/O — :class:`ZipEntryIO` IS-A :class:`BytesIO`
---------------------------------------------------------

A :class:`ZipEntryIO` is a :class:`BytesIO` whose backing buffer is a
single entry's uncompressed payload. Byte writes mutate only that
entry's section (its own ``_buf``), and :meth:`_commit` rewrites that
entry inside the parent archive — never the whole zip. The full
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
(``ArrowIPCIO``, ``ParquetIO``, …) wrapping the entry's buffer.

Virtual folder entries — :class:`ZipEntryFolderIO`
--------------------------------------------------

Zip entries are flat (one path per entry), but their names commonly
embed a path-like structure (``dir/a.txt``, ``dir/sub/b.txt``).
:class:`ZipEntryFolderIO` is a :class:`TabularIO` that views a name
prefix as a folder: its children are the entries under that prefix,
yielded as :class:`ZipEntryIO` (direct files) or
:class:`ZipEntryFolderIO` (sub-prefixes). The class has no byte
buffer — it's a pure children-surface view; storage stays in the
parent :class:`ZipIO`.

Dirty tracking
--------------

:class:`BytesIO` itself has no dirty/clean concept — its contract is
"you wrote bytes, those bytes are there." But :class:`ZipEntryIO`
needs to know whether to bother committing back to the parent, so it
tracks ``_dirty`` itself by overriding the two write primitives
(``_write_at`` and ``_set_size``). ``_acquire`` clears the flag after
pulling the entry's bytes in (acquire is not a user-driven mutation);
``_commit`` runs only when the dirty flag is set.

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

import contextlib
import dataclasses
import io as _io
import time
import zipfile
from typing import Any, ClassVar, Iterable, Iterator
from weakref import WeakValueDictionary

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.enums import MediaType, MediaTypes, MimeTypes, Mode
from .base import NestedOptions

__all__ = [
    "ZipIO",
    "ZipOptions",
    "ZipEntryIO",
    "ZipEntryOptions",
    "ZipEntryFolderIO",
]


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


class ZipIO(BytesIO):
    """A :class:`BytesIO` whose bytes form a zip archive.

    The byte buffer (or path-bound spill file) IS the ``.zip`` file
    on disk. On top of the byte surface, :class:`ZipIO` exposes a
    children surface — its entries, yielded as :class:`ZipEntryIO`
    (file) or :class:`ZipEntryFolderIO` (virtual folder).

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
        parent: "TabularIO | None" = None,
        auto_open: bool | None = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(data, path=path, auto_open=auto_open, **kwargs)
        # Folder-tree linkage for archives nested inside other
        # children-bearing IOs (FolderIO of zip files, etc.). ``None``
        # for the top-level handle.
        self.parent = parent
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

    @contextlib.contextmanager
    def _preserve_cursor(self) -> "Iterator[None]":
        """Snapshot ``self._pos`` and restore it on exit.

        Only needed around write-back operations that swap the
        archive's bytes (clear, per-entry commit, per-entry delete) —
        :meth:`_replace_archive_bytes` calls
        :meth:`replace_with_payload`, which resets ``_pos`` to 0.
        Read-only paths (``_open_archive_reader``, name listing,
        per-entry info / payload reads) hand the buffer to
        :mod:`zipfile` through a :meth:`BytesIO.view`, so the parent
        cursor is naturally untouched and doesn't need this guard.
        """
        saved = self._pos
        try:
            yield
        finally:
            # Re-clamp in case the archive shrank under us (in-memory
            # rewrite to fewer bytes); a cursor past EOF is nonsense.
            self._pos = min(saved, self._size)

    # ------------------------------------------------------------------
    # Archive backing — path or in-memory bytes, behind one interface
    # ------------------------------------------------------------------

    def _archive_is_present(self) -> bool:
        """True iff this :class:`ZipIO` already has archive bytes.

        Path-bound: the bound file exists. In-memory: the buffer is
        non-empty. ``False`` for the "fresh handle, nothing written
        yet" case in either mode — callers treat that as "open this
        as a writable empty archive".
        """
        if self.path is not None:
            return self.path.exists()
        return self._size > 0

    @contextlib.contextmanager
    def _open_archive_reader(self) -> "Iterator[zipfile.ZipFile]":
        """Yield a ``zipfile.ZipFile`` reader over the current archive.

        Path-bound: open the bound file. In-memory: hand a
        :meth:`BytesIO.view` of the buffer to ``zipfile.ZipFile`` —
        the view has its own cursor, so seeks/reads from
        :mod:`zipfile` (central directory probe, per-entry reads)
        leave the parent's cursor untouched and avoid copying the
        archive bytes into a fresh stdlib ``BytesIO``. Caller is
        responsible for handling :class:`zipfile.BadZipFile` and
        :class:`FileNotFoundError`.
        """
        if self.path is not None:
            with zipfile.ZipFile(self.path.full_path(), mode="r") as zf:
                yield zf
            return
        if self._size == 0:
            # Empty in-memory archive — surface the same "missing"
            # signal as a non-existent path file.
            raise FileNotFoundError("in-memory ZipIO is empty")
        with self.view(pos=0) as v, zipfile.ZipFile(v, mode="r") as zf:
            yield zf

    def _replace_archive_bytes(self, payload: bytes) -> None:
        """Write *payload* as the archive's full backing bytes.

        Path-bound: overwrite the bound file. In-memory: replace the
        parent's buffer via :meth:`replace_with_payload` — which
        resets ``_pos`` to 0, so callers that care about the byte
        cursor must wrap this in :meth:`_preserve_cursor`.
        """
        if self.path is not None:
            with open(self.path.full_path(), "wb") as fh:
                fh.write(payload)
            return
        # In-memory: wholesale buffer swap. ``replace_with_payload``
        # resets ``_pos`` to 0; the surrounding ``_preserve_cursor``
        # restores the caller's position (clamped to the new size).
        self.replace_with_payload(payload)

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

        Read-only — :meth:`_list_entry_names` and the entry's lazy
        ``_acquire`` go through :meth:`_open_archive_reader`, which
        feeds :mod:`zipfile` a :meth:`BytesIO.view` for in-memory
        archives, so neither building nor walking the iterator
        touches the parent's byte cursor.
        """
        if not self._archive_is_present():
            return
        names = [
            name
            for name in self._list_entry_names()
            if not self._is_ignored_name(name)
            and options.matches_name(name)
        ]
        for name in names:
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
        children-surface contract but is ignored — entry framing is
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

    def folder(self, prefix: str) -> "ZipEntryFolderIO":
        """View a name prefix inside the archive as a virtual folder.

        Returns a :class:`ZipEntryFolderIO` whose children are the
        entries under ``prefix``. Useful for archives that embed a
        directory layout in entry names (``dir/a.txt``,
        ``dir/sub/b.txt``).
        """
        return ZipEntryFolderIO(parent=self, prefix=prefix)

    def __iter__(self) -> "Iterator[ZipEntryIO]":
        return self._iter_children(self._default_options())

    def __next__(self):
        # BytesIO inherits a ``__next__`` that returns ``readline()``
        # output. ``ZipIO.__iter__`` yields children, not lines, so a
        # bare ``next(zio)`` would silently switch surfaces and hand
        # back archive bytes instead of an entry. Reject it loudly.
        raise TypeError(
            f"{type(self).__name__} is not directly iterable with next(); "
            "use 'for entry in zio:' or zio.iter_children() to walk the "
            "archive's children, or zio.readline() / zio.read() for "
            "byte-level access."
        )

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        if self._is_ignored_name(name):
            return False
        return self._entry_exists(name)

    def list_entries(self) -> list[str]:
        """Return entry names currently in the archive, sorted.

        Hidden entries (name starting with ``.``) are filtered out to
        match :meth:`_iter_children` / ``__contains__``.
        """
        return sorted(
            name
            for name in self._list_entry_names()
            if not self._is_ignored_name(name)
        )

    def has_children(self) -> bool:
        """True iff the archive holds at least one non-ignored entry."""
        if not self._archive_is_present():
            return False
        try:
            with self._open_archive_reader() as zf:
                for info in zf.infolist():
                    if not self._is_ignored_name(info.filename):
                        return True
                return False
        except (zipfile.BadZipFile, FileNotFoundError):
            return False

    # ==================================================================
    # Empty / clear (children-flavored — overrides BytesIO's byte view)
    # ==================================================================

    def is_empty(self) -> bool:
        """True if the archive is missing or has no non-ignored entries."""
        return not self.has_children()

    def _is_ignored_name(self, name: str) -> bool:
        """Hide hidden entries (name starts with ``.``) from enumeration."""
        return name.startswith(".")

    def _clear_children(self) -> None:
        """Reset the archive to empty by writing a fresh empty zip.

        Works for both path-bound and in-memory archives: stages an
        empty central directory and pushes the resulting bytes
        through :meth:`_replace_archive_bytes`.
        """
        if not self._archive_is_present():
            return
        with self._preserve_cursor():
            staged = _io.BytesIO()
            with zipfile.ZipFile(
                staged, mode="w", compression=zipfile.ZIP_STORED
            ):
                pass
            self._replace_archive_bytes(staged.getvalue())

    # ==================================================================
    # Mode resolution — folder-flavored, overrides BytesIO's byte view
    # ==================================================================

    def _resolve_save_mode(self, mode: Any) -> Mode:
        m = Mode.from_(mode, default=Mode.AUTO)

        if m in (Mode.AUTO, Mode.OVERWRITE, Mode.TRUNCATE):
            return Mode.OVERWRITE

        if m is Mode.APPEND:
            return Mode.APPEND

        if m is Mode.IGNORE:
            return Mode.IGNORE if not self.is_empty() else Mode.OVERWRITE

        if m is Mode.ERROR_IF_EXISTS:
            if not self.is_empty():
                raise FileExistsError(
                    f"{type(self).__name__} write with "
                    f"Mode.ERROR_IF_EXISTS but archive is non-empty. "
                    f"Path: {self.path!r}"
                )
            return Mode.OVERWRITE

        return m

    # ==================================================================
    # Read derivation — chain children's batches
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: ZipOptions,
    ) -> "Iterator[pa.RecordBatch]":
        """Chain :meth:`iter_children` into a single Arrow batch stream.

        Each entry is opened in turn and its batches are forwarded.
        Non-tabular entries (no recognized media type) are skipped.
        """
        if self.cached:
            yield from self._read_arrow_batches_from_cache(options)
            return

        for child in self._iter_children(options):
            with child:
                try:
                    yield from child.read_arrow_batches(options=options)
                except NotImplementedError:
                    # Non-tabular entry — skip; bytes still reachable
                    # via iter_children().
                    continue

    def _collect_schema(self, options: ZipOptions) -> Schema:
        """Merge per-entry schemas into a single archive schema."""
        merged: Schema | None = None
        for child in self._iter_children(options):
            with child:
                try:
                    schema = child.collect_schema(options=options)
                except Exception:
                    continue
            if merged is None:
                merged = schema
            else:
                merged = merged.merge_with(schema, inplace=True)
        return merged if merged is not None else Schema.empty()

    # ==================================================================
    # Write derivation — entry-atomic commits, no on-disk staging
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ZipOptions,
    ) -> None:
        """Drain *batches* into one or more zip entries.

        Mode dispatch: OVERWRITE clears the archive first, APPEND
        adds new entries beside existing ones, IGNORE skips when the
        archive is non-empty.
        """
        mode = self._resolve_save_mode(options.mode)

        if mode is Mode.IGNORE:
            return

        if mode is Mode.UPSERT:
            self._arrow_upsert_via_rewrite(batches, options)
            return

        if mode is Mode.OVERWRITE:
            self._clear_children()

        self._drain_into_children(batches, options)

    def _drain_into_children(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ZipOptions,
    ) -> None:
        """Drain *batches* into one or more child entries."""
        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            return

        media_type = options.child_media_type or self._default_child_media_type()
        row_threshold = options.child_row_size or 0
        byte_threshold = options.child_byte_size or 0

        if row_threshold <= 0 and byte_threshold <= 0:
            self._write_one_child(
                _chain_first(first, batch_iter),
                media_type=media_type,
                options=options,
            )
            return

        for chunk in _split_batches(
            _chain_first(first, batch_iter),
            row_threshold=row_threshold,
            byte_threshold=byte_threshold,
        ):
            self._write_one_child(
                chunk,
                media_type=media_type,
                options=options,
            )

    def _write_one_child(
        self,
        batches: "Iterable[pa.RecordBatch]",
        *,
        media_type: Any,
        options: ZipOptions,
    ) -> None:
        """Drain *batches* into one fresh zip entry.

        Skips the staging-file dance NestedIO uses for folder writes:
        a zip entry is committed atomically to the central directory
        when the :class:`ZipEntryIO` releases, so there's nothing to
        rename. On exception we best-effort delete the half-written
        entry.
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
        if self._archive_is_present():
            try:
                with self._open_archive_reader() as zf:
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
    # Internal helpers
    # ==================================================================

    def _attach(self, child: Any) -> Any:
        """Stamp ``child.parent = self`` and return *child*."""
        try:
            child.parent = self
        except (AttributeError, TypeError):
            pass
        return child

    def _default_options(self) -> ZipOptions:
        return self.options_class()()

    def _arrow_upsert_via_rewrite(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ZipOptions,
    ) -> None:
        """UPSERT for an archive: append batches as new entries.

        Zip archives don't have a per-row primary key, so UPSERT
        degrades to APPEND at the entry level — mirrors what
        :class:`NestedIO` did for folders without a key column.
        """
        self._drain_into_children(batches, options)

    # ==================================================================
    # Entry-namespace internals (used by ZipEntryIO and ZipEntryFolderIO)
    # ==================================================================

    def _list_entry_names(self) -> list[str]:
        if not self._archive_is_present():
            return []
        try:
            with self._open_archive_reader() as zf:
                return list(zf.namelist())
        except (zipfile.BadZipFile, FileNotFoundError):
            return []

    def _entry_info(self, name: str) -> "zipfile.ZipInfo | None":
        if not self._archive_is_present():
            return None
        try:
            with self._open_archive_reader() as zf:
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
        if not self._archive_is_present():
            return b""
        try:
            with self._open_archive_reader() as zf:
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
        commit. Only this entry's section of the archive is rewritten;
        every other entry is preserved bit-for-bit.

        * **Cheap append (path-bound only)** — entry doesn't exist (or
          archive is missing). Open the archive in ``"a"`` mode (or
          ``"w"`` if missing) and append the entry to the central
          directory. No rewrite.
        * **Rewrite swap** — entry exists, or the archive is in
          memory. Stream all other entries through a fresh in-memory
          archive with ``payload`` substituted for (or appended to)
          ``name``, then push the resulting bytes back via
          :meth:`_replace_archive_bytes`. Required because zip's
          central directory sits at EOF and per-entry sizes change
          on rewrite — and because in-memory archives have no
          equivalent of zip-native append.

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

        with self._preserve_cursor():
            # Cheap zip-native append: only available when the archive
            # backing is a real file and the entry doesn't exist yet.
            if self.path is not None:
                archive_present = self.path.exists()
                if not archive_present or not self._entry_exists(name):
                    mode = "a" if archive_present else "w"
                    with zipfile.ZipFile(
                        self.path.full_path(), mode=mode, **zf_kwargs
                    ) as zf:
                        zf.writestr(name, payload_bytes)
                    return

            # Rewrite-swap path. Covers:
            #   * path-bound + existing entry (full rewrite),
            #   * in-memory + any state (append-as-rewrite or swap).
            staged = _io.BytesIO()
            with zipfile.ZipFile(staged, mode="w", **zf_kwargs) as dst_zf:
                if self._archive_is_present():
                    try:
                        with self._open_archive_reader() as src_zf:
                            for info in src_zf.infolist():
                                if info.filename == name:
                                    continue
                                dst_zf.writestr(
                                    info, src_zf.read(info.filename)
                                )
                    except (zipfile.BadZipFile, FileNotFoundError):
                        # Treat unreadable backing as "fresh archive"
                        # and let the new entry seed it.
                        pass
                dst_zf.writestr(name, payload_bytes)

            self._replace_archive_bytes(staged.getvalue())

    def _delete_entry(self, name: str) -> None:
        """Remove ``name`` from the archive. No-op if not present."""
        if not self._archive_is_present() or not self._entry_exists(name):
            return
        opts = ZipOptions()
        zf_kwargs: dict[str, Any] = {"compression": opts.compression}
        if opts.compresslevel is not None:
            zf_kwargs["compresslevel"] = opts.compresslevel
        with self._preserve_cursor():
            staged = _io.BytesIO()
            with self._open_archive_reader() as src_zf:
                with zipfile.ZipFile(staged, mode="w", **zf_kwargs) as dst_zf:
                    for info in src_zf.infolist():
                        if info.filename == name:
                            continue
                        dst_zf.writestr(info, src_zf.read(info.filename))
            self._replace_archive_bytes(staged.getvalue())

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
    :class:`BytesIO`, but mutations only ever touch this entry's
    section. The "ZipEntry" half adds three things:

    * **Parent linkage** — :attr:`parent` is the enclosing
      :class:`ZipIO`, :attr:`entry_name` is this entry's filename,
      and :attr:`zip_info` carries the central-directory metadata
      (compression / CRC / mtime / size) when the entry exists.
    * **Per-entry overrides** — :attr:`compression` /
      :attr:`compresslevel` let a caller pin an entry's framing
      independently of the parent's defaults.
    * **Parent-aware lifecycle** — ``_acquire`` pulls the entry's
      bytes from the archive's central directory so the buffer
      reads transparently; ``_commit`` writes dirty bytes back via
      the parent's optimized commit path (which only rewrites this
      one entry — never the whole archive).

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
    # has_children — entry-as-file is a leaf, never has children
    # ------------------------------------------------------------------

    def has_children(self) -> bool:
        return False

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
# ZipEntryFolderIO
# ---------------------------------------------------------------------------


class ZipEntryFolderIO(TabularIO[ZipOptions]):
    """A virtual sub-folder view into a :class:`ZipIO`.

    Zip entries live in a flat namespace, but their names commonly
    embed a path-like structure (``dir/a.txt``, ``dir/sub/b.txt``).
    :class:`ZipEntryFolderIO` exposes one prefix as a folder: its
    children are the entries that begin with ``prefix``, yielded as
    :class:`ZipEntryIO` for direct files and as
    :class:`ZipEntryFolderIO` for sub-prefixes.

    No byte buffer of its own — storage is the parent :class:`ZipIO`.
    Reads chain children's batches; writes mint new entries under the
    prefix via the parent's commit path.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_mime_type(cls):
        # No public mime — this is a structural view, not a wire format.
        return None

    @classmethod
    def options_class(cls):
        return ZipOptions

    def __init__(
        self,
        *,
        parent: ZipIO,
        prefix: str,
    ) -> None:
        super().__init__()
        self.parent = parent
        self._prefix = _normalize_folder_prefix(prefix)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def prefix(self) -> str:
        """The entry-name prefix (always ends with ``/`` or is empty)."""
        return self._prefix

    @property
    def path(self):
        """Pass-through to the parent archive's path.

        :class:`ZipEntryFolderIO` doesn't have its own on-disk path —
        it's a logical slice of the parent. Surfacing the parent's
        path keeps tooling that probes ``io.path`` (logging, error
        messages) honest about where the bytes actually live.
        """
        return self.parent.path

    @property
    def cached(self) -> bool:
        return self._arrow_table is not None or self._spark_frame is not None

    def unpersist(self) -> None:
        self._arrow_table = None
        self._spark_frame = None

    def persist(self, engine="auto", *, data=None) -> "ZipEntryFolderIO":
        if self.cached:
            return self
        if data is None:
            self._arrow_table = self.read_arrow_table()
        else:
            from yggdrasil.arrow.cast import any_to_arrow_table
            self._arrow_table = any_to_arrow_table(data)
        return self

    # ------------------------------------------------------------------
    # Children — slice the parent's central directory by prefix
    # ------------------------------------------------------------------

    def _direct_children_names(self) -> "Iterator[tuple[str, bool]]":
        """Yield ``(name, is_folder)`` for direct children under prefix.

        Direct == one path segment past the prefix. Sub-folders
        collapse into a single entry (``dir/sub/`` rather than every
        individual ``dir/sub/*`` leaf).
        """
        seen_folders: set[str] = set()
        for full in self.parent._list_entry_names():
            if not full.startswith(self._prefix):
                continue
            tail = full[len(self._prefix):]
            if not tail or tail.startswith("."):
                continue
            slash = tail.find("/")
            if slash == -1:
                yield full, False
                continue
            sub = tail[: slash + 1]
            full_sub_prefix = self._prefix + sub
            if full_sub_prefix in seen_folders:
                continue
            seen_folders.add(full_sub_prefix)
            yield full_sub_prefix, True

    def _iter_children(
        self,
        options: ZipOptions,
    ) -> "Iterator[TabularIO]":
        for name, is_folder in self._direct_children_names():
            if is_folder:
                yield ZipEntryFolderIO(parent=self.parent, prefix=name)
            else:
                yield self.parent._open_entry_io(name, auto_open=False)

    def has_children(self) -> bool:
        return next(self._direct_children_names(), None) is not None

    def list_entries(self, *, recursive: bool = True) -> list[str]:
        """List entry names under this folder.

        ``recursive=True`` (default) walks every descendant entry
        (still scoped to the prefix); ``recursive=False`` only
        yields names one segment past the prefix.
        """
        if recursive:
            return sorted(
                full
                for full in self.parent._list_entry_names()
                if full.startswith(self._prefix) and full != self._prefix
            )
        return sorted(name for name, _ in self._direct_children_names())

    def __iter__(self) -> "Iterator[TabularIO]":
        return self._iter_children(self._has_children_options())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        full = name if name.startswith(self._prefix) else self._prefix + name
        return self.parent._entry_exists(full)

    # ------------------------------------------------------------------
    # Mutators — minted via the parent's commit path
    # ------------------------------------------------------------------

    def make_child(
        self,
        name: str,
        *,
        media_type: Any = None,
    ) -> "ZipEntryIO":
        """Mint a child entry under this folder's prefix."""
        if "\\" in name:
            raise ValueError(
                f"Entry name must not contain backslashes; got {name!r}. "
                "Use forward slashes."
            )
        full = name if name.startswith(self._prefix) else self._prefix + name
        return self.parent.make_child(full, media_type=media_type)

    def folder(self, sub_prefix: str) -> "ZipEntryFolderIO":
        """View a deeper sub-prefix as another folder."""
        normalized = _normalize_folder_prefix(sub_prefix)
        if not normalized.startswith(self._prefix):
            normalized = self._prefix + normalized
        return ZipEntryFolderIO(parent=self.parent, prefix=normalized)

    # ------------------------------------------------------------------
    # Read/write — chain children
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self,
        options: ZipOptions,
    ) -> "Iterator[pa.RecordBatch]":
        if self.cached:
            yield from self._read_arrow_batches_from_cache(options)
            return
        for child in self._iter_children(options):
            with child:
                try:
                    yield from child.read_arrow_batches(options=options)
                except NotImplementedError:
                    continue

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ZipOptions,
    ) -> None:
        """Drain *batches* into one or more entries under prefix."""
        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            return

        media_type = options.child_media_type or MimeTypes.ARROW_IPC
        row_threshold = options.child_row_size or 0
        byte_threshold = options.child_byte_size or 0

        chunks: Iterable[Iterable[pa.RecordBatch]]
        if row_threshold <= 0 and byte_threshold <= 0:
            chunks = [_chain_first(first, batch_iter)]
        else:
            chunks = _split_batches(
                _chain_first(first, batch_iter),
                row_threshold=row_threshold,
                byte_threshold=byte_threshold,
            )

        for chunk in chunks:
            name = self.parent._next_child_name(
                media_type=media_type, options=options
            )
            full = name if name.startswith(self._prefix) else self._prefix + name
            child = self.parent.make_child(full, media_type=media_type)
            try:
                with child:
                    child.write_arrow_batches(chunk, options=options)
            except Exception:
                try:
                    self.parent._delete_entry(full)
                except Exception:
                    pass
                raise

    def is_empty(self) -> bool:
        return not self.has_children()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_folder_prefix(prefix: str) -> str:
    """Coerce *prefix* to the central-directory slice form.

    Empty / ``"/"`` collapse to the empty string (the whole archive).
    Other inputs are stripped of leading slashes and given a trailing
    ``/`` so the prefix match is folder-shaped, not substring-shaped.
    """
    if not prefix or prefix in ("/", "."):
        return ""
    cleaned = prefix.lstrip("/")
    if not cleaned:
        return ""
    if not cleaned.endswith("/"):
        cleaned = cleaned + "/"
    return cleaned


def _chain_first(
    first: pa.RecordBatch,
    rest: "Iterator[pa.RecordBatch]",
) -> "Iterator[pa.RecordBatch]":
    """Yield *first*, then every batch from *rest*."""
    yield first
    yield from rest


def _split_batches(
    batches: "Iterator[pa.RecordBatch]",
    *,
    row_threshold: int,
    byte_threshold: int,
) -> "Iterator[Iterator[pa.RecordBatch]]":
    """Split a batch iterator into chunks by row / byte threshold.

    ``row_threshold`` wins when both are set. With a row threshold,
    incoming batches are sliced so each emitted chunk holds exactly
    ``row_threshold`` rows (the final chunk may be shorter).
    """

    def _size_bytes(batch: pa.RecordBatch) -> int:
        try:
            return int(batch.nbytes)
        except Exception:
            return 0

    if row_threshold > 0:
        pending: list[pa.RecordBatch] = []
        pending_rows = 0
        for batch in batches:
            offset = 0
            remaining = batch.num_rows
            while remaining > 0:
                take = min(remaining, row_threshold - pending_rows)
                slice_ = batch.slice(offset, take)
                pending.append(slice_)
                pending_rows += take
                offset += take
                remaining -= take
                if pending_rows >= row_threshold:
                    yield iter(pending)
                    pending = []
                    pending_rows = 0
        if pending:
            yield iter(pending)
        return

    pending = []
    nbytes = 0
    for batch in batches:
        pending.append(batch)
        if byte_threshold > 0:
            nbytes += _size_bytes(batch)
        if 0 < byte_threshold <= nbytes:
            yield iter(pending)
            pending = []
            nbytes = 0
    if pending:
        yield iter(pending)


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
