"""Zip-archive Tabular leaf with lazy per-entry I/O.

:class:`ZipFile` IS-A :class:`IO` whose backing bytes are a zip
archive. It exposes two surfaces:

1. **Byte surface** — inherited from :class:`IO`. Read / write /
   seek the raw archive bytes (useful for "open zip, drive
   :mod:`zipfile` yourself" flows).
2. **Children surface** — :meth:`iter_children` walks every entry as
   a :class:`ZipEntryFile`. The entries are **lazy**: their bytes are
   fetched from the parent archive on first read and cached after.
   Iterating doesn't decompress every entry up front.

Lazy children
-------------

A :class:`ZipEntryFile` is an :class:`IO` over a :class:`Memory`
holder, but the holder's payload starts empty and is materialized
on first access through :meth:`ZipEntryFile._materialize`. The first
``read`` / ``size`` / ``seek_end`` / ``to_bytes`` / Tabular hook
triggers exactly one ``zipfile.ZipFile(parent_view).read(name)``
call; subsequent accesses hit the cached :class:`Memory` directly.

Tabular dispatch on a child uses the entry name's extension to pick
the right :class:`Tabular` leaf (parquet / csv / arrow / ndjson),
so ``zio["data.parquet"].read_arrow_table()`` works end-to-end
without the caller knowing the inner format.

Writing
-------

Two surfaces:

- :meth:`ZipFile.entry` returns a :class:`ZipEntryFile` per name —
  the per-entry handle is both readable and writable.
  ``with z.entry("data.bin").open("wb") as f: f.write(...)`` stages
  bytes in a private buffer and commits on clean exit; commits
  against different entry names can run in parallel
  (per-archive lock holds only for the brief central-directory
  rewrite, survivors stream chunk-by-chunk).

- :meth:`_write_arrow_batches` (the whole-archive Tabular hook)
  packs the incoming batches into one entry whose name comes from
  ``options.entry_name``. ``OVERWRITE`` writes a fresh archive
  containing the single entry; ``APPEND`` keeps existing entries
  whose names differ from the new one and adds the new entry;
  ``IGNORE`` / ``ERROR_IF_EXISTS`` guard non-empty archives.

Convenience helper :meth:`write_entries` packs arbitrary
``(name, bytes)`` pairs into a fresh archive.
"""

from __future__ import annotations

import dataclasses
import threading
import zipfile
from typing import ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.enums import MimeTypes, Mode, ModeLike
from yggdrasil.enums.media_type import MediaType
from yggdrasil.enums.mime_type import MimeType
from yggdrasil.io.holder import IO
from yggdrasil.io.holder import Holder
from yggdrasil.path.memory import Memory
from yggdrasil.io.tabular.base import Tabular


__all__ = ["ZipFile", "ZipOptions", "ZipEntryFile"]


def _registered_tabular_extensions() -> "list[str]":
    """Return a sorted list of extensions a zip entry name can carry
    that will dispatch to a registered format leaf.

    Used by the read- and write-side errors so the message points at
    actual valid suffixes for the current registry state instead of a
    hard-coded sample.
    """
    out: "set[str]" = set()
    for name in Holder.registered_classes():
        mt = MediaType.from_(name, default=None)
        if mt is None:
            continue
        out.update(MimeType.extensions_for(mt.mime_type))
    return sorted(out)


def _describe_entry_resolution_failure(entry_name: str) -> str:
    """Return a one-line reason *entry_name* didn't resolve to a leaf.

    Distinguishes the three failure shapes so the caller's error
    points at the right thing: (1) name has no recognizable
    extension, (2) extension maps to a non-tabular mime, (3)
    extension maps to a tabular mime with no registered leaf.
    """
    try:
        mt = MediaType.from_(entry_name, default=None)
    except Exception:
        mt = None
    if mt is None:
        return (
            f"{entry_name!r}: no MediaType could be inferred from the "
            "entry name (no recognized extension)"
        )
    try:
        cls = Holder.class_for_media_type(mt, default=None)
    except Exception:
        cls = None
    if cls is None:
        return (
            f"{entry_name!r}: MediaType {mt.mime_type.value!r} has no "
            "registered Holder leaf"
        )
    return (
        f"{entry_name!r}: resolved to {cls.__name__} — unexpected, "
        "should have dispatched"
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ZipOptions(CastOptions):
    """:class:`CastOptions` extended with zip-archive knobs."""

    #: Name (and implied format) of the entry written by
    #: :meth:`ZipFile._write_arrow_batches`. The extension picks the
    #: inner Tabular leaf.
    entry_name: str = "data.parquet"
    compression: int = zipfile.ZIP_DEFLATED
    compresslevel: "int | None" = None


# ---------------------------------------------------------------------------
# ZipEntryFile — lazy per-entry IO
# ---------------------------------------------------------------------------


class ZipEntryFile(IO):
    """:class:`IO` over a single zip entry — readable AND writable.

    Read side
        Payload is fetched from the parent archive on first access
        and cached in the inner :class:`Memory` holder. Reading the
        archive's directory
        (:meth:`ZipFile.list_entries` / :meth:`ZipFile.iter_children`)
        is a fixed-cost walk; per-entry decompression only happens for
        the entries the caller actually touches.

    Write side
        Bytes written into the inner :class:`Memory` holder get
        committed to the parent archive on close
        (``with entry.open("wb") as f: f.write(...)``) or after a
        Tabular call (:meth:`write_arrow_batches`). The commit takes a
        per-archive :class:`threading.Lock`, so concurrent writers
        against different entry names are safe.

    The ``mode`` slot, set by :meth:`ZipFile.entry`, controls how a
    commit handles a pre-existing entry of the same name: ``AUTO`` /
    ``OVERWRITE`` replace, ``ERROR_IF_EXISTS`` raises, ``IGNORE``
    skips, ``APPEND`` concatenates the new bytes onto the existing
    payload.

    Tabular hooks dispatch on the entry name's extension via
    :meth:`Holder.class_for_media_type`, so a parquet entry's
    :meth:`read_arrow_batches` runs the parquet reader against the
    entry bytes — no extra wiring.
    """

    __slots__ = (
        "entry_name",
        "_zip_parent",
        "_materialized",
        "_uncompressed_size",
        "_zip_info",
        "_entry_mode",
    )

    def __init__(
        self,
        *,
        entry_name: str,
        zip_parent: "ZipFile",
        zip_info: "zipfile.ZipInfo | None" = None,
        mode: ModeLike = Mode.AUTO,
        **kwargs,
    ) -> None:
        # Empty Memory holder; bytes land here on materialize or write.
        super().__init__(holder=Memory(), owns_holder=True, **kwargs)
        self.entry_name: str = entry_name
        self._zip_parent: "ZipFile" = zip_parent
        self._zip_info: "zipfile.ZipInfo | None" = zip_info
        # ``None`` until first directory probe; we keep the hint so
        # ``size`` can answer without decompressing.
        self._uncompressed_size: "int | None" = (
            zip_info.file_size if zip_info is not None else None
        )
        # ``zip_info=None`` AND no parent bytes → treat as empty
        # writer target (don't try to materialize a non-existent
        # entry). ``zip_info=None`` with a non-empty parent leaves
        # ``_materialized`` False so a read falls through to the
        # directory probe and surfaces the missing-entry error.
        self._materialized: bool = False
        self._entry_mode: Mode = Mode.from_(mode, default=Mode.AUTO)

    # ==================================================================
    # Lazy materialization
    # ==================================================================

    def _materialize(self) -> None:
        """Pull the entry's bytes out of the parent archive once."""
        if self._materialized:
            return
        parent_size = self._zip_parent.size
        if parent_size == 0:
            self._materialized = True
            return
        with self._zip_parent.view(pos=0) as v:
            with zipfile.ZipFile(v, "r") as zf:
                payload = zf.read(self.entry_name)
        self._parent.write_bytes(payload, 0)
        # Track in case the directory hint was missing or out of date.
        self._uncompressed_size = len(payload)
        self._materialized = True

    # ==================================================================
    # Cheap size — directory hint without materializing
    # ==================================================================

    @property
    def size(self) -> int:
        if self._materialized:
            return self._parent.size
        if self._uncompressed_size is not None:
            return int(self._uncompressed_size)
        # Probe the central directory once; only fired when neither
        # hint nor materialized payload is available.
        if self._zip_parent.size == 0:
            return 0
        with self._zip_parent.view(pos=0) as v:
            with zipfile.ZipFile(v, "r") as zf:
                info = zf.getinfo(self.entry_name)
                self._zip_info = info
                self._uncompressed_size = int(info.file_size)
        return self._uncompressed_size

    # ==================================================================
    # Active holder routing — materialize on first read/write
    # ==================================================================

    def _active(self):
        # Any access through ``_active`` (read / write / pread /
        # pwrite / memoryview) needs the bytes to be there. This is
        # the single chokepoint the parent class uses, so guarding
        # it covers every entry-bytes access path.
        if not self._materialized:
            self._materialize()
        return super()._active()

    # ==================================================================
    # Tabular dispatch — use the entry's extension to pick the leaf
    # ==================================================================

    def _resolve_leaf(self) -> "Tabular | None":
        # Triggers the side-effecting registrations on every primitive
        # leaf — without this a caller that starts at zip_file directly
        # would see an empty Tabular registry and fail to dispatch.
        import yggdrasil.io.primitive  # noqa: F401

        try:
            mt = MediaType.from_(self.entry_name, default=None)
        except Exception:
            mt = None
        if mt is None:
            return None
        try:
            cls = Holder.class_for_media_type(mt, default=None)
        except Exception:
            cls = None
        if cls is None or cls is ZipEntryFile:
            return None
        # Force materialization before we hand the holder to the
        # tabular leaf — it'll read bytes off it directly.
        if not self._materialized:
            self._materialize()
        return cls(holder=self._parent, owns_holder=False)

    def _read_arrow_batches(self, options) -> Iterator[pa.RecordBatch]:
        leaf = self._resolve_leaf()
        if leaf is None:
            exts = _registered_tabular_extensions()
            raise ValueError(
                f"{type(self).__name__}: cannot read Arrow batches from "
                f"{_describe_entry_resolution_failure(self.entry_name)}. "
                f"Registered tabular extensions: {exts!r}."
            )
        return leaf._read_arrow_batches(leaf.options_class()())

    def _collect_schema(self, options) -> Schema:
        leaf = self._resolve_leaf()
        if leaf is None:
            return Schema.empty()
        return leaf._collect_schema(leaf.options_class()())

    # ==================================================================
    # Write surface — commits to parent archive on close
    # ==================================================================

    def open(
        self,
        mode: ModeLike = "rb+",
        **kwargs,
    ) -> "IO":
        """Acquire an :class:`IO` cursor for read or write.

        Write modes (``"wb"``, ``"w"``, ``"ab"``, …) reset the inner
        :class:`Memory` holder and return a cursor wrapped in a
        commit-on-exit context manager — the staged bytes commit into
        the parent archive when the ``with`` block exits cleanly. An
        exception inside the block drops the staged bytes without
        committing.

        Read modes materialize from the parent archive on first
        access and return a regular cursor — no commit fires.
        """
        m = Mode.from_(mode, default=Mode.AUTO)
        is_write = m.writable and m is not Mode.READ_ONLY and m is not Mode.AUTO
        if is_write:
            self._reset_staged_bytes()
            kwargs.setdefault("owns_holder", True)
            cursor = super().open(mode, **kwargs)
            return _CommitOnExit(entry=self, cursor=cursor)
        if not self._materialized:
            self._materialize()
        return super().open(mode, **kwargs)

    def _reset_staged_bytes(self) -> None:
        """Truncate the inner Memory holder so a write starts clean."""
        self._parent.seek(0)
        self._parent.truncate(0)
        self._materialized = True  # the empty buffer IS the entry now

    def _flush_to_archive(self) -> None:
        """Push the staged bytes into the parent archive.

        Called explicitly by :class:`_CommitOnExit` on clean exit of
        ``with entry.open("wb") as f:``, never by Disposable lifecycle
        — the name avoids colliding with the inherited
        :meth:`Disposable.commit` hook which fires on every close.

        Honours :attr:`_entry_mode` against any pre-existing entry of
        the same name:

        - ``AUTO`` / ``OVERWRITE`` — replace.
        - ``ERROR_IF_EXISTS`` — raise :class:`FileExistsError`.
        - ``IGNORE`` — skip silently.
        - ``APPEND`` — concatenate new bytes after the existing
          payload (entry-level append, not archive-level).
        """
        payload = self._parent.to_bytes()
        mode = self._entry_mode
        if mode in (Mode.ERROR_IF_EXISTS, Mode.IGNORE, Mode.APPEND):
            existing = self._read_existing_payload()
            if existing is not None:
                if mode is Mode.ERROR_IF_EXISTS:
                    raise FileExistsError(
                        f"Entry {self.entry_name!r} already exists in "
                        f"{self._zip_parent!r}; refusing to overwrite "
                        f"under mode={mode!r}."
                    )
                if mode is Mode.IGNORE:
                    return
                if mode is Mode.APPEND:
                    payload = existing + payload
        self._zip_parent._commit_entry(self.entry_name, payload)

    def _read_existing_payload(self) -> "bytes | None":
        """Return the pre-existing entry's bytes, or ``None`` if missing."""
        if self._zip_parent.size == 0:
            return None
        with self._zip_parent.view(pos=0) as v:
            with zipfile.ZipFile(v, "r") as zf:
                try:
                    return zf.read(self.entry_name)
                except KeyError:
                    return None

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options,
    ) -> None:
        """Pack *batches* through the format implied by ``entry_name``.

        Inferred Tabular leaf encodes the batches into a fresh
        :class:`Memory` buffer; those bytes become the entry's payload
        and commit through the same path the raw-bytes write uses.
        """
        # Side-effect import: ensures the primitive leaves are registered.
        import yggdrasil.io.primitive  # noqa: F401

        try:
            mt = MediaType.from_(self.entry_name, default=None)
        except Exception:
            mt = None
        cls = None
        if mt is not None:
            try:
                cls = Holder.class_for_media_type(mt, default=None)
            except Exception:
                cls = None
        if cls is None or cls is ZipEntryFile:
            exts = _registered_tabular_extensions()
            raise ValueError(
                f"{type(self).__name__}: cannot infer Tabular leaf for "
                f"entry_name {self.entry_name!r} — "
                f"{_describe_entry_resolution_failure(self.entry_name)}. "
                f"Registered tabular extensions: {exts!r}."
            )

        leaf = cls(holder=Memory(), owns_holder=True)
        leaf.write_arrow_batches(batches, options=leaf.options_class()())
        encoded = leaf.to_bytes()
        self._reset_staged_bytes()
        self._parent.write_bytes(encoded, 0)
        self._flush_to_archive()

    def __repr__(self) -> str:
        state = "materialized" if self._materialized else "lazy"
        return (
            f"<{type(self).__name__} {self.entry_name!r} "
            f"size={self.size} {state} parent={self._zip_parent!r}>"
        )


# ---------------------------------------------------------------------------
# ZipFile
# ---------------------------------------------------------------------------


class ZipFile(IO):
    """:class:`Tabular` leaf for ``.zip`` archives."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.ZIP

    @classmethod
    def options_class(cls):
        return ZipOptions

    # ``_write_lock`` serializes the per-entry commit (central-directory
    # rewrite) so concurrent ``z.entry("a").open()`` /
    # ``z.entry("b").open()`` writers can stage bytes in private
    # buffers in parallel without corrupting the archive's directory.
    @property
    def _write_lock(self) -> threading.Lock:
        lock = self.__dict__.get("__zip_write_lock__")
        if lock is None:
            lock = threading.Lock()
            self.__dict__["__zip_write_lock__"] = lock
        return lock

    # ==================================================================
    # Children surface — lazy iteration
    # ==================================================================

    def iter_children(self) -> Iterator[ZipEntryFile]:
        """Yield every archive entry as a lazy :class:`ZipEntryFile`.

        Skips directories (entries whose name ends with ``/``).
        Per-entry payloads are NOT fetched here — the directory walk
        is one ``zipfile.ZipFile.infolist`` call. The first read /
        ``size`` / Tabular hook on a child triggers its own
        single-entry decompression.
        """
        if self.size == 0:
            return
        with self.view(pos=0) as v:
            with zipfile.ZipFile(v, "r") as zf:
                infos = list(zf.infolist())
        for info in infos:
            if info.is_dir():
                continue
            yield ZipEntryFile(
                entry_name=info.filename,
                zip_parent=self,
                zip_info=info,
                tabular_parent=self,
            )

    def list_entries(self) -> "list[str]":
        """Return entry names in stored order. One directory walk; no decompression."""
        if self.size == 0:
            return []
        with self.view(pos=0) as v:
            with zipfile.ZipFile(v, "r") as zf:
                return [info.filename for info in zf.infolist() if not info.is_dir()]

    # ==================================================================
    # Per-entry handle — read AND write surface
    # ==================================================================

    def entry(
        self,
        name: str,
        *,
        mode: ModeLike = Mode.AUTO,
    ) -> ZipEntryFile:
        """Return a :class:`ZipEntryFile` for *name* — readable AND writable.

        The returned handle supports both surfaces::

            # Raw byte write — auto-commits on close.
            with z.entry("data.bin").open("wb") as f:
                f.write(b"...")

            # Tabular write — encoded through the inner format.
            z.entry("data.parquet").write_arrow_batches(batches)

            # Read existing entries.
            entry = z.entry("data.bin")
            payload = entry.to_bytes()

        Writes against different *name* values can run in parallel:
        each handle stages its bytes in a private :class:`Memory`
        holder, and the commit phase takes a per-archive
        :class:`threading.Lock` only for the brief central-directory
        rewrite. Writes against the same *name* race — the
        :attr:`mode` controls the resolution
        (``AUTO``/``OVERWRITE`` replace, ``ERROR_IF_EXISTS`` raises,
        ``IGNORE`` skips, ``APPEND`` concatenates).

        Unlike :meth:`child`, no error is raised when *name* doesn't
        exist — the returned handle is a writer-target.
        """
        info: "zipfile.ZipInfo | None" = None
        if self.size > 0:
            with self.view(pos=0) as v:
                with zipfile.ZipFile(v, "r") as zf:
                    try:
                        info = zf.getinfo(name)
                    except KeyError:
                        info = None
        return ZipEntryFile(
            entry_name=name,
            zip_parent=self,
            zip_info=info,
            mode=mode,
            tabular_parent=self,
        )

    def child(self, entry_name: str) -> ZipEntryFile:
        """Return a lazy :class:`ZipEntryFile` for *entry_name*.

        Raises :class:`KeyError` when the entry doesn't exist.
        Doesn't pre-fetch bytes — the returned handle materializes
        on first read.
        """
        with self.view(pos=0) as v:
            with zipfile.ZipFile(v, "r") as zf:
                try:
                    info = zf.getinfo(entry_name)
                except KeyError:
                    names = [i.filename for i in zf.infolist() if not i.is_dir()]
                    raise KeyError(
                        f"No entry named {entry_name!r} in {self!r}. "
                        f"Available: {names!r}."
                    )
        return ZipEntryFile(
            entry_name=entry_name,
            zip_parent=self,
            zip_info=info,
            tabular_parent=self,
        )

    # ==================================================================
    # Schema — first tabular entry's schema
    # ==================================================================

    def _collect_schema(self, options: ZipOptions) -> Schema:
        for child in self.iter_children():
            try:
                schema = child._collect_schema(child.options_class()())
            except Exception:
                continue
            if schema and not schema.is_empty():
                return schema
        return Schema.empty()

    # ==================================================================
    # Read path — concat batches across every tabular entry
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: ZipOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Walk every tabular entry and yield its batches in order.

        Single archive open — the directory is parsed ONCE and every
        tabular entry is decompressed inside the same
        :class:`zipfile.ZipFile` handle. The lazy per-child
        :meth:`ZipEntryFile._materialize` path re-parses the directory
        per entry, which is wasteful when the caller is walking the
        whole archive (the common Tabular read shape).

        Entries that DO resolve to a registered tabular leaf
        (parquet / csv / ndjson / arrow / …) stream their batches in
        archive order. Entries that don't resolve are skipped
        silently when at least one tabular entry is present — that's
        the contract for the zip-as-Tabular view. Use
        :meth:`iter_children` for an unfiltered lazy walk.

        Raises :class:`ValueError` when the archive has entries but
        NONE of them resolve to a registered tabular leaf — silently
        returning zero batches in that case hides the real problem
        (entry names missing the format extension, an unknown
        format, …) behind an empty read. Empty archives still
        return zero batches without raising.
        """
        if self.size == 0:
            return

        # Side-effect import: ensures the primitive leaves registered
        # before we probe the media-type → Holder class table.
        import yggdrasil.io.primitive  # noqa: F401

        with self.view(pos=0) as v:
            with zipfile.ZipFile(v, "r") as zf:
                file_entries = [
                    info for info in zf.infolist() if not info.is_dir()
                ]
                if not file_entries:
                    return

                resolved: "list[tuple[zipfile.ZipInfo, type[Tabular]]]" = []
                unresolved: "list[str]" = []
                for info in file_entries:
                    try:
                        mt = MediaType.from_(info.filename, default=None)
                    except Exception:
                        mt = None
                    cls = None
                    if mt is not None:
                        try:
                            cls = Holder.class_for_media_type(mt, default=None)
                        except Exception:
                            cls = None
                    if cls is None or cls is ZipEntryFile:
                        unresolved.append(
                            _describe_entry_resolution_failure(info.filename),
                        )
                        continue
                    resolved.append((info, cls))

                if not resolved:
                    exts = _registered_tabular_extensions()
                    raise ValueError(
                        f"{type(self).__name__}: archive has "
                        f"{len(file_entries)} entries but none resolve to a "
                        "registered Tabular leaf. Reasons: "
                        f"{unresolved!r}. Registered tabular extensions: "
                        f"{exts!r}."
                    )

                # Decompress + dispatch inline so we never pay N
                # directory parses for N entries.
                for info, cls in resolved:
                    payload = zf.read(info.filename)
                    leaf = cls(holder=Memory(payload), owns_holder=True)
                    yield from leaf._read_arrow_batches(
                        leaf.options_class()(),
                    )

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ZipOptions,
    ) -> None:
        """Persist *batches* as a single archive entry.

        Mode dispatch:

        - **OVERWRITE / AUTO / TRUNCATE** — fresh archive containing
          one entry whose bytes are the format implied by
          ``options.entry_name``'s extension.
        - **APPEND** — pull the existing archive's entries (streamed,
          one at a time), drop any whose name matches
          ``options.entry_name``, then write a fresh archive
          containing the survivors plus the new entry.
        - **IGNORE** — skip when non-empty.
        - **ERROR_IF_EXISTS** — raise when non-empty.
        - **UPSERT / MERGE** — degrade to APPEND.
        """
        action = self._resolve_action(options.mode)

        if action is Mode.IGNORE and self.size > 0:
            return
        if action is Mode.ERROR_IF_EXISTS and self.size > 0:
            raise FileExistsError(
                f"{type(self).__name__} buffer is non-empty "
                f"({self.size} bytes); refusing to overwrite under "
                f"mode={options.mode!r}."
            )
        if action is Mode.IGNORE or action is Mode.ERROR_IF_EXISTS:
            action = Mode.OVERWRITE

        # Pack the batches into the inner format implied by entry name.
        # Side-effect import: ensures the primitive leaves registered.
        import yggdrasil.io.primitive  # noqa: F401
        try:
            inner_mt = MediaType.from_(options.entry_name, default=None)
        except Exception:
            inner_mt = None
        inner_cls = None
        if inner_mt is not None:
            try:
                inner_cls = Holder.class_for_media_type(inner_mt, default=None)
            except Exception:
                inner_cls = None
        if inner_cls is None:
            exts = _registered_tabular_extensions()
            raise ValueError(
                f"{type(self).__name__}: cannot infer Tabular leaf for "
                f"entry_name {options.entry_name!r} — "
                f"{_describe_entry_resolution_failure(options.entry_name)}. "
                f"Set ZipOptions.entry_name to a value whose suffix is one "
                f"of the registered tabular extensions: {exts!r}."
            )

        leaf = inner_cls(holder=Memory(), owns_holder=True)
        leaf.write_arrow_batches(batches, options=leaf.options_class()())
        entry_payload = leaf.to_bytes()

        write_kwargs: dict = {"compression": options.compression}
        if options.compresslevel is not None:
            write_kwargs["compresslevel"] = options.compresslevel

        if action is Mode.APPEND and self.size > 0:
            # Append: build the new archive in a scratch buffer first.
            # Reading FROM self while writing TO self is impossible, and
            # the previous "read every survivor into a list" path
            # materialized every survivor's decompressed bytes in
            # memory simultaneously — multi-GB archives would balloon.
            # Stream each survivor chunk-by-chunk from the source's
            # ZipExtFile straight into the destination's writer; only
            # the in-flight chunk lives in memory at any moment.
            scratch = Memory()
            with self.view(pos=0) as src_v:
                with zipfile.ZipFile(src_v, "r") as src_zf:
                    with zipfile.ZipFile(scratch, "w", **write_kwargs) as dst_zf:
                        for info in src_zf.infolist():
                            if info.is_dir():
                                continue
                            if info.filename == options.entry_name:
                                continue
                            with src_zf.open(info, "r") as src_entry:
                                with dst_zf.open(info, "w") as dst_entry:
                                    while True:
                                        chunk = src_entry.read(1 << 20)
                                        if not chunk:
                                            break
                                        dst_entry.write(chunk)
                        dst_zf.writestr(options.entry_name, entry_payload)
            self.seek(0)
            self.truncate(0)
            # Zero-copy view of the rebuilt archive (scratch is a
            # separate holder); ``write_bytes`` copies it once into
            # ``self`` rather than ``to_bytes`` copying a second time.
            self.write_bytes(scratch.read_mv(-1, 0))
            return

        self.seek(0)
        self.truncate(0)
        with zipfile.ZipFile(self, "w", **write_kwargs) as zf:
            zf.writestr(options.entry_name, entry_payload)

    def write_entries(self, entries: Iterable[tuple[str, bytes]]) -> None:
        """Pack arbitrary ``(name, bytes)`` pairs into a fresh archive.

        Convenience for non-tabular use — bundling a few files
        without dropping into :mod:`zipfile`. Uses default
        :class:`ZipOptions` compression.
        """
        options = ZipOptions()
        self.seek(0)
        self.truncate(0)
        write_kwargs: dict = {"compression": options.compression}
        if options.compresslevel is not None:
            write_kwargs["compresslevel"] = options.compresslevel
        with zipfile.ZipFile(self, "w", **write_kwargs) as zf:
            for name, blob in entries:
                zf.writestr(name, blob)

    def _resolve_action(self, mode: Mode) -> Mode:
        if mode is Mode.AUTO or mode is Mode.OVERWRITE or mode is Mode.TRUNCATE:
            return Mode.OVERWRITE
        if mode is Mode.APPEND:
            return Mode.APPEND
        if mode is Mode.IGNORE:
            return Mode.IGNORE
        if mode is Mode.ERROR_IF_EXISTS:
            return Mode.ERROR_IF_EXISTS
        if mode is Mode.UPSERT or mode is Mode.MERGE:
            return Mode.APPEND
        return Mode.OVERWRITE

    # ==================================================================
    # Per-entry commit — driven by ZipEntryFile._flush_to_archive
    # ==================================================================

    def _commit_entry(
        self,
        entry_name: str,
        payload: bytes,
        *,
        compression: int = zipfile.ZIP_DEFLATED,
        compresslevel: "int | None" = None,
    ) -> None:
        """Merge *payload* into the archive under *entry_name*.

        Replaces any existing entry of the same name. The archive's
        central directory is rewritten under :attr:`_write_lock` so
        concurrent commits against different entry names don't
        corrupt the directory. Survivors stream chunk-by-chunk into a
        scratch buffer — the merge never materializes the whole
        archive in memory.
        """
        write_kwargs: dict = {"compression": compression}
        if compresslevel is not None:
            write_kwargs["compresslevel"] = compresslevel

        with self._write_lock:
            if self.size == 0:
                # Empty target — fresh archive with just this entry.
                with zipfile.ZipFile(self, "w", **write_kwargs) as zf:
                    zf.writestr(entry_name, payload)
                return

            scratch = Memory()
            with self.view(pos=0) as src_v:
                with zipfile.ZipFile(src_v, "r") as src_zf:
                    with zipfile.ZipFile(scratch, "w", **write_kwargs) as dst_zf:
                        for info in src_zf.infolist():
                            if info.is_dir():
                                continue
                            if info.filename == entry_name:
                                continue
                            with src_zf.open(info, "r") as src_entry:
                                with dst_zf.open(info, "w") as dst_entry:
                                    while True:
                                        chunk = src_entry.read(1 << 20)
                                        if not chunk:
                                            break
                                        dst_entry.write(chunk)
                        dst_zf.writestr(entry_name, payload)
            self.seek(0)
            self.truncate(0)
            # Zero-copy view of the rebuilt archive — one copy into
            # ``self`` instead of an extra ``to_bytes`` snapshot.
            self.write_bytes(scratch.read_mv(-1, 0))


# ---------------------------------------------------------------------------
# Internal: commit-on-clean-exit wrapper for ZipEntryFile.open(write_mode)
# ---------------------------------------------------------------------------


class _CommitOnExit:
    """Wrap a write cursor so the staged bytes commit on clean exit.

    On exception inside the ``with`` block the staged bytes are
    discarded — the parent archive is not modified. This is the
    single chokepoint :meth:`ZipEntryFile.open` uses to make
    ``with entry.open("wb") as f: f.write(...)`` atomic from the
    caller's viewpoint.
    """

    __slots__ = ("_entry", "_cursor")

    def __init__(self, *, entry: "ZipEntryFile", cursor: IO) -> None:
        self._entry = entry
        self._cursor = cursor

    def __enter__(self) -> IO:
        return self._cursor.__enter__()

    def __exit__(self, exc_type, exc, tb) -> "bool | None":
        # Flush BEFORE closing the cursor — closing the cursor
        # releases the ZipEntryFile's Memory holder, after which
        # ``self._parent.to_bytes()`` reads as empty.
        if exc_type is None:
            try:
                self._entry._flush_to_archive()
            finally:
                self._cursor.__exit__(None, None, None)
            return False
        # Exception path: drop staged bytes, just close the cursor.
        return self._cursor.__exit__(exc_type, exc, tb)
