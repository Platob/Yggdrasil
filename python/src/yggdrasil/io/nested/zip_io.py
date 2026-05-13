"""Zip-archive Tabular leaf with lazy per-entry I/O.

:class:`ZipIO` IS-A :class:`BytesIO` whose backing bytes are a zip
archive. It exposes two surfaces:

1. **Byte surface** — inherited from :class:`BytesIO`. Read / write /
   seek the raw archive bytes (useful for "open zip, drive
   :mod:`zipfile` yourself" flows).
2. **Children surface** — :meth:`iter_children` walks every entry as
   a :class:`ZipEntryIO`. The entries are **lazy**: their bytes are
   fetched from the parent archive on first read and cached after.
   Iterating doesn't decompress every entry up front.

Lazy children
-------------

A :class:`ZipEntryIO` is a :class:`BytesIO` over a :class:`Memory`
holder, but the holder's payload starts empty and is materialized
on first access through :meth:`ZipEntryIO._materialize`. The first
``read`` / ``size`` / ``seek_end`` / ``to_bytes`` / Tabular hook
triggers exactly one ``zipfile.ZipFile(parent_view).read(name)``
call; subsequent accesses hit the cached :class:`Memory` directly.

Tabular dispatch on a child uses the entry name's extension to pick
the right :class:`Tabular` leaf (parquet / csv / arrow / ndjson),
so ``zio["data.parquet"].read_arrow_table()`` works end-to-end
without the caller knowing the inner format.

Writing
-------

:meth:`_write_arrow_batches` packs the incoming batches into a
single entry whose name comes from ``options.entry_name``. The
extension picks the inner format (``data.parquet`` → parquet,
``data.csv`` → csv, …). Mode dispatch:

- **OVERWRITE** — fresh archive containing one entry.
- **APPEND** — keep existing entries that don't share the new name,
  add the new entry, rewrite the archive (zip's central directory
  sits at EOF, so a "true" append still requires a rewrite at this
  layer — but per-entry payloads are streamed through, not
  re-decompressed).
- **IGNORE** / **ERROR_IF_EXISTS** — guard non-empty archives.

Convenience helper :meth:`write_entries` packs arbitrary
``(name, bytes)`` pairs into a fresh archive.
"""

from __future__ import annotations

import dataclasses
import zipfile
from typing import ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.data.enums.media_type import MediaType
from yggdrasil.data.enums.mime_type import MimeType
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.memory import Memory
from yggdrasil.io.tabular.base import Tabular


__all__ = ["ZipIO", "ZipOptions", "ZipEntryIO"]


def _registered_tabular_extensions() -> "list[str]":
    """Return a sorted list of extensions a zip entry name can carry
    that will dispatch to a registered :class:`Tabular` leaf.

    Used by the read- and write-side errors so the message points at
    actual valid suffixes for the current registry state instead of a
    hard-coded sample.
    """
    from yggdrasil.io.tabular.base import _TABULAR_REGISTRY

    out: "set[str]" = set()
    for name in _TABULAR_REGISTRY:
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
    from yggdrasil.io.tabular.base import Tabular

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
        cls = Tabular.class_for_media_type(mt, default=None)
    except Exception:
        cls = None
    if cls is None:
        return (
            f"{entry_name!r}: MediaType {mt.mime_type.value!r} has no "
            "registered Tabular leaf"
        )
    return (
        f"{entry_name!r}: resolved to {cls.__name__} — unexpected, "
        "should have dispatched"
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ZipOptions(CastOptions):
    """:class:`CastOptions` extended with zip-archive knobs."""

    #: Name (and implied format) of the entry written by
    #: :meth:`ZipIO._write_arrow_batches`. The extension picks the
    #: inner Tabular leaf.
    entry_name: str = "data.parquet"
    compression: int = zipfile.ZIP_DEFLATED
    compresslevel: "int | None" = None


# ---------------------------------------------------------------------------
# ZipEntryIO — lazy per-entry BytesIO
# ---------------------------------------------------------------------------


class ZipEntryIO(BytesIO):
    """:class:`BytesIO` over a single zip entry's uncompressed payload.

    The payload is fetched from the parent archive on first access
    and cached in the inner :class:`Memory` holder. Reading the
    archive's directory (``ZipIO.list_entries`` / ``iter_children``)
    is a fixed-cost walk; per-entry decompression only happens for
    the entries the caller actually touches.

    Tabular hooks dispatch on the entry name's extension via
    :class:`Tabular.class_for_media_type`, so a parquet entry's
    :meth:`read_arrow_batches` runs the parquet reader against the
    entry bytes — no extra wiring.
    """

    __slots__ = (
        "entry_name",
        "_zip_parent",
        "_materialized",
        "_uncompressed_size",
        "_zip_info",
    )

    def __init__(
        self,
        *,
        entry_name: str,
        zip_parent: "ZipIO",
        zip_info: "zipfile.ZipInfo | None" = None,
        **kwargs,
    ) -> None:
        # Empty Memory holder; bytes land here on materialize.
        super().__init__(holder=Memory(), owns_holder=True, **kwargs)
        self.entry_name: str = entry_name
        self._zip_parent: "ZipIO" = zip_parent
        self._zip_info: "zipfile.ZipInfo | None" = zip_info
        # ``None`` until first directory probe; we keep the hint so
        # ``size`` can answer without decompressing.
        self._uncompressed_size: "int | None" = (
            zip_info.file_size if zip_info is not None else None
        )
        self._materialized: bool = False

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
        self._holder.write_bytes(payload, 0)
        # Track in case the directory hint was missing or out of date.
        self._uncompressed_size = len(payload)
        self._materialized = True

    # ==================================================================
    # Cheap size — directory hint without materializing
    # ==================================================================

    @property
    def size(self) -> int:
        if self._materialized:
            return self._holder.size
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
        # leaf — without this a caller that starts at zip_io directly
        # would see an empty Tabular registry and fail to dispatch.
        import yggdrasil.io.primitive  # noqa: F401

        try:
            mt = MediaType.from_(self.entry_name, default=None)
        except Exception:
            mt = None
        if mt is None:
            return None
        try:
            cls = Tabular.class_for_media_type(mt, default=None)
        except Exception:
            cls = None
        if cls is None or cls is ZipEntryIO:
            return None
        # Force materialization before we hand the holder to the
        # tabular leaf — it'll read bytes off it directly.
        if not self._materialized:
            self._materialize()
        return cls(holder=self._holder, owns_holder=False)

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

    def __repr__(self) -> str:
        state = "materialized" if self._materialized else "lazy"
        return (
            f"<{type(self).__name__} {self.entry_name!r} "
            f"size={self.size} {state} parent={self._zip_parent!r}>"
        )


# ---------------------------------------------------------------------------
# ZipIO
# ---------------------------------------------------------------------------


class ZipIO(BytesIO):
    """:class:`Tabular` leaf for ``.zip`` archives."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.ZIP

    @classmethod
    def options_class(cls):
        return ZipOptions

    # ==================================================================
    # Children surface — lazy iteration
    # ==================================================================

    def iter_children(self) -> Iterator[ZipEntryIO]:
        """Yield every archive entry as a lazy :class:`ZipEntryIO`.

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
            yield self.adopt_child(
                ZipEntryIO(
                    entry_name=info.filename,
                    zip_parent=self,
                    zip_info=info,
                )
            )

    def list_entries(self) -> "list[str]":
        """Return entry names in stored order. One directory walk; no decompression."""
        if self.size == 0:
            return []
        with self.view(pos=0) as v:
            with zipfile.ZipFile(v, "r") as zf:
                return [info.filename for info in zf.infolist() if not info.is_dir()]

    def child(self, entry_name: str) -> ZipEntryIO:
        """Return a lazy :class:`ZipEntryIO` for *entry_name*.

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
        return self.adopt_child(
            ZipEntryIO(entry_name=entry_name, zip_parent=self, zip_info=info)
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

        Entries that DO resolve to a registered tabular leaf
        (parquet / csv / ndjson / arrow / …) stream their batches in
        archive order. Entries that don't resolve are skipped
        silently when at least one tabular entry is present — that's
        the contract for the zip-as-Tabular view. Use
        :meth:`iter_children` for an unfiltered walk.

        Raises :class:`ValueError` when the archive has entries but
        NONE of them resolve to a registered tabular leaf — silently
        returning zero batches in that case hides the real problem
        (entry names missing the format extension, an unknown
        format, …) behind an empty read. Empty archives still
        return zero batches without raising.
        """
        children = list(self.iter_children())
        if not children:
            return

        leaves: "list[tuple[ZipEntryIO, Tabular]]" = []
        unresolved: "list[str]" = []
        for child in children:
            leaf = child._resolve_leaf()
            if leaf is None:
                unresolved.append(
                    _describe_entry_resolution_failure(child.entry_name)
                )
                continue
            leaves.append((child, leaf))

        if not leaves:
            exts = _registered_tabular_extensions()
            raise ValueError(
                f"{type(self).__name__}: archive has {len(children)} "
                "entries but none resolve to a registered Tabular leaf. "
                f"Reasons: {unresolved!r}. Registered tabular extensions: "
                f"{exts!r}."
            )

        for _child, leaf in leaves:
            yield from leaf._read_arrow_batches(leaf.options_class()())

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
                inner_cls = Tabular.class_for_media_type(inner_mt, default=None)
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

        # Append: read survivors entry-by-entry to avoid materializing
        # the whole archive at once.
        survivors: "list[tuple[str, bytes]]" = []
        if action is Mode.APPEND and self.size > 0:
            with self.view(pos=0) as v:
                with zipfile.ZipFile(v, "r") as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        if info.filename == options.entry_name:
                            continue
                        survivors.append((info.filename, zf.read(info.filename)))

        self.seek(0)
        self.truncate(0)
        write_kwargs: dict = {"compression": options.compression}
        if options.compresslevel is not None:
            write_kwargs["compresslevel"] = options.compresslevel
        with zipfile.ZipFile(self, "w", **write_kwargs) as zf:
            for name, blob in survivors:
                zf.writestr(name, blob)
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
