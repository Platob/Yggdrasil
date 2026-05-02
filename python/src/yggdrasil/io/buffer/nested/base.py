"""Abstract base for folder-oriented tabular aggregations.

:class:`NestedIO` is the sibling abstraction to :class:`PrimitiveIO`:
where ``PrimitiveIO`` is the multi-inheritance fold of
:class:`BytesIO + TabularIO` for single-buffer leaves (Parquet, IPC,
CSV, ...), :class:`NestedIO` is the fold of :class:`Path` (a folder
URL, no buffer) and :class:`TabularIO` for fragment-based
aggregations (folders of homogeneous files, Delta tables, Hive
partitions, multi-file IPC streams).

Why the split?
--------------

A primitive IO *is* the buffer — read/write a single sequence of
bytes, dispatched through one Arrow codec. A nested IO has no
buffer: its data is the union of its children's data. The two share
the Arrow record-batch protocol from :class:`TabularIO` but
nothing else — different storage model, different save-mode
semantics, different write strategy.

``isinstance(io, NestedIO)`` is the dual of
``isinstance(io, PrimitiveIO)`` and is used by orchestration code
(folder-of-folders writers, scan planners, fragment consumers) to
decide whether the IO must be drained fragment-by-fragment or can
be treated as a single byte stream.

The contract
------------

A subclass implements three things:

- :meth:`options_class`        — :class:`NestedOptions` subtype it
                                  consumes. Default :class:`NestedOptions`.
- :meth:`iter_fragments`       — yield :class:`Fragment` instances,
                                  one per child data unit. *Primary
                                  read API.* All read-side machinery
                                  (``_read_arrow_batches``,
                                  ``_collect_schema``) derives from
                                  this hook.
- :meth:`_make_child_io`       — mint a fresh :class:`PrimitiveIO`
                                  for a write target with a given
                                  child name and media type. *Primary
                                  write API.* :meth:`_write_arrow_batches`
                                  derives from this.

Storage model
-------------

A :class:`NestedIO` holds a single :class:`Path` pointing at the
folder root. Lifecycle (``open`` / ``close``) chains to the path —
opening the IO opens the path, closing the IO releases it.
``Disposable`` ownership ensures temp folders are unlinked when
the IO closes.

There is no buffer, no codec, no spill. Compression is the
responsibility of the child format (e.g. parquet's internal
codecs) — folder-level compression doesn't make sense when the
folder *is* the unit of storage.

Save mode semantics
-------------------

Folder formats naturally support OVERWRITE and APPEND:

- :attr:`Mode.OVERWRITE` — clear the folder of non-ignored
  children, then write a fresh child (or children, per
  ``options.child_row_size``). The folder itself is never deleted
  (would race with concurrent readers); only its contents.
- :attr:`Mode.APPEND` — mint a new child file alongside existing
  ones. The next index is computed from existing non-ignored
  children. Always native at the base.
- :attr:`Mode.UPSERT` — not supported by default. Subclasses with
  meaningful merge semantics (Delta, Iceberg) override.

Write strategy is caller-controlled via
:attr:`NestedOptions.child_row_size` and ``child_byte_size``: the
default is "one child file per write call" (single batch
iteration drained into a single staging file). Subclasses can
override :meth:`_write_arrow_batches` if they need more control.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    Literal,
    TypeVar,
)

import pyarrow as pa

from yggdrasil.arrow.cast import any_to_arrow_table
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.disposable import Disposable
from yggdrasil.environ import PyEnv
from yggdrasil.io.enums import MimeType, Mode
from yggdrasil.io.fragment import Fragment, FragmentInfos
from yggdrasil.io.fs import Path
from yggdrasil.io.tabular import TabularIO
from yggdrasil.lazy_imports import path_class

if TYPE_CHECKING:
    from yggdrasil.io.buffer.primitive import PrimitiveIO


__all__ = ["NestedIO", "NestedOptions"]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class NestedOptions(CastOptions):
    """Cast options extended with folder-write knobs.

    Inherits everything from :class:`CastOptions` (mode, row_size,
    byte_size, schema-cast hooks). The fragment-iteration knobs
    (``recursive``, key-glob filter, ``open_io``) are still on
    :class:`FragmentOptions` — they govern enumeration shape, not
    cast behavior. ``NestedOptions`` carries only the knobs that
    must reach the Arrow read/write hooks.

    :param child_media_type: the :class:`MediaType` to mint child
        files as on write. ``None`` (default) means "infer from the
        folder's child convention" — the concrete subclass decides
        (e.g. :class:`FolderIO` looks at the first existing child
        or falls back to a class-level default).
    :param child_row_size: row count per child file on write. ``0``
        or ``None`` means "one child file per write call" — the
        whole batch iterator is drained into a single staging
        file. Positive values cause the writer to roll over to a
        new staging file every ``child_row_size`` rows.
    :param child_byte_size: same as ``child_row_size`` but in
        approximate bytes. Mutually exclusive with
        ``child_row_size`` (row threshold wins if both set).
    :param populate_metadata: when iterating fragments, populate
        each :class:`FragmentInfos` with a :class:`Schema` (one
        ``collect_schema`` call per child) and ``mtime``. Default
        ``False`` — keeps enumeration cheap; consumers that need
        the schema can call ``frag.io.collect_schema()`` lazily.
    """

    child_media_type: Any = None
    child_row_size: int = 0
    child_byte_size: int = 0
    populate_metadata: bool = False


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

O = TypeVar("O", bound=NestedOptions)


# ---------------------------------------------------------------------------
# NestedIO
# ---------------------------------------------------------------------------


class NestedIO(TabularIO[O], ABC):
    """Marker base for folder-oriented tabular aggregations.

    Pairs with :class:`PrimitiveIO` as the two storage flavors of
    :class:`TabularIO`. Holds a single :class:`Path` (the folder
    root); reads enumerate fragments, writes mint child IOs.

    Concrete leaves implement :meth:`iter_fragments` and
    :meth:`_make_child_io`; everything else (Arrow batches, schema
    inspection, save-mode handling, write loop) is derived here.

    :class:`NestedIO` does NOT inherit from :class:`BytesIO`. It is
    Disposable + TabularIO. The held :class:`Path` is owned via
    :meth:`Disposable.add_owned` so its lifecycle (including
    ``temporary=True`` unlinking) chains to the IO.
    """

    # Used by :meth:`TabularIO.__new__` to short-circuit dispatch on
    # already-resolved leaves. Concrete subclasses (FolderIO,
    # DeltaIO, ...) flip this to True so a direct ``FolderIO(...)``
    # call bypasses the registry lookup.
    _FINAL_TABULAR_IO: ClassVar[bool] = False

    # ------------------------------------------------------------------
    # Registry hook
    # ------------------------------------------------------------------

    @classmethod
    def default_mime_type(cls) -> "MimeType | None":
        """Don't claim any mime type at the abstract layer.

        Mirrors :meth:`PrimitiveIO.default_mime_type`: returning
        ``None`` keeps :class:`NestedIO` itself out of the registry
        so :meth:`TabularIO.media_type_class` doesn't accidentally
        dispatch to the abstract base. Concrete subclasses
        (:class:`FolderIO` against ``MimeTypes.FOLDER``) override.
        """
        return None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __new__(
        cls,
        data: Any = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Route through :meth:`TabularIO.__new__` for media-type dispatch.

        - ``NestedIO(path=...)`` — dispatch to the right concrete
          leaf based on the path's inferred mime type. A directory
          path resolves to ``MimeTypes.FOLDER`` via
          ``Path.is_dir_sink``, landing on :class:`FolderIO` (or a
          subclass).
        - ``FolderIO(path=...)`` — already a concrete leaf;
          ``TabularIO.__new__`` short-circuits via ``_FINAL_TABULAR_IO``
          and returns ``object.__new__(cls)``.

        The data positional is forwarded so the dispatch sees both
        a kwarg ``media_type`` and any path-ish positional.
        """
        return TabularIO.__new__(cls, data, *args, **kwargs)

    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        media_type: Any = None,
        auto_open: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize from a folder path.

        ``data`` and ``path`` are accepted as the same thing — a
        path-ish object pointing at the folder root. ``data`` is
        the positional shape ``TabularIO.__new__`` already
        normalized; ``path`` is the kwarg shape callers prefer.
        Whichever is non-None wins; if both are supplied, ``path``
        takes precedence.

        ``media_type`` is captured by the registry but otherwise
        unused at the base — concrete leaves with ``default_mime_type``
        already know what they are.
        """
        Disposable.__init__(self)
        self._arrow_table = None
        self._spark_frame = None

        # Resolve the folder path. ``path`` kwarg wins over positional
        # ``data`` so callers who pass both don't get a surprise.
        raw = path if path is not None else data
        if raw is None:
            raise ValueError(
                f"{type(self).__name__} requires a path; got None. "
                "Pass path=... or a path-ish positional."
            )
        if isinstance(raw, Path):
            self.path = raw
        else:
            self.path = path_class().from_(raw)

        if auto_open:
            Disposable.open(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        """Ensure the underlying path is open.

        The path's own ``_acquire`` is a no-op (paths don't
        materialize files just because something opens a reference),
        but going through ``Disposable.open`` flips its acquired
        flag so subsequent operations see a live handle.
        """
        if not self.path.opened:
            self.path.open()

    def _release(self) -> None:
        """Tear down: clear caches and let the owned path release."""
        self.unpersist()
        # Owned-child release runs through the Disposable graph;
        # we don't need to call self.path.close() explicitly.
        # ``add_owned`` registered the path; the framework's
        # close pass walks owned children.

    @property
    def cached(self) -> bool:
        return self._arrow_table is not None or self._spark_frame is not None

    def unpersist(self) -> None:
        self._arrow_table = None
        self._spark_frame = None

    def persist(
        self,
        engine: Literal["arrow", "polars", "spark", "auto"] = "auto",
        *,
        data: Any | None = None,
    ) -> "NestedIO":
        """Materialize the folder into an in-process cache.

        Same shape as :meth:`PrimitiveIO.persist`. For folders,
        materialization runs the full fragment iteration and Arrow
        merge — useful when the same folder is read multiple times
        in close succession (avoids re-listing + re-opening every
        child). Engine ``auto`` resolves to spark on Databricks
        (driver-side), arrow elsewhere.
        """
        if self.cached:
            return self

        if not engine or engine == "auto":
            engine = "spark" if PyEnv.in_databricks() else "arrow"

        if data is None:
            if engine == "spark":
                self._spark_frame = self.read_spark_frame()
            elif engine == "arrow":
                self._arrow_table = self.read_arrow_table()
            else:
                raise ValueError(f"Unsupported engine: {engine}")
        else:
            if engine == "spark":
                from yggdrasil.spark.cast import any_to_spark_dataframe

                self._spark_frame = any_to_spark_dataframe(data)
            elif engine == "arrow":
                self._arrow_table = any_to_arrow_table(data)
            else:
                raise ValueError(f"Unsupported engine: {engine}")

        return self

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    @classmethod
    def options_class(cls) -> type[O]:
        return NestedOptions  # type: ignore[return-value]

    # ==================================================================
    # Fragment surface — the primary read API
    # ==================================================================

    @abstractmethod
    def iter_fragments(self, options: "O | None" = None, **kwargs: Any) -> Iterator[Fragment]:
        """Yield :class:`Fragment` instances, one per child data unit.

        *The* primary read API for nested IOs. All Arrow-batch
        machinery on this class derives from this hook.

        Subclasses implement enumeration (directory listing, log
        replay for Delta, manifest scan for Iceberg) and yield
        fragments with ``parent=None`` at the top level. Recursive
        descent is the consumer's responsibility — they walk the
        ``frag.io`` and re-call ``iter_fragments`` if it's another
        :class:`NestedIO`.

        Each yielded fragment carries:

        - ``infos.url`` — the child's location.
        - ``infos.mtime`` — the child's modification time, or
          ``0.0`` if not cheaply available (or
          ``populate_metadata`` is False).
        - ``infos.schema`` — the child's :class:`Schema`, or
          ``None`` if ``populate_metadata`` is False (the default;
          collect_schema costs a footer read per child).
        - ``io`` — a fresh :class:`PrimitiveIO` bound to the child
          path, ready for ``read_arrow_batches`` etc. The IO is
          owned by the iterator's lifetime; consumers that need
          to keep it past iteration call ``frag.with_io(...)`` to
          re-bind.

        :param options: a :class:`NestedOptions` (or subclass)
            instance. ``populate_metadata`` controls whether
            schema/mtime are filled in eagerly.
        """

    def to_fragment_infos(self) -> FragmentInfos:
        """Build a :class:`FragmentInfos` describing this folder.

        Used when this IO is itself wrapped as a fragment in a
        larger nested aggregation (folder-of-folders, Hive parent).
        """
        try:
            schema = self.collect_schema()
        except Exception:
            # collect_schema goes through every child; if any one
            # fails, prefer ``None`` over crashing the metadata
            # build. Consumers that need a schema call
            # ``collect_schema`` directly and handle the error.
            schema = None

        try:
            mtime = self.path.mtime or 0.0
        except Exception:
            mtime = 0.0

        return FragmentInfos(
            url=self.path.url,
            mtime=mtime,
            schema=schema,
        )

    def to_fragment(self) -> Fragment:
        return Fragment(
            infos=self.to_fragment_infos(),
            io=self,  # type: ignore[arg-type]
        )

    # ==================================================================
    # Child IO factory — the primary write API
    # ==================================================================

    @abstractmethod
    def _make_child_io(
        self,
        name: str,
        media_type: Any = None,
    ) -> "PrimitiveIO":
        """Mint a fresh :class:`PrimitiveIO` for a child write target.

        Returns a closed (un-acquired) IO bound to ``self.path / name``
        with the given media type. The writer opens it inside the
        write loop and closes it on success — the bound-path
        write-back fires on close.

        :param name: child filename (no path separators), already
            including the format extension. Subclasses that use
            staging should accept the staging name from
            :meth:`Path.make_staging` here and rename on success.
        :param media_type: media type for the child. ``None`` lets
            the subclass infer from extension or class default.
        """

    # ==================================================================
    # Mode resolution — folder-flavored counterpart of PrimitiveIO's
    # ==================================================================

    def is_empty(self) -> bool:
        """True when the folder has no non-ignored children.

        The base implementation defers to
        :meth:`iter_fragments` with a fresh empty options instance,
        peeks one fragment, and discards. Subclasses with a cheaper
        existence check (``ls`` directly, manifest probe) should
        override.
        """
        try:
            return next(iter(self.iter_fragments()), None) is None
        except FileNotFoundError:
            # The folder doesn't exist yet — that's empty.
            return True

    def _resolve_save_mode(self, mode: Any) -> Mode:
        """Resolve any :class:`Mode` to one a folder writer can branch on.

        Folder-flavored counterpart of
        :meth:`PrimitiveIO._resolve_save_mode`. Returns one of:

        - :attr:`Mode.OVERWRITE` — clear non-ignored children and
          write fresh.
        - :attr:`Mode.APPEND` — mint a new sibling alongside
          existing children (always native at the base).
        - :attr:`Mode.IGNORE` — folder non-empty, caller wants to
          skip.
        - :attr:`Mode.UPSERT` — only when subclass advertises
          ``_SUPPORTED_UPSERT``.

        AUTO/TRUNCATE map to OVERWRITE; ERROR_IF_EXISTS raises
        :class:`FileExistsError` when the folder is non-empty.
        """
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
                    f"Mode.ERROR_IF_EXISTS but folder is non-empty. "
                    f"Path: {self.path!r}"
                )
            return Mode.OVERWRITE

        # UPSERT and any subclass-defined modes pass through; the
        # writer is responsible for raising if it doesn't support
        # them.
        return m

    # ==================================================================
    # Folder mutators — cheap helpers used by the write path
    # ==================================================================

    def _clear_children(self) -> None:
        """Remove every non-ignored child of the folder.

        Used by OVERWRITE. Does not remove the folder itself —
        concurrent readers may hold the directory handle. Subclasses
        with ignore rules (``_is_ignored_path``) should override or
        feed the rule list through ``ls`` filters.

        Default implementation: remove every child that
        :meth:`_is_ignored_path` returns False for.
        """
        if not self.path.exists():
            return

        for child in self.path.iterdir():
            if self._is_ignored_path(child):
                continue
            try:
                child.remove(recursive=True, allow_not_found=True)
            except Exception:
                # Best-effort: if a single child can't be removed
                # (race with another writer, permissions), the
                # subsequent write will surface the failure.
                pass

    def _is_ignored_path(self, child: Path) -> bool:
        """Return True for paths that should be hidden from enumeration.

        Default: hide hidden files (name starts with ``.``).
        Subclasses (DeltaIO, IcebergIO) override to also hide their
        own metadata directories (``_delta_log/``, ``metadata/``).
        """
        return child.name.startswith(".")

    # ==================================================================
    # Read derivation — Arrow batches via fragment chaining
    # ==================================================================

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Chain :meth:`iter_fragments` into a single Arrow batch stream.

        Each fragment's IO is opened, drained, and closed in turn.
        The ``options`` are forwarded to each child read so cast /
        projection / row-size knobs apply per-batch the same way
        they would on a primitive.
        """
        if self.cached:
            yield from self._read_arrow_batches_from_cache(options)
            return

        for frag in self.iter_fragments(options):
            child_io = frag.io
            if child_io is None:
                # Pure location descriptor — skip, the caller asked
                # for io-less iteration.
                continue
            # Opening through the context manager guarantees the
            # child closes even if the consumer breaks out of the
            # outer loop early.
            with child_io:
                yield from child_io.read_arrow_batches(options=options)

    def _collect_schema(self, options: O) -> Schema:
        """Merge per-fragment schemas into a single folder schema.

        Stays in :class:`Schema` space throughout — ``Schema.merge_with``
        does the union with the same semantics ZIP uses. Empty
        folders return :meth:`Schema.empty`.
        """
        merged: Schema | None = None

        for frag in self.iter_fragments(options):
            # Prefer the fragment's pre-populated schema (when
            # ``populate_metadata`` was on) over re-opening the
            # child. Falls back to collect_schema on the IO.
            child_schema = frag.infos.schema
            if child_schema is None and frag.io is not None:
                with frag.io:
                    child_schema = frag.io.collect_schema(options=options)

            if child_schema is None:
                continue

            if merged is None:
                merged = child_schema
            else:
                merged = merged.merge_with(child_schema, inplace=True)

        return merged if merged is not None else Schema.empty()

    # ==================================================================
    # Write derivation — fragment minting + dispatch on save mode
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Dispatch a batch iterable into one or more child files.

        Resolves save mode first (so OVERWRITE can clear before any
        bytes hit disk), then streams batches into staging children
        per :attr:`NestedOptions.child_row_size` /
        ``child_byte_size``. On a successful drain, staging files
        are renamed to their final names.

        Failure modes:

        - Mid-stream exception: staging files remain (with their
          TTL-encoded names). Subsequent calls clean them up via
          ``Path.make_staging``'s sweep. Existing children are
          untouched.
        - Mode IGNORE on non-empty folder: no-op.
        - Mode UPSERT without subclass support: ``ValueError``.
        """
        mode = self._resolve_save_mode(options.mode)

        if mode is Mode.IGNORE:
            return

        if mode is Mode.UPSERT:
            self._arrow_upsert_via_rewrite(batches, options)
            return

        if mode is Mode.OVERWRITE:
            self._clear_children()

        # APPEND and OVERWRITE share the rest of the path: mint a
        # staging child (or several, if rolling over) and drain
        # batches into it.
        self._drain_into_children(batches, options)

    def _drain_into_children(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Drain *batches* into one or more child files.

        Single-child fast path when ``child_row_size`` and
        ``child_byte_size`` are both 0 (default): stream the whole
        iterator into one child.

        Roll-over path: every ``child_row_size`` rows (or
        ``child_byte_size`` bytes), close the current child and
        mint the next.
        """
        # Materialize the iterator once so we can probe whether
        # there's anything to write before minting a child.
        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            # Nothing to write. APPEND with empty input is a no-op;
            # OVERWRITE leaves the cleared folder empty (which is
            # the correct OVERWRITE-of-nothing state).
            return

        media_type = options.child_media_type or self._default_child_media_type()
        row_threshold = options.child_row_size or 0
        byte_threshold = options.child_byte_size or 0

        if row_threshold <= 0 and byte_threshold <= 0:
            # Single-child path. Re-prepend the peeked batch.
            self._write_one_child(
                _chain_first(first, batch_iter),
                media_type=media_type,
                options=options,
            )
            return

        # Roll-over path: split the stream into chunks by row /
        # byte threshold, write one child per chunk.
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
        batches: Iterable[pa.RecordBatch],
        *,
        media_type: Any,
        options: O,
    ) -> None:
        """Mint one staging child, drain *batches* into it, finalize.

        Uses :meth:`Path.make_staging` for the staging name (TTL-
        encoded so a failed write gets reaped automatically) and
        renames to the final ``part-{N}.{ext}`` name on success.
        Subclasses can override to skip the rename (e.g. Delta
        writes the staging file directly into a commit-log entry).
        """
        staging_path = self._make_staging_path(media_type)
        staging_name = staging_path.name

        child = self._make_child_io(staging_name, media_type=media_type)
        try:
            with child:
                # Forward the options unchanged — child format
                # writers see the cast knobs, row_size, etc.
                # ``mode`` is irrelevant for a fresh child but
                # passing it through keeps the call signature
                # consistent.
                child.write_arrow_batches(batches, options=options)
        except Exception:
            # Staging file's TTL ensures eventual cleanup; explicit
            # remove here is best-effort.
            try:
                staging_path.remove(allow_not_found=True)
            except Exception:
                pass
            raise

        # Rename into final position. Subclass override point:
        # _finalize_child can leave staging in place (Delta) or
        # rename to a transactional name (S3 multipart commit).
        self._finalize_child(staging_path, media_type=media_type)

    # ------------------------------------------------------------------
    # Staging / finalization hooks — subclasses override the bits they
    # care about.
    # ------------------------------------------------------------------

    def _make_staging_path(self, media_type: Any) -> Path:
        """Mint a staging :class:`Path` under :attr:`path`.

        Default: ``self.path.make_staging(media_type=media_type)`` —
        TTL-encoded name, parent-rate-limited sweep, ``temporary=True``
        so a failed lifecycle unlinks it. Subclasses with their own
        staging discipline (Delta puts staging files in the same
        directory as final ones to avoid a cross-directory rename)
        override.
        """
        return self.path.make_staging(media_type=media_type)

    def _finalize_child(self, staging_path: Path, *, media_type: Any) -> None:
        """Promote a staging file to its final name.

        Default: rename to ``part-{N}.{ext}`` where N is one past
        the largest existing ``part-`` index. Subclasses with their
        own naming (Delta's UUID-based filenames, Hive's
        partition-key-based) override.
        """
        final_name = self._next_child_name(media_type=media_type)
        final_path = self.path / final_name

        # Path.rename copies + removes; on a same-filesystem local
        # path the LocalPath fast path uses os.rename. Cross-backend
        # rename falls through to streaming copy.
        staging_path.rename(final_path)

    def _next_child_name(self, *, media_type: Any) -> str:
        """Compute the next ``part-{N}.{ext}`` name.

        Walks existing children once. The default ``part-`` prefix
        keeps APPEND counts monotonic across writers in the same
        process; collisions across processes are resolved by
        retrying with an incremented index (file-existence check
        is the writer's responsibility).
        """
        ext = self._extension_for(media_type)
        prefix = "part-"
        max_idx = -1

        if self.path.exists():
            for child in self.path.iterdir():
                name = child.name
                if not name.startswith(prefix):
                    continue
                # Strip prefix and the extension to isolate the
                # index part. Tolerant of extra dots in the
                # extension (``part-3.parquet.gz``).
                stem = name[len(prefix):]
                idx_str = stem.split(".", 1)[0]
                if not idx_str.isdigit():
                    continue
                idx = int(idx_str)
                if idx > max_idx:
                    max_idx = idx

        next_idx = max_idx + 1
        suffix = f".{ext}" if ext else ""
        return f"{prefix}{next_idx:05d}{suffix}"

    def _extension_for(self, media_type: Any) -> str:
        """Resolve a file extension for a child of the given media type.

        Goes through the :class:`MediaType` enum's
        ``full_extension`` if available, else empty string. Same
        helper :meth:`Path._staging_extension` uses internally —
        kept here as a method so subclasses can override
        per-format (e.g. Delta always writes ``parquet`` regardless
        of the configured child media type).
        """
        return Path._staging_extension(media_type)

    def _default_child_media_type(self) -> Any:
        """The media type to use for new children when not specified.

        Default ``None`` lets :meth:`_make_child_io` infer from the
        child name (which is what staging gives us — extension
        baked into the staging name). Subclasses with a fixed
        format (Delta = parquet, IcebergV1 = avro) override.
        """
        return None

    # ==================================================================
    # Append / upsert via rewrite — folder-flavored
    # ==================================================================

    def _arrow_upsert_via_rewrite(self, batches: Any, options: O) -> None:
        """Read existing, merge, OVERWRITE.

        Mirrors :meth:`TabularIO._arrow_upsert_via_rewrite` but at
        the folder level: read every existing child into one Arrow
        table, merge with incoming on ``options.match_by_names``,
        clear the folder, write a single fresh child with the
        merged result.

        Subclasses with cheaper merge semantics (Delta's commit
        log, Iceberg's manifest rewrite) override to avoid the
        full read.
        """
        match_by = options.match_by_names
        if not match_by:
            raise ValueError(
                f"{type(self).__name__} UPSERT requires "
                "options.match_by_names to be a non-empty sequence "
                "of column names. For 'replace everything,' use "
                "Mode.OVERWRITE instead."
            )

        existing_table = self._read_arrow_table(options.copy(read_seek=0))
        incoming_table = any_to_arrow_table(batches, options)

        merged = self.merge_upsert_tables(
            existing_table, incoming_table, match_by=match_by,
        )

        # Clear and write the merged result as a single new child.
        # Subclasses that need to preserve fine-grained child layout
        # override this whole method.
        overwrite_options = options.copy(mode=Mode.OVERWRITE)
        row_size = getattr(overwrite_options, "row_size", None) or None
        self._write_arrow_batches(
            merged.to_batches(max_chunksize=row_size),
            overwrite_options,
        )


# ---------------------------------------------------------------------------
# Internal stream helpers
# ---------------------------------------------------------------------------


def _chain_first(
    first: pa.RecordBatch,
    rest: Iterator[pa.RecordBatch],
) -> Iterator[pa.RecordBatch]:
    """Yield *first*, then every batch from *rest*.

    Lets the caller probe an iterator (peek the first batch to
    detect emptiness) without losing the peeked element.
    """
    yield first
    yield from rest


def _split_batches(
    batches: Iterator[pa.RecordBatch],
    *,
    row_threshold: int,
    byte_threshold: int,
) -> Iterator[Iterator[pa.RecordBatch]]:
    """Split a batch iterator into chunks by row / byte threshold.

    Each yielded inner iterator is one chunk; consumers should
    drain it before pulling the next outer item, since the outer
    loop drives the underlying ``batches`` cursor. ``row_threshold``
    wins when both are set.

    Edge cases:

    - A single batch larger than the threshold goes into its own
      chunk untouched (we don't slice batches — that's the caller
      format's concern).
    - Threshold of 0 is treated as "no threshold" by the caller;
      this helper assumes at least one threshold is positive.
    """

    def _size_bytes(batch: pa.RecordBatch) -> int:
        try:
            return int(batch.nbytes)
        except Exception:
            return 0

    pending: list[pa.RecordBatch] = []
    rows = 0
    nbytes = 0

    for batch in batches:
        pending.append(batch)
        rows += batch.num_rows
        if byte_threshold > 0:
            nbytes += _size_bytes(batch)

        flush = False
        if 0 < row_threshold <= rows:
            flush = True
        elif 0 < byte_threshold <= nbytes:
            flush = True

        if flush:
            yield iter(pending)
            pending = []
            rows = 0
            nbytes = 0

    if pending:
        yield iter(pending)