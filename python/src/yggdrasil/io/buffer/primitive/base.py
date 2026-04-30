"""Marker subclass of :class:`PrimitiveIO` for single-buffer tabular
formats.

All mode resolution, codec round-tripping, append/upsert via
rewrite, and lifecycle context management live on :class:`PrimitiveIO`
itself — :class:`PrimitiveIO` is now purely an isinstance-target
that distinguishes single-buffer leaves (Parquet, Arrow IPC, CSV,
…) from folder-oriented :class:`NestedIO` aggregations.

Why keep the layer at all?
--------------------------

1. ``isinstance(io, PrimitiveIO)`` is used in folder/nested writers
   to decide whether a child can be drained as a single buffer.
2. Some formats may grow single-buffer-specific behavior later
   (e.g. format-specific footer caching) that doesn't generalize
   to nested writers — having the hook point in the hierarchy
   means we don't need to refactor again to add it.

Multiple inheritance contract
-----------------------------

:class:`PrimitiveIO` is the multi-inheritance fold point of
:class:`BytesIO` (the byte buffer + path-bound storage) and
:class:`TabularIO` (the engine-agnostic Arrow protocol). The
two share :class:`Disposable` as their common base. For the
slot layout to compose, exactly one of them may add slots
beyond Disposable — :class:`TabularIO` is therefore declared
with ``__slots__ = ()`` and its two cache fields
(``_arrow_table`` / ``_spark_frame``) are carried here on
:class:`PrimitiveIO` instead. The conflicts that DO need
resolution beyond layout:

- ``__new__``  — :class:`BytesIO` doesn't override it; ``TabularIO``
  does media-type dispatch through its registry. We funnel
  ``PrimitiveIO(...)`` calls through ``TabularIO.__new__`` so the
  registry lookup picks the right leaf (ParquetIO, ArrowIPCIO,
  …); calls already specialized to a leaf class skip dispatch.

- ``__init__`` — :class:`BytesIO.__init__` accepts the buffer's
  kwarg signature; :class:`TabularIO.__init__` only zeroes the
  cache slots. We call ``BytesIO.__init__`` (which forwards
  ``auto_open`` and friends) and then poke the two cache slots
  directly, side-stepping a redundant ``Disposable.__init__``.

- ``_release`` — both parents define it. Python MRO would silently
  drop ``TabularIO._release`` (the cache flush). We override here
  to call both, in the right order: clear caches first (cheap,
  pure Python state), then let ``BytesIO`` do its fd/spill
  teardown.

- ``default_mime_type`` — :class:`TabularIO`'s default returns
  :data:`MimeTypes.OCTET_STREAM`, which would auto-register
  :class:`PrimitiveIO` itself and shadow the BytesIO fallback in
  :meth:`TabularIO.media_type_class`. Override to return ``None``
  so the registry skips this layer.
"""

from __future__ import annotations

import contextlib
from abc import ABC
from typing import Any, Iterator, Literal

from yggdrasil.arrow.cast import any_to_arrow_table
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.environ import PyEnv
from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.enums import MimeType, Mode, MediaType
from yggdrasil.io.fragment import Fragment
from yggdrasil.io.tabular import TabularIO
from yggdrasil.lazy_imports import fragment_class, fragment_infos_class

__all__ = ["PrimitiveIO"]


class PrimitiveIO(BytesIO, TabularIO, ABC):
    """Marker base for single-buffer tabular formats.

    The IO *is* the buffer: a :class:`PrimitiveIO` instance behaves
    as both a :class:`BytesIO` (positional read/write, spill,
    path binding) and a :class:`TabularIO` (engine-agnostic Arrow
    protocol). Concrete leaves (ParquetIO, ArrowIPCIO, CsvIO, …)
    implement the two abstract Arrow hooks (``_read_arrow_batches``
    / ``_write_arrow_batches``) and inherit everything else.
    """

    # ------------------------------------------------------------------
    # Registry hook
    # ------------------------------------------------------------------

    @classmethod
    def default_mime_type(cls) -> "MimeType | None":
        """Don't claim any mime type — only concrete leaves register.

        Without this override, :meth:`TabularIO.__init_subclass__`
        would register :class:`PrimitiveIO` itself against
        :data:`MimeTypes.OCTET_STREAM` (inherited default), shadowing
        :class:`BytesIO` as the fallback in
        :meth:`TabularIO.media_type_class`.
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

        - ``PrimitiveIO(path=...)`` — dispatch to the right concrete
          leaf based on the path's inferred mime type, falling back
          to :class:`BytesIO` if nothing claims it (handled inside
          ``TabularIO.__new__``).
        - ``ParquetIO(path=...)`` — already a concrete leaf;
          ``TabularIO.__new__`` short-circuits via the
          ``_FINAL_TABULAR_IO`` flag or the ``target is cls`` check
          and returns ``object.__new__(cls)``.

        The data positional is forwarded so the dispatch sees both
        kwarg-supplied media_type and any path-ish data positional.
        """
        return TabularIO.__new__(cls, data, *args, **kwargs)

    def __init__(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        BytesIO.__init__(self, data, *args, **kwargs)
        # Don't call TabularIO.__init__ — it would re-run Disposable.__init__
        # and stomp BytesIO's already-initialized state. Set the cache slots
        # directly (matches the docstring contract).
        self._arrow_table = None
        self._spark_frame = None

        if self._media_type is None:
            mt = self.default_mime_type()

            if not mt.is_any_bytes:
                self._media_type = MediaType(mt)

    # ------------------------------------------------------------------
    # Lifecycle — chain both parents' _release
    # ------------------------------------------------------------------

    def _release(self, committed: bool) -> None:
        """Tear down both halves on close.

        MRO would otherwise silently drop ``TabularIO._release``
        (which clears the Arrow / Spark caches). Order matters:
        clear the in-process caches first — they're pure Python
        state, can't fail meaningfully — then let ``BytesIO`` do
        its fd-close / spill-unlink dance, which is the part that
        can raise and must run regardless.
        """
        super()._release(committed=committed)

        # BytesIO side: flush transaction buffer, close fd, unlink
        # owned spill. May raise; caller wants that error.
        BytesIO._release(self, committed)

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
    ) -> "TabularIO":
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
    # Self-wrap shortcut
    # ------------------------------------------------------------------

    def as_media(self, media_type: "Any | None" = None) -> "TabularIO":
        """A :class:`PrimitiveIO` is already its own tabular view.

        :meth:`BytesIO.as_media` builds a fresh :class:`TabularIO`
        wrapper around the buffer; for a :class:`PrimitiveIO`
        instance that's just self with extra indirection. If the
        caller passed a ``media_type``, we route through
        ``with_media_type`` so the request still has effect.
        """
        if media_type is not None:
            return self.with_media_type(media_type, copy=False)
        return self

    # ------------------------------------------------------------------
    # Fragment surface
    # ------------------------------------------------------------------

    def to_fragment_infos(self):
        try:
            schema = self.collect_schema()
        except AttributeError:
            schema = Schema.empty()

        return fragment_infos_class()(
            url=self.url,
            mtime=self.stat().mtime,
            schema=schema,
        )

    def to_fragment(self) -> "Fragment":
        return fragment_class()(
            infos=self.to_fragment_infos(),
            io=self,
        )

    # ==================================================================
    # Mode resolution
    # ==================================================================

    def _resolve_save_mode(self, mode: Any) -> Mode:
        """Resolve any :class:`Mode` to one a writer can branch on.

        Returns one of:

        - :attr:`Mode.OVERWRITE` — truncate and write fresh.
          Includes AUTO/TRUNCATE, IGNORE-with-empty-buffer,
          ERROR_IF_EXISTS-with-empty-buffer.
        - :attr:`Mode.APPEND` — only when ``_SUPPORTED_APPEND``.
        - :attr:`Mode.IGNORE` — buffer non-empty, caller wants
          to skip.
        - :attr:`Mode.UPSERT` — only when ``_SUPPORTED_UPSERT``.

        Raises :class:`ValueError` for unsupported APPEND/UPSERT
        with a subclass-specific hint, :class:`FileExistsError` for
        ERROR_IF_EXISTS on a non-empty buffer.
        """
        m = Mode.from_(mode, default=Mode.AUTO)

        if m in (Mode.AUTO, Mode.OVERWRITE, Mode.TRUNCATE):
            return Mode.OVERWRITE

        if m is Mode.IGNORE:
            return Mode.IGNORE if not self.is_empty() else Mode.OVERWRITE

        if m is Mode.ERROR_IF_EXISTS:
            if not self.is_empty():
                raise FileExistsError(
                    f"{type(self).__name__} write with "
                    f"Mode.ERROR_IF_EXISTS but buffer is non-empty "
                    f"({self.size} bytes). Path: {self.path!r}"
                )
            return Mode.OVERWRITE

        return m

    # ==================================================================
    # Codec siblings
    # ==================================================================

    def _make_uncompressed_sibling(self) -> "PrimitiveIO":
        """Build an uncompressed sibling carrying self's bytes decompressed.

        The sibling is the same concrete class as ``self``; it gets
        ``default_mime_type()`` (no codec) as its media type so a
        downstream lookup of ``codec`` on the sibling returns
        ``None`` and any recursion through the codec branch
        terminates.
        """
        codec = self.codec
        if codec is None:
            raise RuntimeError(
                f"_make_uncompressed_sibling called on {type(self).__name__} "
                "with no codec; this is a bug in the caller."
            )

        decompressed_buf = codec.decompress(self, copy=True)
        return type(self)(
            decompressed_buf,
            media_type=type(self).default_mime_type(),
        )

    def _make_empty_sibling(self) -> "PrimitiveIO":
        """Empty sibling, no source bytes — same format minus the codec.

        Used by the write codec branch: the body fills the sibling
        with raw format bytes, then we compress on the way out.
        Deliberately not via :meth:`_make_uncompressed_sibling` —
        that decompresses self's current bytes, which for a write
        target are either empty or the previous compressed version
        we're about to overwrite.
        """
        return type(self)(
            media_type=type(self).default_mime_type(),
        )

    # ==================================================================
    # Lifecycle context managers — open/seek/codec
    # ==================================================================

    @contextlib.contextmanager
    def _reading_context(self, options: CastOptions) -> Iterator["PrimitiveIO"]:
        """Open an IO for reading; yield the IO the body should read from.

        Cursor-transparent: if ``self`` was already open on entry, the
        cursor is restored on exit regardless of where the body left
        it. This makes incidental reads (``collect_schema``, footer
        probes) safe to call mid-stream without disturbing an outer
        iteration.

        With a codec, yields a transient decompressed sibling whose
        lifetime is bounded by this context — the sibling is opened on
        entry and closed (scratch buffer unlinked) on exit, including
        the unhappy-path exit where the consumer breaks out early or
        an exception propagates. The codec path is naturally cursor-
        transparent because the sibling is discarded.

        Driven by *options*:

        - ``options.read_seek`` — cursor to seek to before the body
          runs on the yielded IO. ``None`` leaves it untouched.
          Defaults to ``0`` on CastOptions.
        """
        with contextlib.ExitStack() as stack:
            was_opened = self.opened

            try:
                if self.codec is not None:
                    target = stack.enter_context(self._make_uncompressed_sibling())
                else:
                    target = self
                    if not target.opened:
                        target.open()
                        stack.callback(target.close)
                    elif target.seekable():
                        # Always restore on exit when we didn't open it
                        # ourselves — incidental reads (collect_schema,
                        # footer probes) must not disturb an outer
                        # cursor.
                        stack.callback(target.seek, target.tell())

                if options.read_seek is not None and target.seekable():
                    target.seek(options.read_seek)

                yield target
            finally:
                if was_opened and self.closed:
                    self.open()

    @contextlib.contextmanager
    def _writing_context(self, options: CastOptions) -> Iterator["PrimitiveIO"]:
        """Open an IO for writing; yield the IO the body should write to.

        With no codec, yields ``self``. With a codec, yields a
        transient uncompressed sibling — the body writes the raw
        format bytes into the sibling, and on successful exit the
        sibling's bytes are compressed back into ``self`` and
        ``self`` is marked dirty so the bound path's write-back
        fires on close.

        On exception inside the body during the codec branch,
        ``self`` is left untouched (the sibling is discarded).

        Driven by *options*:

        - ``options.truncate_before_write`` — truncate the yielded
          IO to zero before the body. Set by OVERWRITE; cleared by
          APPEND.
        - ``options.write_seek`` — cursor on the yielded IO before
          the body. ``None`` leaves it untouched, ``0`` rewinds,
          ``-1`` seeks to end (SEEK_END). APPEND sets ``-1``.
        - ``options.mark_dirty_on_write`` — if True (default), mark
          the yielded IO dirty after the body. In the codec branch
          ``self`` is additionally marked dirty after compression
          replaces its payload, regardless of this flag.
        - ``options.reset_seek`` — restore the pre-entry cursor on
          exit (only when the IO stays open).
        """
        if self.codec is not None:
            yield from self._writing_context_compressed(options)
            return

        with contextlib.ExitStack() as stack:
            was_opened = self.opened
            if not was_opened:
                self.open()
                stack.callback(self.close)
            elif options.reset_seek and self.seekable():
                stack.callback(self.seek, self.tell())

            try:
                if options.mode is Mode.OVERWRITE:
                    self.truncate(0)

                if options.write_seek is not None and self.seekable():
                    self.seek(options.write_seek)

                if options.mark_dirty_on_write:
                    self.mark_dirty()

                yield self
            finally:
                if was_opened and self.closed:
                    self.open()

    def _writing_context_compressed(self, options: CastOptions) -> Iterator["PrimitiveIO"]:
        """Codec branch of :meth:`_writing_context`.

        Pulled out so the no-codec fast path stays flat. Yields the
        sibling the body should write to; on successful exit
        compresses sibling bytes into ``self``.
        """
        codec = self.codec
        assert codec is not None

        if options.mode is Mode.APPEND:
            sibling = self.decompress(codec=codec, copy=True)
        else:
            sibling = self._make_empty_sibling()
        with sibling:
            if options.write_seek is not None and sibling.seekable():
                sibling.seek(options.write_seek)

            yield sibling

            sibling.seek(0)
            compressed = codec.compress(sibling)

        # Sibling closed and scratch unlinked. Replace self's payload.
        self.truncate(0)
        self.seek(0)
        self.replace_with_payload(compressed)
        self.mark_dirty()