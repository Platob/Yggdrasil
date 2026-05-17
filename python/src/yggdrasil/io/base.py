"""Cursor + tabular handle bound to a managed :class:`Holder`.

:class:`IO[T, O]` IS-A :class:`Tabular`. It is a **pure seekable
cursor over a :class:`Holder`** — every public op is "do this at the
cursor" and dispatches straight to the bound holder. The IO carries
no scratch buffer of its own; reads / writes / truncates flow through
the holder's positional API, so the holder remains the single source
of truth for the durable bytes.

On top of that cursor it implements the bytes surface (``read`` /
``write`` / ``seek`` / ``tell`` / positional reads / structured
primitives / codec-aware format helpers) and inherits the
row-oriented surface from :class:`Tabular` so byte-level and
Arrow-level operations live on the same handle.

The default :class:`IO` doesn't know what format its bytes encode,
so the two abstract :class:`Tabular` hooks
(:meth:`_read_arrow_batches` / :meth:`_write_arrow_batches`) raise
unless overridden. Format-specific leaves (Parquet, CSV, Arrow IPC,
…) subclass :class:`IO` and override the hooks against the same byte
holder.

*T* is the chunk type the concrete handle exposes (``bytes`` for the
canonical path); *O* is the :class:`CastOptions` subtype carried by
:class:`Tabular`. The generic split lets a future text-shaped or
block-shaped handle reuse the holder + cursor + lifecycle plumbing
without inheriting the bytes-specific helpers.

Construction shapes
-------------------

- **Borrow** — ``IO(holder=h)``. Closing the IO never touches *h*.
  Two cursors over a long-lived buffer:
  ``c1, c2 = IO(holder=mem), IO(holder=mem)``.
- **Own** — ``IO(holder=h, owns_holder=True)``. Closing the IO closes
  *h*. This is what :meth:`Holder.open` returns.
- **Auto** — ``IO(data)``, ``IO(path=...)``, ``IO(binary=...)``, or
  ``IO.from_(obj)`` route any bytes-like / path-like / file-like /
  holder input through the right :class:`Holder` constructor and
  return an owning IO.
- **Format dispatch** — ``IO(media_type="csv")``,
  ``IO(path="x.parquet")``, … resolve to the registered Tabular leaf
  (:class:`CsvIO`, :class:`ParquetIO`, …) automatically.

Lifecycle
---------

:meth:`_acquire` makes sure the holder is acquired (when owned),
applies the mode-driven side effects (``ERROR_IF_EXISTS`` guard,
``OVERWRITE`` / ``TRUNCATE`` zeroing the holder, ``APPEND``-style
cursor positioning) and returns. Reads and writes routed through the
IO between :meth:`_acquire` and :meth:`_release` go directly to the
durable holder. :meth:`_release` closes the holder when ``self`` owns
it; the holder honors its own :attr:`temporary` flag and discards the
payload at that point.

Modes
-----

Mode is stored as a typed :class:`Mode` enum. Construction accepts
the :class:`ModeLike` family (POSIX strings ``"rb"`` / ``"wb"`` /
``"ab"`` / ``"xb"`` plus ``+`` variants, human aliases, or the enum
directly). Effects on top of a holder:

- Initial cursor — :data:`Mode.APPEND` lands at EOF; everything else at 0.
- :meth:`readable` / :meth:`writable` reporting via Mode predicates.
- :data:`Mode.OVERWRITE` / :data:`Mode.TRUNCATE` truncate the holder on
  :meth:`_acquire`.
- :data:`Mode.ERROR_IF_EXISTS` raises :class:`FileExistsError` on
  :meth:`_acquire` when the holder is non-empty.
"""

from __future__ import annotations

import io as _stdlib_io
import os
import pathlib
import tempfile
import time
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterator,
    Optional,
    TypeVar,
    Union,
)

import pyarrow as pa

from yggdrasil.data.enums.mode import Mode, ModeLike
from yggdrasil.data.options import CastOptions
from yggdrasil.disposable import Disposable
from yggdrasil.io.tabular import Tabular

if TYPE_CHECKING:
    from yggdrasil.io.holder import Holder


__all__ = ["IO", "BytesLike", "T", "O"]


T = TypeVar("T")
O = TypeVar("O", bound=CastOptions)

BytesLike = Union[bytes, bytearray, memoryview]


def _mint_spill_path(ext: str, ttl_seconds: int) -> pathlib.Path:
    """Mint a fresh temp file path under :func:`tempfile.gettempdir`.

    Filename layout (time-sortable):
    ``tmp-{start}-{end}-{seed}.{ext}``. Both timestamps are zero-
    padded to 12 digits so a lexical sort of the temp directory
    yields chronological order — useful for debugging and the
    cross-process janitor that reaps orphans oldest-first. The file
    itself is not created here — the caller writes to it.
    """
    seed = os.urandom(8).hex()
    start = int(time.time())
    end = start + max(0, int(ttl_seconds))
    name = f"tmp-{start:012d}-{end:012d}-{seed}.{ext}"
    return pathlib.Path(tempfile.gettempdir()) / name


def _as_byte_mv(data: BytesLike) -> memoryview:
    """Normalize bytes-like input to a 1-D unsigned-byte memoryview.

    pyarrow ``Buffer`` arrives as ``format='b'`` with itemsize 1,
    which trips bytearray slice assignment unless we cast. Centralize
    the cast/contiguity dance so every write path stays consistent.
    """
    mv = memoryview(data)
    if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
        mv = mv.cast("B")
    if not mv.c_contiguous:
        mv = memoryview(bytes(mv))
    return mv


def _resolve_format_target(
    cls: type,
    *,
    media_type: Any,
    path: Any,
    data: Any,
    holder: "Holder | None",
) -> "type | None":
    """Resolve the registered :class:`Tabular` class for the given inputs.

    Resolution priority:

    1. Explicit *media_type* kwarg.
    2. *path* — extension via :meth:`URL.infer_media_type`.
    3. *data* — same, when it's URL-shaped (``str`` / ``pathlib.PurePath``
       / :class:`URL`); bytes-like and file-like inputs are skipped.
    4. *holder*'s stamped ``stat().media_type``.

    Returns ``None`` when no media type can be resolved or no registered
    leaf exists for the resolved type. Side-effect-imports
    :mod:`yggdrasil.io.primitive` so every concrete leaf's
    ``mime_type`` claim is in the registry by the time we look up.
    """
    # Side-effect import: ensures every leaf module has registered its
    # mime_type by the time we hit the registry.
    import yggdrasil.io.primitive  # noqa: F401
    from yggdrasil.data.enums.media_type import MediaType
    from yggdrasil.io.tabular.base import Tabular

    mt = (
        MediaType.from_(media_type, default=None)
        if media_type is not None else None
    )

    if mt is None:
        from yggdrasil.io.url import URL
        for src in (path, data):
            if src is None or isinstance(src, (bytes, bytearray, memoryview)):
                continue
            if hasattr(src, "read") and not isinstance(src, str):
                continue
            try:
                url = URL.from_(src)
            except Exception:
                continue
            mt = url.infer_media_type(default=None)
            if mt is not None:
                break
        if mt is None and holder is not None:
            try:
                # Read the holder's stamped media_type directly instead
                # of through :meth:`stat()` — a remote-backed holder's
                # ``stat()`` is a network probe, but ``media_type``
                # resolves from the URL extension cache without one.
                mt = getattr(holder, "media_type", None)
            except Exception:
                pass

    if mt is None:
        return None
    return Tabular.class_for_media_type(mt, default=None)


# ===========================================================================
# IO[T, O]
# ===========================================================================


class IO(Tabular[O], Disposable, Generic[T, O]):
    """Cursor + bytes surface + tabular view over a managed :class:`Holder`.

    The IO is a **pure seekable cursor** over the bound holder: every
    read / write / truncate dispatches straight through to the
    holder's positional API, so the holder remains the single source
    of truth for the durable bytes. There is no IO-side scratch
    buffer — the IO carries only the cursor and the mode.

    Mode-aware but format-agnostic at this layer. Mode controls
    cursor position (:data:`Mode.APPEND` → EOF), :meth:`readable` /
    :meth:`writable` reporting, whether :meth:`_acquire` truncates
    the holder (:data:`Mode.OVERWRITE` / :data:`Mode.TRUNCATE`), and
    the :class:`FileExistsError` for :data:`Mode.ERROR_IF_EXISTS`.

    The two :class:`Tabular` batch hooks default to
    :class:`NotImplementedError` — a plain :class:`IO` doesn't know
    what its bytes encode. Subclasses (ParquetIO, CsvIO, ArrowIPCIO,
    …) override the hooks to do format-specific decoding against the
    same holder.
    """

    __slots__ = (
        "_holder",
        "_owns_holder",
        "_pos",
        "_mode",
    )

    def __new__(
        cls,
        data: Any = None,
        *,
        holder: "Holder | None" = None,
        owns_holder: bool = False,
        mode: ModeLike = "rb+",
        media_type: Any = None,
        path: Any = None,
        binary: Any = None,
        url: Any = None,
        **kwargs: Any,
    ):
        """Allocate the instance, redirect by media type, and resolve a holder.

        Two-stage dispatch:

        1. **Format dispatch.** The inputs are inspected for a
           :class:`MediaType` (explicit *media_type*, *path*'s
           extension, *data*'s URL form when path-shaped, or the
           bound *holder*'s stamped media). When the resolved type
           has a registered :class:`Tabular` leaf and that leaf is
           a different class than *cls*, ``__new__`` recurses into
           the leaf so ``IO(path="x.csv")`` and
           ``BytesIO(path="x.parquet")`` land on :class:`CsvIO` and
           :class:`ParquetIO` respectively.
        2. **Holder resolution.** When no holder is supplied, the
           holder-shaped kwargs (``data`` / ``path`` / ``binary`` /
           ``url``) are forwarded to :class:`Holder`, whose own
           ``__new__`` scheme-dispatches to the right concrete
           subclass (:class:`Memory`, :class:`LocalPath`, …).

        Inputs :class:`Holder` doesn't recognize (file-like objects,
        backend-specific shapes) are left for the subclass
        ``__init__`` to drain — ``_holder`` stays ``None`` until the
        subclass populates it.
        """
        # Validate up front so the dispatch redirect (which may skip
        # __init__ when target isn't a subclass of cls) doesn't lose
        # the conflicting-args guards.
        if holder is not None and (data is not None or path is not None):
            raise TypeError(
                f"{cls.__name__} accepts holder= OR data OR path=, "
                "not multiple. Use IO(holder=h) to borrow an existing "
                "holder, IO(data) for bytes/file-like inputs, or "
                "IO(path=...) for filesystem/URL paths."
            )
        if data is not None and path is not None:
            raise TypeError(
                f"{cls.__name__} accepts data= OR path=, not both. "
                "Use IO(data=...) for bytes/file-like inputs and "
                "IO(path=...) for filesystem/URL paths."
            )

        target = _resolve_format_target(
            cls, media_type=media_type, path=path, data=data, holder=holder,
        )
        if target is not None and target is not cls and issubclass(target, IO):
            instance = target.__new__(
                target,
                data=data,
                holder=holder,
                owns_holder=owns_holder,
                mode=mode,
                media_type=media_type,
                path=path,
                binary=binary,
                url=url,
                **kwargs,
            )
            # When target isn't a subclass of cls, Python won't
            # auto-invoke __init__ on the returned instance — do it
            # ourselves so the instance is fully set up.
            if not isinstance(instance, cls):
                type(instance).__init__(
                    instance,
                    data=data,
                    holder=holder,
                    owns_holder=owns_holder,
                    mode=mode,
                    media_type=media_type,
                    path=path,
                    binary=binary,
                    url=url,
                    **kwargs,
                )
            return instance

        instance = super().__new__(cls)
        instance._holder = holder
        instance._owns_holder = bool(owns_holder)
        if holder is None:
            from yggdrasil.io.holder import Holder as _Holder
            try:
                instance._holder = _Holder(
                    data=data, path=path, binary=binary, url=url,
                )
                instance._owns_holder = True
            except TypeError:
                # Subclass may have richer drain logic (e.g. file-like
                # objects in :meth:`from_`). Leave the slots at their
                # initial values for the subclass to finish.
                pass
        return instance

    def __init__(
        self,
        data: Any = None,
        *,
        holder: "Holder | None" = None,
        owns_holder: bool = False,
        mode: ModeLike = "rb+",
        media_type: Any = None,
        path: Any = None,
        binary: Any = None,
        url: Any = None,
        **kwargs: Any,
    ) -> None:
        """Construct a cursor over a :class:`Holder`. Does NOT open.

        Pass at most one of *holder*, *data*, *path*, *binary*,
        *url*:

        - *holder*: borrow an existing holder. Set ``owns_holder=True``
          to transfer close-ownership to this IO (typically only
          :meth:`Holder.open` does this).
        - *data*: bytes-like / file-like input routed through
          :meth:`from_` to build a fresh in-memory holder; the new
          IO owns it.
        - *path*: ``str`` / :class:`pathlib.PurePath` / :class:`URL`
          routed through :class:`Holder` scheme dispatch (file://,
          s3://, dbfs://, …). The new IO owns the resulting path-bound
          holder.

        ``mode`` follows stdlib :func:`open` semantics, normalized to
        a :class:`Mode` enum. The IO is constructed un-acquired; reads
        and writes are valid against the durable holder regardless of
        whether the IO has been entered via ``with`` / :meth:`acquire`.
        Entering the IO applies the mode-driven side effects
        (``OVERWRITE`` truncates, ``ERROR_IF_EXISTS`` guards, ``APPEND``
        positions the cursor at EOF) and acquires the holder when
        :attr:`owns_holder`.
        """
        super().__init__(**kwargs)
        self._pos: int = 0
        self._mode: Mode = Mode.from_(mode)

        # :meth:`__new__` resolves the holder via :class:`Holder`'s
        # scheme dispatch for everything Holder understands (path,
        # binary, url, bytes-like data, all-None → empty Memory). The
        # only shape it can't drain is a file-like ``data`` argument —
        # :meth:`from_` handles that drain into a fresh Memory holder.
        if self._holder is None:
            tmp = self.from_(data, mode=mode)
            self._holder = tmp._holder
            self._owns_holder = True

        # Stamp media type onto the holder's IOStats — gives the codec
        # auto-handling path something to inspect, and makes the buffer
        # self-describing for downstream serializers.
        if media_type is not None:
            try:
                from yggdrasil.data.enums.media_type import MediaType
                mt = MediaType.from_(media_type, default=None)
                if mt is not None:
                    self._holder.media_type = mt
            except Exception:
                pass

    # ==================================================================
    # Construction routing
    # ==================================================================

    @classmethod
    def from_(cls, obj: Any, *, mode: ModeLike = "rb+", **kwargs: Any) -> "IO":
        """Auto-route *obj* to the right holder, return an owning IO.

        - :class:`IO` — pass through (idempotent).
        - :class:`Holder` — borrow into a fresh IO.
        - bytes-like (``bytes`` / ``bytearray`` / ``memoryview``) —
          wrap in a fresh :class:`Memory`.
        - path-like (``str`` / ``pathlib.Path`` / ``URL``) — wrap in
          a fresh holder via :class:`Holder` registry dispatch (file,
          s3, dbfs, …).
        - file-like (has ``read``) — drain into a fresh
          :class:`Memory`.

        The returned IO always owns its holder unless *obj* was
        already an IO of the calling class.
        """
        from yggdrasil.io.holder import Holder
        from yggdrasil.io.memory import Memory

        if isinstance(obj, cls):
            return obj

        if isinstance(obj, IO):
            # Different IO subclass over the same byte substrate —
            # borrow the holder rather than drain (drain would advance
            # ``obj``'s cursor and miss any bytes already consumed).
            return cls(holder=obj._holder, owns_holder=False, mode=mode, **kwargs)

        if isinstance(obj, Holder):
            return cls(holder=obj, owns_holder=False, mode=mode, **kwargs)

        if isinstance(obj, (bytes, bytearray, memoryview)):
            return cls(
                holder=Memory(binary=obj), owns_holder=True, mode=mode, **kwargs,
            )

        if hasattr(obj, "read") and not isinstance(obj, (str, bytes)):
            mem = Memory()
            mem.acquire()
            try:
                pos = 0
                while True:
                    chunk = obj.read(64 * 1024)
                    if not chunk:
                        break
                    mem.pwrite(chunk, pos)
                    pos += len(chunk)
            except BaseException:
                mem.close()
                raise
            return cls(holder=mem, owns_holder=True, mode=mode, **kwargs)

        # Path-like — let Holder dispatch decide the scheme via its
        # __new__ registry routing.
        try:
            holder = Holder(data=obj)
            return cls(holder=holder, owns_holder=True, mode=mode, **kwargs)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Cannot wrap {type(obj).__name__} as a {cls.__name__}. "
                f"Accepted: IO, Holder, bytes-like, file-like, "
                f"str/PurePath/URL. Got {obj!r}."
            ) from exc

    @classmethod
    def from_holder(
        cls,
        holder: "Holder",
        *,
        owns_holder: bool = False,
        mode: ModeLike = "rb+",
        media_type: Any = None,
        auto_open: bool = True,
        **kwargs: Any,
    ) -> "IO":
        """Construct an IO over *holder*, dispatching to the format leaf.

        Resolves the format-specific :class:`IO` leaf via *media_type*
        (when given) or the holder's stamped ``stat().media_type``, and
        returns an instance of that leaf bound to *holder*. When no
        leaf can be resolved, falls back to ``cls`` itself — so
        ``IO.from_holder(h)`` returns a plain :class:`IO`,
        ``BytesIO.from_holder(h)`` a :class:`BytesIO`.

        With *auto_open=True* (the default) the returned cursor is
        already acquired, so the caller can immediately read/write
        without entering a ``with`` block. Set *auto_open=False* to
        defer the acquire to the caller's ``with`` / :meth:`acquire`.

        *owns_holder=True* hands close-ownership of *holder* to the
        returned cursor — closing the cursor closes the holder. The
        default ``False`` keeps the holder's lifetime in the caller's
        hands; the returned cursor is a non-owning borrow.
        """
        instance = cls(
            holder=holder,
            owns_holder=owns_holder,
            mode=mode,
            media_type=media_type,
            **kwargs,
        )
        if auto_open and not instance._acquired:
            Disposable.open(instance)
        return instance

    # ==================================================================
    # Identity / state
    # ==================================================================

    @property
    def holder(self) -> "Holder":
        """The bound :class:`Holder`."""
        return self._holder

    @property
    def owns_holder(self) -> bool:
        """Whether closing self also closes the holder."""
        return self._owns_holder

    def remaining_bytes(self) -> int:
        return self.holder.size - self.tell()

    def is_empty(self):
        return self.remaining_bytes() == 0

    # ==================================================================
    # Disposable lifecycle — apply mode side effects, acquire the holder
    # ==================================================================

    def _acquire(self) -> None:
        """Acquire the bound holder and apply the mode side effects.

        Mode side-effects:

        - :data:`Mode.OVERWRITE` / :data:`Mode.TRUNCATE` — truncate
          the durable holder to zero bytes.
        - :data:`Mode.APPEND` — cursor parked at EOF.
        - :data:`Mode.ERROR_IF_EXISTS` — fail-fast
          :class:`FileExistsError` if the durable holder is non-empty.
        - :data:`Mode.READ_ONLY` / :data:`Mode.AUTO` / default —
          cursor at 0, durable bytes untouched.

        Note: must NOT call ``self._holder.open()`` — that's the
        IO-returning convenience and would recurse.
        """
        if self._owns_holder:
            self._holder.acquire()

        if self._mode is Mode.ERROR_IF_EXISTS and self._holder.size > 0:
            raise FileExistsError(
                f"{type(self).__name__} opened with mode={self._mode!r} "
                f"but holder is non-empty ({self._holder.size} bytes)."
            )

        if self._mode in (Mode.OVERWRITE, Mode.TRUNCATE):
            self._holder.truncate(0)

        self._pos = self._holder.size if self._mode.appendable else 0

    def _release(self) -> None:
        """Release the durable holder when ``self`` owns it.

        Pure cleanup — the cursor sits on top of the holder; there is
        no IO-side scratch to tear down. Holders honor their own
        :attr:`temporary` flag and discard the payload at close time.
        """
        if self._owns_holder:
            try:
                self._holder.close()
            except Exception:
                pass

        self._unpersist_schema()

    def _active(self) -> "Holder":
        """The holder this cursor reads / writes against.

        Returns ``self._holder`` directly — the IO is a pure cursor
        over a single holder. Subclasses that need a side effect
        before every byte-level access (lazy materialization in
        :class:`ZipEntryIO` / :class:`XlsxSheetIO`) override this
        hook to drive the side effect, then ``return super()._active()``.
        """
        return self._holder

    def view(
        self,
        *,
        pos: int = 0,
        size: Optional[int] = None,
        mode: ModeLike = "rb",
    ) -> "IO":
        """Return a fresh, non-owning IO over the buffer.

        With *size* unset the view shares the same holder as ``self``
        — zero copy, cursor seeded at *pos*. Useful for Parquet
        footer probes, zip directory walks, magic-byte sniffs.

        With *size* set, the view holds an in-memory copy of bytes
        ``[pos, pos+size)``. That's the right shape for a *bounded*
        sub-view that should not race with later mutations of the
        parent buffer.
        """
        if size is None:
            v = type(self)(holder=self._holder, owns_holder=False, mode=mode)
            v._pos = int(pos)
            return v
        if size < 0:
            raise ValueError(f"view size must be >= 0, got {size!r}")
        # Bounded view: snapshot the requested range.
        payload = self.pread(int(size), int(pos))
        return type(self)(payload)

    # ==================================================================
    # Codec auto-handling — peeks at the holder's MediaType
    # ==================================================================

    def _codec(self):
        """The codec on this buffer's :class:`MediaType`, or ``None``.

        Path-bound holders learn their media type from the URL suffix
        at construction (``data.csv.gz`` → CSV + GZIP); callers that
        build a :class:`Memory` holder by hand can seed
        ``stat().media_type`` to opt the buffer into codec
        round-tripping.
        """
        holder = self._holder
        if holder is None:
            return None
        try:
            # Read the holder's stamped ``media_type`` directly — going
            # through ``stat()`` would issue a network probe on every
            # arrow_input_stream open for a remote-backed holder, just
            # to extract the same value the lazy ``media_type`` slot
            # already carries.
            mt = holder.media_type
        except Exception:
            return None
        return getattr(mt, "codec", None) if mt is not None else None

    def _format_view(self) -> "IO":
        """A read-only IO over the *format* bytes.

        When the holder is uncompressed (no codec on the media type),
        returns a non-owning :meth:`view` of ``self``. When a codec is
        present, returns a fresh in-memory IO whose bytes are the
        decompressed payload — leaf readers parse the format directly
        from it without knowing the wire was compressed.

        The returned buffer is the caller's to close.
        """
        codec = self._codec()
        if codec is None:
            return self.view(pos=0)
        # ``codec.decompress`` accepts the source IO and returns a
        # freshly-allocated decompressed IO; caller closes.
        return codec.decompress(self)

    def _format_input(self) -> "_FormatInputContext":
        """Context manager yielding a pyarrow-friendly input source.

        Resolution:

        - **Local-path holder, no codec** → :func:`pyarrow.memory_map`
          opens an mmap over the file. The Parquet / Arrow IPC / CSV
          readers consume the mmap directly, so the file's pages stay
          in the kernel page cache and no Python-side copy happens.
        - **Anything else** → fall back to :meth:`_format_view`.

        The yielded value is whichever NativeFile / file-like object
        won the resolution; the context manager closes it on exit.
        """
        return _FormatInputContext(self)

    def _format_buffer(self) -> "_FormatBufferContext":
        """Context manager yielding a buffer to write raw format bytes into.

        For an uncompressed holder, the yielded buffer is ``self``,
        already truncated to zero so the writer starts clean.
        For a codec-tagged holder, the yielded buffer is a fresh
        in-memory IO; on exit the bytes are compressed and committed
        to ``self``.
        """
        return _FormatBufferContext(self)

    def arrow_input_stream(self) -> "_ArrowInputStreamContext":
        """Context manager yielding the cheapest :class:`pa.NativeFile` over the payload.

        Resolution:

        - **Local-path holder, no codec** → :func:`pyarrow.memory_map`
          (zero-copy :class:`pa.MemoryMappedFile`).
        - **Codec-tagged holder** → decompress the payload (via the
          codec's streaming :meth:`Codec.decompress`) and wrap the
          uncompressed bytes in a :class:`pa.BufferReader`.
        - **Anything else** → snapshot the payload and wrap in a
          :class:`pa.BufferReader`.

        The yielded stream is always a real :class:`pa.NativeFile`, so
        the caller can hand it directly to
        :class:`pa.ipc.RecordBatchFileReader`,
        :func:`pa.parquet.read_table`, :func:`pa.csv.read_csv`, etc.
        without re-wrapping. The stream and any scratch decompression
        buffer are closed on exit.
        """
        return _ArrowInputStreamContext(self)

    def arrow_output_stream(
        self, *, append: bool = False,
    ) -> "_ArrowOutputStreamContext":
        """Context manager yielding a :class:`pa.BufferOutputStream` writer.

        Caller pattern: ``with bio.arrow_output_stream() as sink:
        writer(sink)``. The yielded sink accepts the format encoder's
        writes against a pure-Arrow in-memory buffer. On a clean exit
        the encoded bytes are committed to ``self`` via
        :meth:`_commit_format_payload`, which:

        - compresses with the holder's codec when one is bound, and
        - replaces the buffer (``append=False``, default) or seeks
          to EOF and appends (``append=True``).

        On an exception the sink is closed and the parent is left
        untouched.
        """
        return _ArrowOutputStreamContext(self, append=append)

    def _commit_format_payload(
        self,
        payload: "Any",
        *,
        append: bool = False,
    ) -> int:
        """Bulk-commit a fully-encoded format payload to this buffer.

        ``payload`` is anything :func:`memoryview`-able — typically
        an :class:`pyarrow.Buffer` from a
        :class:`pyarrow.BufferOutputStream` after the format encoder
        finishes. The codec on :attr:`media_type` (when set) is
        applied here, then the bytes land in ``self`` with one
        ``truncate`` + one ``write`` (overwrite) or one seek-to-end
        + one ``write`` (append).

        Built for the format-leaf write path: each leaf
        (:class:`ParquetIO`, :class:`ArrowIPCIO`, :class:`CsvIO`,
        :class:`NDJsonIO`, …) drives its encoder against an Arrow
        sink, then hands the resulting buffer to this method instead
        of streaming the encoder's per-row-group / per-batch / per-row
        writes through :meth:`write`. Skips the small-write cost on
        path-backed holders without touching the IO open machinery.
        """
        view: "memoryview"
        if isinstance(payload, memoryview):
            view = payload
        elif isinstance(payload, (bytes, bytearray)):
            view = memoryview(payload)
        else:
            # ``pa.Buffer`` exposes the buffer protocol but isn't a
            # memoryview itself.
            view = memoryview(payload)

        codec = self._codec()
        if codec is not None and len(view) > 0:
            scratch = type(self)()
            try:
                scratch.write(view)
                scratch.seek(0)
                compressed = codec.compress(scratch)
                try:
                    view = memoryview(compressed.to_bytes())
                finally:
                    try:
                        compressed.close()
                    except Exception:
                        pass
            finally:
                try:
                    scratch.close()
                except Exception:
                    pass

        # When the IO is idle (not entered via ``with`` / :meth:`open`),
        # the cursor is implementation scratch — callers don't see it,
        # and a one-shot ``write_arrow_table`` shouldn't leave ``tell()``
        # parked at EOF on a buffer that's still un-acquired. Snapshot
        # ``_pos`` here and restore it on the way out so the next
        # idle-mode call (a fresh read, another write) starts from the
        # same cursor it observed before. While the IO is opened the
        # caller owns the cursor — leave it where the write landed.
        restore_pos = self._pos if not self._acquired else None

        n = len(view)
        if append:
            self.seek(0, 2)  # SEEK_END
        else:
            self.seek(0)
            self.truncate(0)
        if n > 0:
            self.write_bytes(view)

        if restore_pos is not None:
            self._pos = min(restore_pos, self.size)
        return n

    @property
    def size(self) -> int:
        """Live size from the bound holder."""
        return self._active().size

    @property
    def size_known(self) -> bool:
        """``True`` when reading :attr:`size` won't trigger a backend probe.

        Forwards to :attr:`Holder.size_known`. Format leaves that
        precheck ``size == 0`` to short-circuit on empty buffers
        should guard the check on this predicate so a cold remote
        path doesn't pay an extra round trip.
        """
        return self._active().size_known

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        return True

    def __bytes__(self) -> bytes:
        """Snapshot the active payload as :class:`bytes`."""
        return self.to_bytes()

    def __repr__(self) -> str:
        state = "acquired" if self._acquired else "idle"
        own = "owns" if self._owns_holder else "borrows"
        return (
            f"<{type(self).__name__} {state} {own} holder={self._holder!r} "
            f"pos={self._pos} mode={self._mode!r}>"
        )

    # ==================================================================
    # Tabular hooks — default raises; format-specific leaves override
    # ==================================================================

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Default — opaque buffer can't honestly yield Arrow batches.

        Format-specific subclasses (ParquetIO, CsvIO, ArrowIPCIO, …)
        override against the same holder. For dispatch by media type,
        construct via the format leaf directly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no tabular decoder. "
            "Construct via the format leaf (ParquetIO, CsvIO, …) "
            "to read Arrow record batches from this byte buffer."
        )

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Default — opaque buffer can't honestly accept Arrow batches.

        Format-specific subclasses override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no tabular encoder. "
            "Construct via the format leaf (ParquetIO, CsvIO, …) "
            "to write Arrow record batches into this byte buffer."
        )

    # ==================================================================
    # Mode predicates — stdlib open() semantics
    # ==================================================================

    @property
    def mode(self) -> Mode:
        """Normalized :class:`Mode` for this handle.

        Stored as an enum so predicates like :meth:`readable`,
        :meth:`writable`, :meth:`appendable` route through one
        canonical token instead of re-parsing strings at every
        call site. The original POSIX form is recoverable via
        ``self.mode.os_mode``.
        """
        return self._mode

    def readable(self) -> bool:
        return self._mode.readable

    def writable(self) -> bool:
        return self._mode.writable

    def appendable(self) -> bool:
        """True when writes append at EOF — :data:`Mode.APPEND` only."""
        return self._mode.appendable

    @property
    def name(self) -> str:
        return str(self._holder.url)

    @property
    def media_type(self):
        """The buffer's :class:`MediaType`, or ``None``.

        Reads :attr:`Holder.media_type` directly — the URL-extension
        cache is the source of truth and ``stat()`` would issue a
        backend probe on remote holders for no extra information.
        """
        try:
            return self._holder.media_type
        except Exception:
            return None

    def with_media_type(self, media_type: Any, *, copy: bool = False) -> "IO":
        """Stamp *media_type* onto the holder's :class:`IOStats`.

        With ``copy=False`` (the default), mutates ``self`` and returns
        it. ``copy=True`` allocates a fresh holder over the same bytes
        and returns a new IO over it.
        """
        from yggdrasil.data.enums.media_type import MediaType
        mt = MediaType.from_(media_type, default=None) if media_type is not None else None
        if copy:
            payload = self.to_bytes()
            return type(self)(payload, media_type=mt)
        if mt is not None:
            self._holder.media_type = mt
        return self

    def as_media(self, media_type: Any = None) -> "IO":
        """Return a typed Tabular leaf bound to this buffer's holder.

        Resolution: explicit *media_type* wins; otherwise the buffer's
        stamped media type (``self._holder.stat().media_type``) is used.
        The leaf borrows the same :class:`Holder` so durable bytes are
        shared without a copy. When ``self`` is already an instance of
        the resolved leaf class, returns ``self`` unchanged.

        Raises :class:`KeyError` when no media type can be resolved or
        the resolved type has no registered Tabular leaf.
        """
        # Side-effect import: every primitive leaf registers its
        # mime_type on import.
        import yggdrasil.io.primitive  # noqa: F401
        from yggdrasil.io.tabular.base import Tabular
        from yggdrasil.data.enums.media_type import MediaType

        mt = MediaType.from_(media_type, default=None) if media_type is not None else None
        if mt is None:
            try:
                mt = self._holder.media_type
            except Exception:
                mt = None
        if mt is None:
            raise KeyError(
                f"No media_type available for {self!r}. "
                "Pass media_type= explicitly or stamp it on the "
                "holder's IOStats via with_media_type()."
            )

        target = Tabular.class_for_media_type(mt)
        if isinstance(self, target):
            return self
        return target(
            holder=self._holder,
            owns_holder=False,
            mode=self._mode,
            media_type=mt,
        )

    @property
    def closed(self) -> bool:
        """Stdlib ``IO[bytes]`` parity — ``False`` while the bound
        holder is reachable.

        Stdlib semantics: ``closed`` means "file unusable for I/O."
        The IO is a pure cursor over the holder, so a fresh
        un-acquired IO is still usable; ``closed`` only flips when
        teardown has dropped the holder reference. This matters for
        pyarrow / pandas / polars / zipfile, which guard every op
        with an ``assert not closed`` and would otherwise refuse to
        write into a fresh, never-explicitly-opened IO.
        """
        return self._holder is None

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        """Underlying fd if the holder exposes one. Raises otherwise."""
        fileno = getattr(self._holder, "fileno", None)
        if fileno is None:
            raise OSError(
                f"{type(self).__name__} over {type(self._holder).__name__} "
                "has no underlying file descriptor."
            )
        return fileno()

    # ==================================================================
    # Cursor
    # ==================================================================

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = _stdlib_io.SEEK_SET) -> int:
        """Seek to *offset* relative to *whence*.

        Mirrors :meth:`io.IOBase.seek` with two ergonomic deviations
        that match the rest of the codebase:

        * ``seek(-1, SEEK_SET)`` is a "go to end" sentinel — pairs
          with ``read(-1)`` / "read all". Any other negative
          ``SEEK_SET`` offset raises :class:`ValueError`.
        * ``SEEK_CUR`` / ``SEEK_END`` with a negative offset that
          would land before byte 0 clamps to 0 instead of raising.
        """
        offset = int(offset)
        size = self.size
        if whence == _stdlib_io.SEEK_SET:
            if offset == -1:
                self._pos = size
            elif offset < 0:
                raise ValueError(
                    f"Negative SEEK_SET offset {offset!r} is invalid; "
                    f"use SEEK_END to count from the end."
                )
            else:
                self._pos = offset
        elif whence == _stdlib_io.SEEK_CUR:
            self._pos = max(0, self._pos + offset)
        elif whence == _stdlib_io.SEEK_END:
            self._pos = max(0, size + offset)
        else:
            raise ValueError(f"Invalid whence: {whence!r}")
        return self._pos

    def seekable(self) -> bool:
        return True

    # ==================================================================
    # Cursorless I/O — pass through to the holder
    # ==================================================================

    def pread(self, n: int, pos: int) -> bytes:
        """Positional read against the active holder. Cursor untouched."""
        return self._active().pread(n, pos)

    def pwrite(
        self, data: BytesLike, pos: int, *, update_stat: bool = True,
    ) -> int:
        """Positional write against the holder. Cursor untouched.

        ``update_stat=False`` is forwarded to the holder so a bulk
        loop can skip the per-write stat refresh.
        """
        return self._active().pwrite(data, pos, update_stat=update_stat)

    def memoryview(self) -> memoryview:
        """Memoryview over the active holder's full payload."""
        return self._active().memoryview()

    # ==================================================================
    # IO[bytes] core
    # ==================================================================

    def read(self, size: int = -1) -> bytes:
        """Read up to *size* bytes from the cursor, advancing past them.

        Stdlib :meth:`io.RawIOBase.read` semantic: ``size < 0`` /
        ``None`` reads to EOF; otherwise reads up to ``size``
        bytes, returning fewer at EOF. The holder's ``read_bytes``
        would raise on an out-of-range window, so cap to remaining
        before dispatching.
        """
        remaining = max(0, self.size - self._pos)
        if size is None or size < 0:
            size = remaining
        else:
            size = min(size, remaining)
        if size == 0:
            return b""
        out = self._active().read_bytes(size, offset=self._pos)
        self._pos += len(out)
        return out

    def readall(self) -> bytes:
        """Read from cursor to EOF, advancing the cursor."""
        return self.read(-1)

    def readinto(self, b: Any) -> int:
        """Fill *b* from the cursor, advance, return bytes written."""
        got = self._active().readinto(b, offset=self._pos)
        self._pos += got
        return got

    def readinto1(self, b: Any) -> int:
        return self.readinto(b)

    def readline(self, limit: int = -1) -> bytes:
        """Read up to the next newline from the cursor, advancing past it."""
        line = self._active().readline(limit, offset=self._pos)
        self._pos += len(line)
        return line

    def readlines(self, hint: int = -1) -> list[bytes]:
        lines = self._active().readlines(hint, offset=self._pos)
        self._pos += sum(len(line) for line in lines)
        return lines

    def write(self, b: Any, *, update_stat: bool = True) -> int:
        """Write *b* at the cursor, advancing it.

        Accepts bytes-like, ``str`` (UTF-8), ``io.BytesIO``, or any
        file-like with ``.read``. File-like sources route through
        :meth:`write_stream` so backends with an atomic whole-object
        upload push a single request. The buffer-protocol fallback
        catches things like :class:`pyarrow.Buffer` that aren't
        bytes/bytearray/memoryview but ARE memoryview-able.
        """
        if b is None:
            return 0
        if isinstance(b, str):
            return self.write_bytes(b.encode("utf-8"), update_stat=update_stat)
        if isinstance(b, (bytes, bytearray, memoryview)):
            return self.write_bytes(b, update_stat=update_stat)
        if hasattr(b, "read"):
            return self.write_stream(b)
        return self.write_bytes(memoryview(b), update_stat=update_stat)

    def write_bytes(
        self, b: BytesLike, *, size: int = -1, update_stat: bool = True,
    ) -> int:
        """Splice *b* at the cursor, advance, return bytes written.

        ``size>=0`` caps the byte count written. The cap is
        applied via a slice of the input buffer before the
        ``write_mv`` call (zero-copy for ``memoryview`` / ``bytes``).
        """
        mv = _as_byte_mv(b)
        if size >= 0 and len(mv) > size:
            mv = mv[:size]
        if len(mv) == 0:
            return 0
        n = self._active().write_mv(mv, self._pos, update_stat=update_stat)
        self._pos += n
        return n

    def write_stream(self, src: Any, *, size: int = -1) -> int:
        """Drain a binary source into self at the cursor.

        Cursor-anchored wrapper around :meth:`Holder.write_stream` —
        the holder's public ``write_stream`` coerces *src* to a
        real :class:`IO[bytes]` and dispatches to
        :meth:`Holder._write_stream`; this thin wrapper just
        advances the cursor by the bytes the holder reported
        writing. ``size>=0`` caps the byte count drained from *src*.
        """
        n = self._active().write_stream(src, offset=self._pos, size=size)
        self._pos += n
        return n

    def write_holder(self, src: Any, *, size: int = -1) -> int:
        """Splice another :class:`Holder`'s bytes into self at the cursor.

        Cursor-anchored wrapper around :meth:`Holder.write_holder` —
        small payloads land in one :meth:`Holder.write_mv` call,
        large ones stream through :meth:`Holder._write_stream`.
        ``size>=0`` caps the byte count pulled from *src*. Cursor
        advances by the byte count.
        """
        n = self._active().write_holder(src, offset=self._pos, size=size)
        self._pos += n
        return n

    def upload(
        self, src: Any, *, size: int = -1, offset: int = 0,
    ) -> "IO":
        """Pull *src*'s bytes into this IO at the current cursor.

        Cursor-anchored counterpart to :meth:`Holder.upload` —
        ``io.upload(src)`` writes *src*'s bytes starting at the
        IO's current position and advances the cursor by the
        number of bytes written. *size* / *offset* slice the
        **source** (same as :meth:`Holder.upload`): ``size=-1``
        reads to EOF, ``size>=0`` caps the count, ``offset`` is
        the position inside *src* to start reading from.

        *src* accepts a :class:`Holder`, another :class:`IO`
        cursor, or a ``str`` / :class:`os.PathLike` coerced via
        ``Path.from_(src)``. Returns *self* so cursor-chained
        writes work (``io.upload(src).write(b)``).
        """
        from yggdrasil.io.holder import (
            _coerce_transfer_endpoint,
            _read_slice_from_source,
        )

        source = _coerce_transfer_endpoint(src)
        self.write_bytes(
            _read_slice_from_source(source, size=size, offset=offset),
        )
        return self

    def download(
        self, to: Any = None, *, size: int = -1, offset: int = 0,
    ) -> "Holder | IO":
        """Push bytes from this IO (cursor-relative) into *to*.

        Cursor-anchored counterpart to :meth:`Holder.download`.
        ``offset`` is **relative to the current cursor**:
        ``offset=0`` reads from the cursor's current position,
        ``offset=N`` skips N bytes forward first. ``size=-1``
        reads to EOF, ``size>=0`` caps the count. The cursor
        advances past the bytes that were actually read.

        When *to* is :data:`None`, bytes land in
        ``~/Downloads/<name>`` (browser-style; falls back to
        ``"download"`` for cursorless / nameless holders).
        Otherwise *to* accepts the same shapes as
        :meth:`Holder.download` (:class:`Holder`, :class:`IO`,
        ``str`` / :class:`os.PathLike`). Returns the resolved
        target.
        """
        from yggdrasil.io.holder import (
            _coerce_transfer_endpoint,
            _default_download_target,
            _join_dir_hint,
            _read_slice_from_source,
        )

        # ``offset`` on self is cursor-relative — the helper does the
        # absolute seek + read pair, advancing the cursor past the
        # bytes it consumed.
        payload = _read_slice_from_source(
            self, size=size, offset=self._pos + offset,
        )

        if to is None:
            to = _default_download_target(self._transfer_filename())
        target = _join_dir_hint(_coerce_transfer_endpoint(to), self)
        target.write_bytes(payload)
        return target

    def _transfer_filename(self) -> str:
        """Filename used when joining onto a directory-shaped target.

        Delegates to the bound :class:`Holder` so an IO over a
        :class:`LocalPath` carries the path's basename through
        ``cp``-style directory targets. IO cursors over
        :class:`Memory` fall back to ``"download"`` like the
        holder side.
        """
        if self._holder is None:
            return "download"
        return self._holder._transfer_filename()

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def truncate(self, size: Optional[int] = None) -> int:
        if size is None:
            size = self._pos
        size = int(size)
        active = self._active()
        n = active.truncate(size)
        if self._pos > n:
            self._pos = n
        return n

    def flush(self) -> None:
        """Forward the flush to the bound holder.

        Backends with deferred-write semantics (remote multipart
        uploads, transactional sinks) commit pending state on
        :meth:`Holder.flush`; in-memory holders treat this as a
        cheap no-op.
        """
        try:
            self._holder.flush()
        except Exception:
            pass

    def close(self, force: bool = False) -> None:
        """Close the IO; closes the holder iff :attr:`owns_holder`.

        Preserves the cursor position across the close. A reopen on
        the same instance lands at the byte the previous transaction
        left off — callers that want a fresh start ``seek(0)``
        explicitly. (The historical behavior — silently resetting to
        byte 0 inside ``close`` — broke the ArrowIPC append flow,
        where a writer opens the buffer in append mode at EOF, drains
        bytes, and a subsequent read needs to start from byte 0 only
        because the reader explicitly seeks there.)
        """
        super().close(force=force)

    def _commit_metadata(self) -> None:
        """Refresh the holder's :class:`IOStats` after a bulk write.

        Bulk writers route through ``options.sync_metadata=False`` for
        the inner per-batch call so each ``write_mv`` skips its
        post-write ``_touch_stat``. This single call at the end stamps
        a fresh ``mtime`` and flushes any buffered backend state — one
        ``time.time()`` (and one optional flush) per write op instead
        of one per batch.
        """
        holder = getattr(self, "_holder", None)
        if holder is None:
            return
        try:
            holder.touch_mtime()
        except AttributeError:
            pass
        try:
            holder.flush()
        except Exception:
            pass

    # ==================================================================
    # Convenience drains — cursorless pass-throughs to the holder
    # ==================================================================

    def to_bytes(self) -> bytes:
        """Whole-payload snapshot from the active holder. Cursor untouched."""
        return self._active().to_bytes()

    def getvalue(self) -> bytes:
        """Stdlib-compatible alias for :meth:`to_bytes`."""
        return self._active().getvalue()

    def decode(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        """Decode the whole payload as text. Cursor untouched."""
        return self._active().decode(encoding, errors=errors)

    def to_base64(self, urlsafe: bool = True) -> str:
        return self._active().to_base64(urlsafe=urlsafe)

    # ==================================================================
    # Iteration
    # ==================================================================

    def __iter__(self) -> "IO":
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    # ==================================================================
    # Structured binary I/O — cursor-anchored wrappers over Holder
    # ==================================================================
    #
    # Every primitive defers to the cursorless equivalent on the
    # bound holder. The holder reads / writes at ``self._pos`` and
    # the IO advances the cursor by the fixed-width count.

    def _read_struct(self, method: str, width: int) -> Any:
        value = getattr(self._active(), method)(offset=self._pos)
        self._pos += width
        return value

    def _write_struct(self, method: str, value: Any, width: int) -> int:
        getattr(self._active(), method)(value, offset=self._pos)
        self._pos += width
        return width

    def read_int8(self) -> int: return self._read_struct("read_int8", 1)
    def write_int8(self, v: int) -> int: return self._write_struct("write_int8", v, 1)
    def read_uint8(self) -> int: return self._read_struct("read_uint8", 1)
    def write_uint8(self, v: int) -> int: return self._write_struct("write_uint8", v, 1)
    def read_int16(self) -> int: return self._read_struct("read_int16", 2)
    def write_int16(self, v: int) -> int: return self._write_struct("write_int16", v, 2)
    def read_uint16(self) -> int: return self._read_struct("read_uint16", 2)
    def write_uint16(self, v: int) -> int: return self._write_struct("write_uint16", v, 2)
    def read_int32(self) -> int: return self._read_struct("read_int32", 4)
    def write_int32(self, v: int) -> int: return self._write_struct("write_int32", v, 4)
    def read_uint32(self) -> int: return self._read_struct("read_uint32", 4)
    def write_uint32(self, v: int) -> int: return self._write_struct("write_uint32", v, 4)
    def read_int64(self) -> int: return self._read_struct("read_int64", 8)
    def write_int64(self, v: int) -> int: return self._write_struct("write_int64", v, 8)
    def read_uint64(self) -> int: return self._read_struct("read_uint64", 8)
    def write_uint64(self, v: int) -> int: return self._write_struct("write_uint64", v, 8)
    def read_f32(self) -> float: return self._read_struct("read_f32", 4)
    def write_f32(self, v: float) -> int: return self._write_struct("write_f32", v, 4)
    def read_f64(self) -> float: return self._read_struct("read_f64", 8)
    def write_f64(self, v: float) -> int: return self._write_struct("write_f64", v, 8)
    def read_bool(self) -> bool: return bool(self.read_uint8())
    def write_bool(self, v: bool) -> int: return self.write_uint8(1 if v else 0)

    def read_bytes_u32(self) -> bytes:
        """Length-prefixed (uint32 LE) bytes blob."""
        n = self.read_uint32()
        data = self.read(n)
        if len(data) != n:
            raise EOFError(f"expected {n} bytes, got {len(data)}")
        return data

    def write_bytes_u32(self, data: BytesLike) -> int:
        mv = memoryview(data)
        return self.write_uint32(len(mv)) + self.write_bytes(mv)

    def read_str_u32(self, encoding: str = "utf-8") -> str:
        """Length-prefixed UTF-8 string."""
        return self.read_bytes_u32().decode(encoding)

    def write_str_u32(self, s: str, encoding: str = "utf-8") -> int:
        return self.write_bytes_u32(s.encode(encoding))

    # ==================================================================
    # Convenience: parse / decompress
    # ==================================================================

    def json_load(self, *, media_type: Any = None, orient: Any = None) -> Any:
        """Parse the buffer, auto-detecting media type and compression.

        Resolution order for the media type:

        1. Explicit *media_type* kwarg.
        2. Cached :attr:`media_type` on the holder.
        3. Magic-byte sniff via :meth:`MediaType.from_io` — when this
           fires and the holder had no cached media type, the sniffed
           value is stamped onto the holder so future callers (codec
           handling, tabular dispatch) see it without re-sniffing.

        If the resolved type carries a codec the buffer is
        decompressed first and the inner mime is stamped onto the
        decompressed buffer. JSON / NDJSON / opaque-bytes payloads go
        through ``json.loads`` (or ``pandas.read_json`` when *orient*
        is set); every other registered format dispatches to its
        :class:`Tabular` leaf and returns ``read_pylist()``.
        """
        import json as _json
        from yggdrasil.data.enums.media_type import MediaType
        from yggdrasil.data.enums.mime_type import MimeTypes

        mt = (
            MediaType.from_(media_type, default=None)
            if media_type is not None else None
        )
        if mt is None:
            mt = self.media_type
            cached = mt is not None
        else:
            cached = True

        if mt is None:
            mt = MediaType.from_io(self, default=None)

        if mt is not None and not cached:
            try:
                self._holder.media_type = mt
            except Exception:
                pass

        if mt is not None and mt.codec is not None:
            buf = mt.codec.decompress(self)
            inner_mt = MediaType(mime_type=mt.mime_type, codec=None)
            try:
                buf._holder.media_type = inner_mt
            except Exception:
                pass
            mt = inner_mt
        else:
            buf = self

        mime = mt.mime_type if mt is not None else None
        is_jsonlike = (
            mime is None
            or mime is MimeTypes.JSON
            or mime.is_any_bytes
        )

        if is_jsonlike:
            text = buf.to_bytes().decode("utf-8", errors="replace")
            if not text.strip():
                return None
            if orient is not None:
                try:
                    from yggdrasil.lazy_imports import pandas as pd
                    return pd.read_json(text, orient=orient)
                except Exception:
                    pass
            return _json.loads(text)

        import yggdrasil.io.primitive  # noqa: F401  -- register leaves
        from yggdrasil.io.tabular.base import Tabular
        leaf_cls = Tabular.class_for_media_type(mt, default=None)
        if leaf_cls is None:
            text = buf.to_bytes().decode("utf-8", errors="replace")
            if not text.strip():
                return None
            return _json.loads(text)
        leaf = (
            buf if isinstance(buf, leaf_cls)
            else leaf_cls(holder=buf._holder, owns_holder=False)
        )
        return leaf.read_pylist()

    def decompress(self, *, codec: Any = None, copy: bool = True) -> "IO":
        """Return a new IO over the decompressed payload.

        ``codec`` may be a :class:`Codec`, a codec name (``"gzip"``,
        ``"zstd"``, …), or a :class:`MediaType`-shaped object whose
        ``codec`` attribute is read. Returns the original buffer when
        no codec is set / supplied.
        """
        if codec is None:
            codec_obj = self._codec()
        else:
            inner = getattr(codec, "codec", None)
            if inner is not None:
                codec_obj = inner
            else:
                from yggdrasil.data.enums.codec import Codec
                codec_obj = Codec.from_(codec, default=None)
        if codec_obj is None:
            if copy:
                return type(self)(self.to_bytes())
            return self
        return codec_obj.decompress(self)

    def xxh3_64(self):
        """Return an :class:`xxhash.xxh3_64` instance over the payload."""
        import xxhash
        return xxhash.xxh3_64(self.to_bytes())

    def xxh3_int64(self) -> int:
        """64-bit xxh3 hash of the buffer's payload as a signed int64.

        ``xxh3_64`` itself produces an unsigned 64-bit value;
        downstream Arrow schemas pin the field as ``int64``, so we
        wrap into signed range ``[-2**63, 2**63)`` here.
        """
        import xxhash
        v = xxhash.xxh3_64(self.to_bytes()).intdigest()
        if v >= 2 ** 63:
            v -= 2 ** 64
        return v


# ===========================================================================
# Codec writer / reader context managers
# ===========================================================================


class _FormatBufferContext:
    """Writer-side of :meth:`IO._format_buffer`.

    Caller does ``with bio._format_buffer() as buf: writer(buf)``;
    the yielded ``buf`` accepts raw format bytes. On exit:

    * No codec → ``buf is bio``; we just leave the bytes in place.
      (We pre-truncate so the writer sees an empty target.)
    * Codec set → ``buf`` is a fresh in-memory IO; on exit the bytes
      are compressed and committed to ``bio``.
    """

    def __init__(self, parent: "IO") -> None:
        self._parent = parent
        self._buf: "IO | None" = None
        self._codec = parent._codec()

    def __enter__(self) -> "IO":
        if self._codec is None:
            # Direct write path: pre-truncate so the leaf writer
            # opens onto an empty target.
            self._parent.seek(0)
            self._parent.truncate(0)
            self._buf = self._parent
            return self._parent
        # Codec path: scratch buffer; we compress on exit.
        self._buf = type(self._parent)()
        return self._buf

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._buf is None or exc_type is not None:
            return
        if self._codec is None:
            return
        # Compress scratch into the durable buffer.
        compressed = self._codec.compress(self._buf)
        try:
            payload = compressed.to_bytes()
        finally:
            try:
                compressed.close()
            except Exception:
                pass
        try:
            self._buf.close()
        except Exception:
            pass
        self._parent.seek(0)
        self._parent.truncate(0)
        self._parent.write(payload)


class _FormatInputContext:
    """Reader-side companion to :class:`_FormatBufferContext`.

    Resolves the cheapest pyarrow-friendly input source for the
    formatted bytes:

    - Local-path holder with no codec → :func:`pyarrow.memory_map`.
      The file lands in the kernel page cache once and every reader
      (Parquet, Arrow IPC, CSV, NDJSON) walks it without a copy.
    - Anything else → :meth:`IO._format_view` (a non-owning view of
      ``self`` when uncompressed, a decompressed in-memory IO when a
      codec is bound).

    The object yielded by ``__enter__`` is closed on ``__exit__`` —
    callers don't have to track which branch fired.
    """

    def __init__(self, parent: "IO") -> None:
        self._parent = parent
        self._mm: "Any | None" = None
        self._view: "IO | None" = None

    def __enter__(self) -> "Any":
        if self._parent._codec() is None:
            holder = self._parent._holder
            if holder is not None and getattr(holder, "is_local_path", False):
                full_path = getattr(holder, "full_path", None)
                if callable(full_path):
                    try:
                        self._mm = pa.memory_map(full_path(), "r")
                        return self._mm
                    except Exception:
                        # Fall through to the view; mmap failures
                        # (race with delete, fs that doesn't support
                        # mmap, sandbox restrictions) shouldn't break
                        # the read.
                        self._mm = None
        self._view = self._parent._format_view()
        return self._view

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if self._view is not None:
            try:
                self._view.close()
            except Exception:
                pass
            self._view = None


class _ArrowInputStreamContext:
    """Reader-side companion for :meth:`IO.arrow_input_stream`.

    Yields a real :class:`pa.NativeFile` over the buffer's payload,
    transparently decompressing first when the holder's media type
    carries a codec. Resolution:

    - Local-path holder + no codec → :func:`pyarrow.memory_map`
      (zero-copy :class:`pa.MemoryMappedFile`). The kernel pages the
      file in once and every reader walks it without a Python copy.
    - Codec-tagged holder → :meth:`Codec.decompress` into a scratch
      in-memory IO; the uncompressed bytes are then handed to a
      :class:`pa.BufferReader`.
    - Anything else → snapshot :meth:`IO.to_bytes` and wrap in a
      :class:`pa.BufferReader`.

    The stream and any scratch decompression buffer are closed on
    ``__exit__``.
    """

    def __init__(self, parent: "IO") -> None:
        self._parent = parent
        self._stream: "pa.NativeFile | None" = None
        self._scratch: "IO | None" = None

    def __enter__(self) -> "pa.NativeFile":
        parent = self._parent
        codec = parent._codec()

        if codec is None:
            holder = parent._holder
            if holder is not None and getattr(holder, "is_local_path", False):
                full_path = getattr(holder, "full_path", None)
                if callable(full_path):
                    try:
                        self._stream = pa.memory_map(full_path(), "r")
                        return self._stream
                    except Exception:
                        # mmap can fail on sandboxed filesystems or if
                        # the file was deleted under us; fall back to
                        # the bytes snapshot path rather than escalate.
                        self._stream = None
            self._stream = pa.BufferReader(parent.to_bytes())
            return self._stream

        # Codec path — decompress through the codec's streaming
        # roundtrip, then expose the uncompressed bytes as a NativeFile.
        self._scratch = codec.decompress(parent)
        self._stream = pa.BufferReader(self._scratch.to_bytes())
        return self._stream

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._scratch is not None:
            try:
                self._scratch.close()
            except Exception:
                pass
            self._scratch = None


class _ArrowOutputStreamContext:
    """Writer-side companion for :meth:`IO.arrow_output_stream`.

    Yields a :class:`pa.NativeFile` so format encoders
    (:class:`pa.ipc.RecordBatchFileWriter`,
    :func:`pa.parquet.ParquetWriter`, :func:`pa.csv.CSVWriter`, …) can
    stream directly into the cheapest available sink:

    - **Local-path holder, no codec** → :func:`pyarrow.OSFile` opened
      against the holder's filesystem path. The encoder writes pages
      straight to disk; no Python-side copy through an in-memory
      buffer. ``append=False`` truncates first; ``append=True`` opens
      with ``"ab"``.
    - **Anything else** (in-memory holder, remote path, codec-tagged
      buffer) → :class:`pa.BufferOutputStream`. On clean exit the
      accumulated bytes are bulk-committed to the parent IO through
      :meth:`IO._commit_format_payload`, which handles codec
      compression and the overwrite-vs-append disposition.

    On exception the sink is closed and the parent is left untouched
    — the caller's prior payload is not overwritten by a half-written
    encoder run.
    """

    def __init__(self, parent: "IO", *, append: bool = False) -> None:
        self._parent = parent
        self._append = bool(append)
        self._sink: "pa.NativeFile | None" = None
        # ``True`` when the sink writes straight to a local file —
        # nothing left to commit on exit; ``False`` when the sink is
        # an in-memory buffer that still needs a commit.
        self._direct: bool = False

    def __enter__(self) -> "pa.NativeFile":
        parent = self._parent
        if parent._codec() is None:
            holder = parent._holder
            if holder is not None and getattr(holder, "is_local_path", False):
                full_path = getattr(holder, "full_path", None)
                if callable(full_path):
                    try:
                        path_str = full_path()
                        # ``OSFile`` doesn't accept ``"ab"`` directly;
                        # we open in ``"wb"`` and seek to EOF for
                        # append, matching :meth:`_commit_format_payload`'s
                        # disposition.
                        if self._append:
                            self._sink = pa.OSFile(path_str, "ab")
                        else:
                            self._sink = pa.OSFile(path_str, "wb")
                        self._direct = True
                        # Local writes bypass the Python-side cursor, so
                        # invalidate the holder's cached stat — the next
                        # ``size`` / ``mtime`` probe re-reads from disk.
                        invalidate = getattr(
                            holder, "invalidate_singleton", None,
                        )
                        if callable(invalidate):
                            invalidate()
                        return self._sink
                    except Exception:
                        # Fall through to the in-memory path; OSFile
                        # failures (sandbox, missing parent dir, mode
                        # mismatch) shouldn't break the write.
                        self._sink = None
                        self._direct = False
        self._sink = pa.BufferOutputStream()
        self._direct = False
        return self._sink

    def __exit__(self, exc_type, exc, tb) -> None:
        sink = self._sink
        self._sink = None
        direct = self._direct
        self._direct = False
        if sink is None:
            return
        if exc_type is not None:
            try:
                sink.close()
            except Exception:
                pass
            return
        if direct:
            # Bytes already on disk — just close the file handle and
            # invalidate the cached stat one more time so the next
            # reader sees the post-write size.
            try:
                sink.close()
            except Exception:
                pass
            holder = self._parent._holder
            if holder is not None:
                invalidate = getattr(holder, "invalidate_singleton", None)
                if callable(invalidate):
                    invalidate()
            return
        try:
            payload = sink.getvalue()
        finally:
            try:
                sink.close()
            except Exception:
                pass
        self._parent._commit_format_payload(payload, append=self._append)
