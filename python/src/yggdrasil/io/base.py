"""Cursor + tabular handle bound to a managed :class:`Holder`.

:class:`IO[T, O]` IS-A :class:`Tabular`. It carries the cursor a
holder lacks (every public op is "do this at the cursor"), implements
the bytes surface (``read`` / ``write`` / ``seek`` / ``tell`` /
positional reads / structured primitives / codec-aware format helpers)
on top of a :class:`Holder`, and inherits the row-oriented surface
from :class:`Tabular` so byte-level and Arrow-level operations live
on the same handle.

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

Lifecycle — closed = direct, open = buffered transaction
--------------------------------------------------------

A :class:`IO` has two operating modes, switched by its
:class:`Disposable` state:

- **Closed** — every read / write goes **straight to the durable
  holder**. No buffering. Useful when you want each op to commit
  immediately (logging, append-only sinks, throwaway one-shots).
  The durable holder must already be acquired by someone — typically
  the caller.

- **Open** — a :class:`Memory` scratch buffer sits between the cursor
  and the holder. :meth:`_acquire` seeds scratch from the durable
  bytes (or empty for OVERWRITE-class modes). All ops route through
  scratch; the durable holder is **untouched** until :meth:`flush`
  or :meth:`_release` commits.

The dirty bit drives commit decisions: writes through the open
IO call :meth:`mark_dirty`; :meth:`flush` and :meth:`_release` check
:attr:`_dirty` before pushing scratch onto the durable holder.
Setting ``temporary=True`` on the holder discards the scratch on
close instead of committing.

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

The mode never touches the holder itself — a :class:`Memory` doesn't
know or care what mode the IO over it is in.
"""

from __future__ import annotations

import base64
import io as _stdlib_io
import os
import pathlib
import struct
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
                mt = getattr(holder.stat(), "media_type", None)
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

    Two operating modes:

    - **Closed** — ops route directly to the durable holder. Each
      write commits synchronously; useful for append-only sinks or
      one-shot reads where buffering would be overhead.
    - **Open** — :meth:`_acquire` builds a :class:`Memory` scratch
      seeded from the durable holder; ops route through scratch,
      the durable holder is committed-to on :meth:`flush` /
      :meth:`_release`. The standard "edit in memory, save on
      close" transaction.

    Mode-aware but format-agnostic at this layer. Mode controls
    cursor position (:data:`Mode.APPEND` → EOF), :meth:`readable` /
    :meth:`writable` reporting, whether :meth:`_acquire` starts
    scratch empty (:data:`Mode.OVERWRITE`) or seeds from durable bytes,
    and the :class:`FileExistsError` for :data:`Mode.ERROR_IF_EXISTS`.

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
        "_scratch",
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
        a :class:`Mode` enum. The IO is constructed in the **closed**
        state — ops in that state commit synchronously to the durable
        holder. To enter the buffered-transaction mode (scratch buffer
        + commit-on-close), call :meth:`acquire`, use a ``with`` block,
        or go through :meth:`Holder.open`.
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

        # Memory scratch buffer — populated by :meth:`_acquire`,
        # commits to ``self._holder`` on :meth:`flush` /
        # :meth:`_release`. ``None`` when the IO is closed; in that
        # state, ops route directly to ``self._holder``.
        self._scratch: "Holder | None" = None

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

    # ==================================================================
    # Disposable lifecycle — open is "buffered transaction"
    # ==================================================================

    def _acquire(self) -> None:
        """Open a buffered transaction over the holder.

        Builds a :class:`Memory` scratch holder seeded from the
        durable holder's bytes (or empty for OVERWRITE-class modes
        that truncate at open). All read / write ops while open
        route through scratch — the durable holder is untouched
        until :meth:`flush` or :meth:`_release` commits.

        Mode side-effects:

        - :data:`Mode.OVERWRITE` / :data:`Mode.TRUNCATE` — scratch
          starts empty (the durable truncate happens at commit time).
        - :data:`Mode.APPEND` — scratch seeded from durable bytes,
          cursor parked at EOF.
        - :data:`Mode.ERROR_IF_EXISTS` — fail-fast
          :class:`FileExistsError` if the durable holder is non-empty.
        - :data:`Mode.READ_ONLY` / :data:`Mode.AUTO` / default —
          scratch seeded from durable bytes, cursor at 0.

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

        from yggdrasil.io.memory import Memory
        scratch = Memory()
        scratch.acquire()

        # Seed scratch from the durable holder for read-friendly
        # modes; OVERWRITE / TRUNCATE start clean (truncate-on-commit).
        if (
            self._mode not in (Mode.OVERWRITE, Mode.TRUNCATE)
            and self._holder.size > 0
        ):
            scratch.pwrite(self._holder.read_bytes(), 0)

        self._scratch = scratch
        self._pos = scratch.size if self._mode.appendable else 0

    def _commit(self) -> None:
        """Push the scratch buffer's bytes onto the durable holder.

        :class:`Disposable._close` calls :meth:`commit` (which routes
        here when :attr:`_dirty`) before :meth:`_release`, so by the
        time :meth:`_release` runs the durable holder already has the
        new bytes. ``temporary`` holders skip this — :meth:`_release`
        on the holder side drops the payload.
        """
        scratch = self._scratch
        if scratch is None:
            return
        if getattr(self._holder, "temporary", False):
            return
        self._commit_scratch(scratch)

    def _release(self) -> None:
        """Tear down scratch and release the durable holder if owned.

        Commit is :meth:`_commit`'s job — this method is pure cleanup.
        """
        scratch = self._scratch
        if scratch is not None:
            try:
                scratch.close()
            except Exception:
                pass
            self._scratch = None

        if self._owns_holder:
            try:
                self._holder.close()
            except Exception:
                pass

    def _commit_scratch(self, scratch: "Holder") -> None:
        """Push *scratch*'s bytes onto the durable holder.

        Truncate-then-rewrite: the simplest correct shape that
        preserves "what's in scratch" exactly. Backends with cheaper
        partial-update primitives (a remote multipart upload, a
        Delta append) override at the holder level — this is the
        universal fallback.
        """
        size = scratch.size
        self._holder.truncate(size)
        if size > 0:
            self._holder.pwrite(scratch.read_bytes(), 0)
        self._holder.flush()

    def _active(self) -> "Holder":
        """The holder ops should hit right now — scratch when open,
        durable otherwise.

        Centralizing the dispatch means every ``read`` / ``write`` /
        ``truncate`` / ``size`` call site stays unchanged whether
        the IO is in a transaction or not.
        """
        return self._scratch if self._scratch is not None else self._holder

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
            mt = holder.stat().media_type
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
        path-backed holders without touching the IO open / scratch /
        commit machinery.
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

        n = len(view)
        if append:
            self.seek(0, 2)  # SEEK_END
        else:
            self.seek(0)
            self.truncate(0)
        if n > 0:
            self.write_bytes(view)
        return n

    @property
    def size(self) -> int:
        """Live size from the active holder — scratch when open,
        durable otherwise."""
        return self._active().size

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        return True

    def __bytes__(self) -> bytes:
        """Snapshot the active payload as :class:`bytes`."""
        return self.to_bytes()

    def __repr__(self) -> str:
        state = "open" if self._acquired else "closed"
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

        Convenience over ``self._holder.stat().media_type`` — same
        thing the codec auto-handling reads.
        """
        try:
            return self._holder.stat().media_type
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
                mt = self._holder.stat().media_type
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
        """Stdlib ``IO[bytes]`` parity — ``False`` while the durable
        holder is reachable.

        Stdlib semantics: ``closed`` means "file unusable for I/O."
        Our holders are reachable in both the closed-Disposable
        (direct synchronous commit) and open-Disposable (scratch
        transaction) states, so the property only flips when a
        ``close(force=True)`` has actually run teardown. This matters
        for pyarrow / pandas / polars / zipfile, which guard every op
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
        """Positional write against the active holder. Cursor untouched.

        Marks the IO dirty when scratch is active so the commit on
        :meth:`flush` / :meth:`_release` actually fires.
        ``update_stat=False`` is forwarded to the holder so a bulk
        loop can skip the per-write stat refresh.
        """
        n = self._active().pwrite(data, pos, update_stat=update_stat)
        if n > 0 and self._scratch is not None:
            self.mark_dirty()
        return n

    def memoryview(self) -> memoryview:
        """Memoryview over the active holder's full payload."""
        return self._active().memoryview()

    # ==================================================================
    # IO[bytes] core
    # ==================================================================

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self.size - self._pos)
        if size is None or size < 0:
            size = remaining
        else:
            # Cap to remaining bytes — stdlib ``IOBase.read`` returns
            # fewer than *size* when EOF is reached, so we do the same
            # rather than asking the holder for an out-of-range slice.
            size = min(size, remaining)
        if size == 0:
            return b""
        out = self._active().pread(size, self._pos)
        self._pos += len(out)
        return out

    def readall(self) -> bytes:
        """Read from cursor to EOF, advancing the cursor."""
        return self.read(-1)

    def readinto(self, b: Any) -> int:
        mv = memoryview(b)
        n = len(mv)
        if n == 0:
            return 0
        chunk = self._active().pread(n, self._pos)
        got = len(chunk)
        if got:
            mv[:got] = chunk
            self._pos += got
        return got

    def readinto1(self, b: Any) -> int:
        return self.readinto(b)

    def readline(self, limit: int = -1) -> bytes:
        size = self.size
        if self._pos >= size:
            return b""
        if limit is None or limit < 0:
            chunk_len = size - self._pos
        else:
            chunk_len = min(limit, size - self._pos)
        if chunk_len <= 0:
            return b""

        chunk = self._active().pread(chunk_len, self._pos)
        nl = chunk.find(b"\n")
        if nl == -1:
            self._pos += len(chunk)
            return chunk

        line = chunk[: nl + 1]
        self._pos += len(line)
        return line

    def readlines(self, hint: int = -1) -> list[bytes]:
        lines: list[bytes] = []
        total = 0
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            total += len(line)
            if hint is not None and hint > 0 and total >= hint:
                break
        return lines

    def write(self, b: Any, *, update_stat: bool = True) -> int:
        """Write *b* at the cursor, advancing it.

        Accepts bytes-like, ``str`` (UTF-8), ``io.BytesIO``, or any
        file-like with ``.read``. The buffer-protocol fallback catches
        things like :class:`pyarrow.Buffer` that aren't
        bytes/bytearray/memoryview but ARE memoryview-able.
        """
        if b is None:
            return 0
        if isinstance(b, str):
            return self.write_bytes(b.encode("utf-8"), update_stat=update_stat)
        if isinstance(b, (bytes, bytearray, memoryview)):
            return self.write_bytes(b, update_stat=update_stat)
        if hasattr(b, "read"):
            total = 0
            while True:
                chunk = b.read(1024 * 1024)
                if not chunk:
                    break
                total += self.write_bytes(chunk, update_stat=update_stat)
            return total
        return self.write_bytes(memoryview(b), update_stat=update_stat)

    def write_bytes(self, b: BytesLike, *, update_stat: bool = True) -> int:
        mv = _as_byte_mv(b)
        if len(mv) == 0:
            return 0
        n = self._active().pwrite(mv, self._pos, update_stat=update_stat)
        self._pos += n
        if n > 0 and self._scratch is not None:
            self.mark_dirty()
        return n

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def truncate(self, size: Optional[int] = None) -> int:
        if size is None:
            size = self._pos
        size = int(size)
        active = self._active()
        prev = active.size
        n = active.truncate(size)
        if self._pos > n:
            self._pos = n
        if n != prev and self._scratch is not None:
            self.mark_dirty()
        return n

    def flush(self) -> None:
        """Push scratch buffer to the durable holder.

        When open: commit the scratch buffer's bytes onto the durable
        holder, then clear the dirty bit. The scratch stays — further
        writes continue to buffer.
        When closed: no-op (every closed-mode op already committed
        synchronously through the durable holder).
        """
        scratch = self._scratch
        if scratch is None:
            return
        if self._dirty:
            self._commit_scratch(scratch)
            self.clear_dirty()

    def close(self, force: bool = False) -> None:
        """Close the IO; closes the holder iff :attr:`owns_holder`."""
        super().close(force=force)

    # ==================================================================
    # Convenience drains
    # ==================================================================

    def to_bytes(self) -> bytes:
        """Whole-payload snapshot from the active holder. Cursor untouched."""
        return self._active().read_bytes()

    def getvalue(self) -> bytes:
        """Stdlib-compatible alias for :meth:`to_bytes`."""
        return self.to_bytes()

    def decode(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        """Decode the whole payload as text. Cursor untouched."""
        return self.to_bytes().decode(encoding, errors=errors)

    def to_base64(self, urlsafe: bool = True) -> str:
        b = self.to_bytes()
        if urlsafe:
            return base64.urlsafe_b64encode(b).decode("ascii")
        return base64.b64encode(b).decode("ascii")

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
    # Structured binary I/O — fixed-width little-endian primitives
    # ==================================================================

    def _read_exact(self, n: int) -> bytes:
        data = self.read(n)
        if len(data) != n:
            raise EOFError(f"expected {n} bytes, got {len(data)}")
        return data

    def read_int8(self) -> int: return struct.unpack("<b", self._read_exact(1))[0]
    def write_int8(self, v: int) -> int: return self.write_bytes(struct.pack("<b", int(v)))
    def read_uint8(self) -> int: return struct.unpack("<B", self._read_exact(1))[0]
    def write_uint8(self, v: int) -> int: return self.write_bytes(struct.pack("<B", int(v)))
    def read_int16(self) -> int: return struct.unpack("<h", self._read_exact(2))[0]
    def write_int16(self, v: int) -> int: return self.write_bytes(struct.pack("<h", int(v)))
    def read_uint16(self) -> int: return struct.unpack("<H", self._read_exact(2))[0]
    def write_uint16(self, v: int) -> int: return self.write_bytes(struct.pack("<H", int(v)))
    def read_int32(self) -> int: return struct.unpack("<i", self._read_exact(4))[0]
    def write_int32(self, v: int) -> int: return self.write_bytes(struct.pack("<i", int(v)))
    def read_uint32(self) -> int: return struct.unpack("<I", self._read_exact(4))[0]
    def write_uint32(self, v: int) -> int: return self.write_bytes(struct.pack("<I", int(v)))
    def read_int64(self) -> int: return struct.unpack("<q", self._read_exact(8))[0]
    def write_int64(self, v: int) -> int: return self.write_bytes(struct.pack("<q", int(v)))
    def read_uint64(self) -> int: return struct.unpack("<Q", self._read_exact(8))[0]
    def write_uint64(self, v: int) -> int: return self.write_bytes(struct.pack("<Q", int(v)))
    def read_f32(self) -> float: return struct.unpack("<f", self._read_exact(4))[0]
    def write_f32(self, v: float) -> int: return self.write_bytes(struct.pack("<f", float(v)))
    def read_f64(self) -> float: return struct.unpack("<d", self._read_exact(8))[0]
    def write_f64(self, v: float) -> int: return self.write_bytes(struct.pack("<d", float(v)))
    def read_bool(self) -> bool: return bool(self.read_uint8())
    def write_bool(self, v: bool) -> int: return self.write_uint8(1 if v else 0)

    def read_bytes_u32(self) -> bytes:
        """Length-prefixed (uint32 LE) bytes blob."""
        return self._read_exact(self.read_uint32())

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
