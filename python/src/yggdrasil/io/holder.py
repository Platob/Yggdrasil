"""Byte holder + cursor + tabular handle — the IO substrate.

An :class:`IO` is "a thing that holds N bytes addressable by
position" plus an opt-in seekable cursor and the stdlib
:class:`typing.BinaryIO` surface on top. Three layered shapes share
the class:

- :class:`yggdrasil.io.memory.Memory` — a :class:`bytearray` we
  manage directly. Every read/write hits memory.
- :class:`yggdrasil.io.path.Path` subclasses (``LocalPath``, remote
  paths) — path-bound storage. Local paths back the storage with a
  long-lived :func:`os.open` fd; remote paths with a transaction
  buffer flushed on commit.
- Format leaves (:class:`ParquetFile`, :class:`CSVFile`,
  :class:`XLSXFile`, …) and plain cursors — IO instances that
  hold no storage of their own. They carry a ``_parent`` pointer
  to a backing storage IO and delegate every byte primitive
  through :meth:`_active`.

The five storage primitives are :meth:`_read_mv`, :meth:`_write_mv`,
:meth:`reserve`, :meth:`truncate`, :meth:`_clear` and the :attr:`size`
property. Storage subclasses (Memory, Path) implement them directly;
cursor/format subclasses inherit the delegating defaults that hand
the call to :meth:`_active`. Everything else (:meth:`pread` /
:meth:`pwrite` / :meth:`read_bytes` / :meth:`write_bytes` /
:meth:`read_text` / :meth:`write_text` / :meth:`write_local_path`)
builds on those, so a new backend gets the full convenience surface
for free.

The default way to interact with a storage IO's bytes is via
:meth:`open`, which returns a fresh :class:`IO` cursor over the
storage — a seekable handle that is also a
:class:`yggdrasil.tabular.Tabular`, so reading the holder as Arrow
record batches is the same call::

    with LocalPath("data.parquet").open() as bio:
        table = bio.read_arrow_table()

For lifecycle without the cursor wrapper, use :meth:`acquire` /
:meth:`close`. Multiple cursors can borrow one storage IO, each
with its own ``_pos``; see :meth:`open` for patterns.
"""

from __future__ import annotations

import os
import pathlib
import struct
import time
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    ClassVar,
    Generic,
    IO as _StdlibIO,
    Iterator,
    Optional,
    TypeVar,
    Union,
)

import pyarrow as pa

from yggdrasil.data.enums import MediaType, MimeType
from yggdrasil.data.enums.mode import Mode, ModeLike
from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.disposable import Disposable
from yggdrasil.io.tabular.base import O, Tabular
from .io_stats import IOStats
from .url import URL, URLBased

if TYPE_CHECKING:
    pass

__all__ = ["IO", "Holder", "BytesLike", "T", "O"]

T = TypeVar("T")

BytesLike = Union[bytes, bytearray, memoryview]


PathLike = Union[str, "os.PathLike[str]", pathlib.PurePath]


_COPY_CHUNK = 1024 * 1024

#: Byte threshold under which :meth:`Holder._write_holder` writes
#: the source's full payload in one :meth:`write_mv` call — above
#: this, the default opens a cursor and streams chunks through
#: :meth:`_write_stream`. 4 MiB is the natural break: small payloads
#: are cheap to materialise; large ones risk doubling peak RSS on
#: copy. Backends with an atomic uploader (Workspace, Volumes, S3)
#: override :meth:`_write_holder` and ignore this constant.
_INLINE_WRITE_THRESHOLD = 4 * 1024 * 1024


def _resolve_pos(pos: int, size: int) -> int:
    """Normalize a position argument with append-at-end semantics.

    - ``pos == -1`` is the explicit "at end of stream" sentinel and
      resolves to ``size`` (POSIX ``SEEK_END`` with offset 0). Reads
      from this position yield zero bytes; writes append.
    - Other negative values count from the end: ``-2`` → ``size - 2``,
      ``-3`` → ``size - 3``, etc. Note the one-step discontinuity at
      ``-1``: this is intentional, so callers have a stable append
      sentinel without giving up from-end indexing.
    - Non-negative values pass through unchanged.

    The result is **not** range-checked; callers do their own bounds
    checks against the operation they're about to perform.
    """
    if pos == -1:
        return size
    if pos < 0:
        return size + pos
    return pos


def _resolve_in_memory_tabular(data: Any) -> "type | None":
    """Return the concrete in-memory :class:`Tabular` for *data*, or ``None``.

    The byte-backed registry on :class:`IO` doesn't know how to wrap
    pure-data shapes (a :class:`pa.Table`, a Spark DataFrame, a
    polars / pandas frame, a ``list[dict]`` / ``dict[str, list]``):
    they're already materialised, never went through a wire format,
    and don't have a URL. This helper is what :meth:`IO.__new__`
    consults before falling through to the scheme/format dispatch so
    those shapes land on the right in-memory holder:

    - :class:`pyarrow.Table` / :class:`pa.RecordBatch` /
      :class:`pa.RecordBatchReader` → :class:`ArrowTabular`.
    - Spark :class:`pyspark.sql.DataFrame` → :class:`Dataset`
      (kept lazy on the executors — no driver collect).
    - polars / pandas frame → :class:`ArrowTabular` (Arrow is the
      narrow waist; the holder's ``_ingest`` knows the conversion).
    - ``list[dict]`` rows / ``dict[str, list]`` columns →
      :class:`ArrowTabular`.

    Returns ``None`` when *data* isn't a shape we recognise — the
    caller should fall through to its existing dispatch (a
    :class:`Memory` byte-holder, a path-backed storage leaf, etc.).
    Module-name sniffing keeps polars / pandas / pyspark out of the
    import graph until we've already confirmed an instance from one
    of those modules.
    """
    if data is None:
        return None

    # PyArrow shapes that :meth:`ArrowTabular._ingest` already handles
    # directly. The two non-Table shapes (RecordBatch /
    # RecordBatchReader) are intentional: callers building a stream
    # with ``pa.ipc.open_stream`` and handing the reader to
    # :class:`IO` get back an in-memory holder rather than a TypeError.
    if isinstance(data, (pa.Table, pa.RecordBatch, pa.RecordBatchReader)):
        from yggdrasil.io.tabular.arrow import ArrowTabular
        return ArrowTabular

    mod = (type(data).__module__ or "").split(".", 1)[0]

    if mod == "pyspark":
        # Spark DataFrame → keep lazy. Other pyspark types (Column,
        # GroupedData, Window …) aren't tabular sources we know how
        # to wrap; route DataFrames only.
        if "DataFrame" in type(data).__name__:
            from yggdrasil.io.tabular.spark import Dataset
            return Dataset
        return None

    if mod in ("polars", "pandas"):
        from yggdrasil.io.tabular.arrow import ArrowTabular
        return ArrowTabular

    # Pure-Python row-list / column-dict shapes. Match the same
    # guards :meth:`ArrowTabular._ingest` uses so the dispatch and
    # the ingest agree on what "I can handle this" means.
    if isinstance(data, list) and data and all(
        isinstance(r, dict) for r in data
    ):
        from yggdrasil.io.tabular.arrow import ArrowTabular
        return ArrowTabular
    if (
        isinstance(data, dict) and data
        and all(isinstance(v, (list, tuple)) for v in data.values())
    ):
        from yggdrasil.io.tabular.arrow import ArrowTabular
        return ArrowTabular

    return None


def _resolve_subclass(
    *,
    scheme: str | None = None,
    url: URL | None = None,
    binary: bytes | bytearray | memoryview | None = None,
    path: PathLike | None = None,
    data: Any = None,
) -> type["IO"]:
    """Pick the concrete :class:`IO` storage subclass for the given inputs.

    Pure routing — no instance allocation. Lives outside :meth:`__new__`
    so the dispatch is testable in isolation and so :meth:`__new__` can
    short-circuit ``cls is IO`` without nesting.
    """
    if url is not None:
        url_obj = URL.from_(url)
        scheme = url_obj.scheme or scheme

    if scheme:
        try:
            return URLBased.for_scheme(scheme)
        except (ValueError, ImportError) as exc:
            raise ValueError(f"Unknown scheme '{scheme}'") from exc

    if path is not None:
        # Resolve the path's URL scheme via the URLBased registry
        # (file:// → LocalPath, s3:// → S3Path, …). The abstract
        # :class:`Path` itself isn't instantiable, so a missing scheme
        # falls back to LocalPath — that's the only path-shaped backend
        # that's always available.
        from .path.local_path import LocalPath
        url_obj = URL.from_(path)
        scheme_from_path = url_obj.scheme
        if scheme_from_path:
            try:
                return URLBased.for_scheme(scheme_from_path)
            except (ValueError, ImportError):
                pass
        return LocalPath

    if isinstance(data, IO):
        return type(data)

    # binary, str, pathlib.Path, None, bytes-like — all default to memory
    from .memory import Memory
    return Memory


#: Format-leaf registry: :class:`MimeType` name → concrete
#: :class:`IO` subclass that owns it. Mirror of the
#: :data:`_URL_BASED_REGISTRY` (scheme → :class:`URLBased` subclass)
#: that lives on :class:`URLBased`. Populated lazily by
#: :meth:`IO.__init_subclass__` whenever a subclass declares a
#: concrete :attr:`mime_type`.
_HOLDER_FORMAT_REGISTRY: "dict[str, type[IO]]" = {}
_HOLDER_FORMAT_REGISTRY_BOOTSTRAPPED: bool = False


def _bootstrap_holder_format_registry() -> None:
    """Force-load every concrete format-leaf package once.

    Each leaf module registers its ``mime_type`` via
    :meth:`IO.__init_subclass__` on import, so importing the leaf
    packages is enough to populate :data:`_HOLDER_FORMAT_REGISTRY`.
    Idempotent — the module-level flag short-circuits repeat calls.
    """
    global _HOLDER_FORMAT_REGISTRY_BOOTSTRAPPED
    if _HOLDER_FORMAT_REGISTRY_BOOTSTRAPPED:
        return
    _HOLDER_FORMAT_REGISTRY_BOOTSTRAPPED = True
    import yggdrasil.io.primitive  # noqa: F401
    import yggdrasil.io.nested  # noqa: F401


def _resolve_format_target(
    cls: type,
    *,
    media_type: Any,
    path: Any,
    data: Any,
    holder: "IO | None",
) -> "type | None":
    """Resolve the registered format-leaf class for the given inputs.

    Resolution priority:

    1. Explicit *media_type* kwarg.
    2. *path* — extension via :meth:`URL.infer_media_type`.
    3. *data* — same, when it's URL-shaped (``str`` / ``pathlib.PurePath``
       / :class:`URL`); bytes-like and file-like inputs are skipped.
    4. *holder*'s stamped ``media_type``.

    Returns ``None`` when no media type can be resolved or no registered
    leaf exists for the resolved type. Uses
    :meth:`IO.class_for_media_type` for the registry lookup, which
    bootstraps the leaf packages on a cold miss.
    """
    mt = (
        MediaType.from_(media_type, default=None)
        if media_type is not None else None
    )

    if mt is None:
        for src in (path, data):
            if src is None or isinstance(src, (bytes, bytearray, memoryview)):
                continue
            if hasattr(src, "read") and not isinstance(src, str):
                continue
            try:
                url_obj = URL.from_(src)
            except Exception:
                continue
            mt = url_obj.infer_media_type(default=None)
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
    return IO.class_for_media_type(mt, default=None)


class IO(Singleton, URLBased, Tabular[O], Disposable, BinaryIO, Generic[T, O]):
    """Position-addressable byte holder + seekable cursor + tabular handle.

    Three layered shapes share the class:

    - **Storage IOs** — :class:`yggdrasil.io.memory.Memory` and
      :class:`yggdrasil.io.path.Path` subclasses. Own their bytes
      directly; implement the storage primitives (``_read_mv`` /
      ``_write_mv`` / ``reserve`` / ``truncate`` / ``_clear`` /
      :attr:`size` / :meth:`_stat`).
    - **Cursor IOs** — borrow a parent storage IO via ``_parent``;
      every byte primitive delegates through :meth:`_active` to
      that parent. Built by :meth:`open` and by format-leaf
      construction with ``parent=`` / ``holder=``.
    - **Format-leaf IOs** — :class:`ParquetFile`, :class:`CSVFile`,
      :class:`ArrowIPCFile`, … — register a :class:`MimeType` to
      claim that format in :data:`_HOLDER_FORMAT_REGISTRY`. They
      inherit the cursor delegation and override the two
      :class:`Tabular` hooks against the bound parent's bytes.

    An IO IS a :class:`Disposable`: it can be opened, closed, used
    in a ``with`` block, marked dirty / clean. It is also a
    :class:`Tabular` — the default :meth:`_read_arrow_batches` /
    :meth:`_write_arrow_batches` contextually open the IO
    (``with self.open() as bio:``) and delegate to whichever
    format-leaf the stamped :class:`MediaType` resolves to. That
    means ``LocalPath("data.xlsx").read_pandas_frame()`` works the
    same way ``LocalPath("data.xlsx").open()`` does — the open /
    dispatch / close cycle is hidden behind the Tabular surface.

    Also subclasses :class:`typing.BinaryIO` so external libraries
    that type-check against the stdlib file-like interface (pandas,
    pyarrow, zipfile, …) accept Yggdrasil byte buffers without a
    separate facade. :attr:`mode` returns the POSIX string
    (``"rb"`` / ``"wb+"`` / …) so pandas/zipfile's
    ``"b" in handle.mode`` sniffs work; the typed value lives on
    :attr:`_mode`.

    Storage subclasses implement five primitives:

    - :meth:`_read_mv(n, pos)` — slice ``n`` bytes from ``pos`` as a
      :class:`memoryview`. Receives normalized ``(n, pos)``.
    - :meth:`_write_mv(data, pos)` — splice ``data`` at ``pos``,
      growing the IO if needed. Returns bytes written.
    - :meth:`reserve(n)` — pre-grow the underlying capacity to *at
      least* ``n`` bytes without changing the visible :attr:`size`.
    - :meth:`truncate(n)` — set the visible :attr:`size` to ``n``.
      Shrinks drop the tail; extends zero-pad.
    - :meth:`_clear` — drop the payload entirely.

    Plus the :attr:`size` property and :meth:`resize` (concrete,
    built on :meth:`truncate`). Cursor / format-leaf subclasses
    inherit the delegating defaults — they hand the call to
    ``self._active()`` (= ``self._parent``).
    """

    #: Inherited from :class:`URLBased`. ``None`` on the abstract base
    #: — concrete subclasses override with ``Scheme.X`` and let
    #: :meth:`URLBased.__init_subclass__` register them in the
    #: cross-cutting :data:`_URL_BASED_REGISTRY`.

    #: Format identity for the media-type registry. Subclasses set to a
    #: concrete :class:`MimeType` (``MimeTypes.PARQUET``,
    #: ``MimeTypes.CSV``, ``MimeTypes.FOLDER``, …) to claim that mime
    #: in :data:`_HOLDER_FORMAT_REGISTRY`. ``None`` (the abstract
    #: default) opts out of registration — :class:`Holder` itself and
    #: intermediate abstracts (:class:`IO`, :class:`BytesIO`,
    #: :class:`Memory`, :class:`LocalPath`, :class:`Path`) leave it
    #: unset so they don't shadow the real format leaves. Mirrors
    #: :attr:`scheme`.
    mime_type: "ClassVar[MimeType | None]" = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses keyed on :attr:`mime_type`.

        Mirrors :meth:`URLBased.__init_subclass__`'s scheme-side
        registration. Intermediate abstracts that don't claim a
        :class:`MimeType` are silently skipped.
        """
        super().__init_subclass__(**kwargs)
        mt = cls.mime_type
        if mt is None:
            return
        key = mt.name
        existing = _HOLDER_FORMAT_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise RuntimeError(
                f"Duplicate IO mime_type {mt.value!r}: "
                f"{cls.__name__} clashes with {existing.__name__}. "
                "If the override is intentional, clear the slot first "
                "via _HOLDER_FORMAT_REGISTRY.pop(...) at module-load time."
            )
        _HOLDER_FORMAT_REGISTRY[key] = cls

    __slots__ = (
        "_url",
        "_size",
        "_mtime",
        "_media_type",
        "temporary",
        # Cached payload digest. ``_xxh3_64_size`` / ``_xxh3_64_mtime``
        # form the invalidation key — bumped together by
        # :meth:`_touch_stat`, which every write path eventually flows
        # through. ``-1`` means "never computed".
        "_xxh3_64_cached",
        "_xxh3_64_size",
        "_xxh3_64_mtime",
        # Cursor + mode state — pulled up from the former :class:`IO`
        # subclass after the Holder ↔ IO merge so every Holder gains
        # an opt-in seekable cursor. Read/write primitives advance
        # ``_pos`` when invoked with ``cursor=True``; ``cursor=False``
        # (the default) keeps the positional, cursor-less contract.
        # ``_mode`` defaults to :data:`Mode.AUTO` (``"rb+"`` semantics)
        # so direct holder access stays read/write-able without an
        # explicit mode kwarg.
        "_pos",
        "_mode",
        # Cursor / wrapping. ``_parent`` is the underlying byte holder
        # this one delegates to (``LocalPath`` underneath a
        # :class:`ParquetFile` cursor, :class:`Memory` underneath a
        # :class:`BytesIO`, …). ``None`` on top-level storage leaves
        # (:class:`Memory`, :class:`LocalPath`, :class:`VolumePath`, …)
        # that own their bytes directly. ``_owns_parent`` decides
        # whether closing this Holder also closes the parent —
        # ``True`` on the cursor returned by :meth:`Holder.open` so
        # ``with path.open() as cursor:`` releases the path on exit.
        "_parent",
        "_owns_parent",
    )

    # ------------------------------------------------------------------
    # URLBased — round-trip through a :class:`URL`
    # ------------------------------------------------------------------

    @property
    def parent(self) -> "IO | None":
        """The IO one level up — cursor parent first, else URL parent.

        Resolution order:

        1. The cursor parent (``self._parent``, set by
           :meth:`IO.open` and by format-leaf construction with
           ``parent=`` / ``holder=``). When set, this IO is a
           cursor and the parent is its backing storage.
        2. The URL parent — a sibling IO of the same concrete
           class at ``self.url.parent``. Used by URL-shaped storage
           leaves (:class:`Path` / :class:`LocalPath` / remote paths)
           to walk up the filesystem.

        Returns ``None`` when neither applies (top-level storage
        with no URL hierarchy — e.g., :class:`Memory`, which
        overrides :meth:`_url_parent` to skip the URL branch).
        """
        if self._parent is not None:
            return self._parent
        return self._url_parent()

    def _url_parent(self) -> "IO | None":
        """Hook: the URL-parent sibling, or ``None`` when not applicable.

        Default behaviour for URL-shaped IOs: returns
        ``self._from_url(self.url.parent)`` when the parent URL is
        distinct from ``self.url`` (i.e., not at the root). Subclasses
        without a meaningful URL hierarchy (:class:`Memory`'s
        synthetic ``mem://...`` URLs) override to return ``None``.

        Detect "at the root" by reading the URL's ``path`` directly
        rather than computing ``url.parent`` and comparing — saves the
        7-slot ``URL.__eq__`` walk on every parent-walk iteration,
        which the cached ``URL.parent`` cannot.
        """
        url = self._url
        if url is None:
            return None
        path = url.path
        if not path or path == "/":
            return None
        return self._from_url(url.parent)

    @property
    def parents(self) -> "Iterator[IO]":
        """Walk the parent chain outward, yielding one IO per step.

        Each step follows :attr:`parent` — cursor parent first, then
        URL parent (when applicable), terminating when ``.parent``
        returns ``None``. Empty on top-level non-URL storage
        (:class:`Memory`).
        """
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def joinpath(self, *segments: Any) -> "IO":
        """Build a sibling IO at ``self.url`` joined with *segments*.

        URL-shaped IOs (:class:`LocalPath`, remote paths) use
        this to mint a child path; :class:`Memory` and other
        non-URL leaves raise :class:`ValueError`.
        """
        if self._url is None:
            raise ValueError(
                f"{type(self).__name__} has no URL — joinpath is only "
                "defined for URL-shaped IOs (paths, remotes)."
            )
        return self._from_url(self._url.joinpath(*segments))

    def __truediv__(self, other: Any) -> "IO":
        return self.joinpath(other)

    def to_url(self) -> "URL":
        """The canonical :class:`URL` that addresses this holder."""
        return self.url

    def __repr__(self) -> str:
        opened = "open" if self.opened else "closed"
        if self._parent is not None:
            state = "acquired" if self._acquired else "idle"
            own = "owns" if self._owns_parent else "borrows"
            return (
                f"<{type(self).__name__} {state} {own} "
                f"holder={self._parent!r} pos={self._pos} mode={self._mode!r}>"
            )
        return f"<{type(self).__name__} {self.url!r} [{opened}] {self.stat()!r}>"

    def __hash__(self) -> int:
        # Content-based hash. Mutates with the payload — caller's
        # problem if you stick a Memory in a dict and then write to it.
        return self.url.__hash__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, IO):
            return (
                self.stat().size == other.stat().size
                and self.memoryview() == other.memoryview()
            )
        if isinstance(other, (bytes, bytearray, memoryview)):
            return self.memoryview() == memoryview(other)
        return False

    def __new__(
        cls,
        data: Any = None,
        *,
        stat: IOStats | None = None,
        scheme: str | None = None,
        url: URL | None = None,
        binary: bytes | bytearray | memoryview | None = None,
        path: PathLike | None = None,
        holder: "IO | None" = None,
        owns_holder: bool = False,
        **kwargs: Any,
    ):
        """Create a new IO.

        Two layered dispatches:

        1. **Scheme dispatch** — when called on the abstract
           :class:`IO` itself, routes to the concrete storage
           subclass implied by the inputs (scheme/url registry →
           ``binary`` → ``path`` → ``data`` type → memory default).
        2. **Format dispatch** — when called on any :class:`IO`
           subclass, routes to the registered format leaf
           (:class:`CSVFile`, :class:`ParquetFile`, …) implied by an
           explicit ``media_type``, the ``path``'s extension, the
           ``data``'s URL form, or the bound ``holder``'s stamped
           media type. Storage leaves (:class:`Memory` /
           :class:`LocalPath` / …) never satisfy the gate because the
           registry maps onto format leaves.

        Non-routing kwargs (``stat``, ``temporary``, ``media_type``,
        ``holder`` / ``parent``, ``mode``, ``owns_holder``,
        ``auto_open``, …) ride through ``**kwargs`` so subclass
        ``__new__`` and the eventual ``__init__`` see them.
        ``parent=h`` is accepted as an alias for ``holder=h`` — the
        cursor pattern ``IO(parent=self, cursor=True)`` lands at the
        same dispatch as ``IO(holder=self)``.
        """
        # ``parent`` alias → ``holder`` (and a no-op ``cursor`` flag
        # is consumed here too: it's a marker that this IO is a
        # cursor over its parent; the parent slot already encodes the
        # relationship, so no extra state is needed).
        parent_kwarg = kwargs.pop("parent", None)
        if parent_kwarg is not None and holder is None:
            holder = parent_kwarg
        kwargs.pop("cursor", None)

        # Conflict-arg guards inherited from the pre-merge ``IO`` —
        # the cursor / data / path shapes are mutually exclusive at
        # the construction surface.
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

        if cls is IO and holder is None:
            # In-memory tabular dispatch — when *data* is already a
            # materialised tabular shape (a :class:`pa.Table`, a Spark
            # / polars / pandas frame, a ``list[dict]`` /
            # ``dict[str, list]``, or an existing :class:`Tabular`),
            # route to the right in-memory holder rather than forcing
            # the input through the byte-backed scheme / format
            # registries. Those registries dispatch on URLs and media
            # types; a pure-data payload has neither.
            #
            # - Already a :class:`Tabular` (and not an :class:`IO`,
            #   which keeps its existing cursor-over-IO semantics
            #   below): return as-is. The caller already has the
            #   holder they want; re-wrapping would lose state.
            # - Otherwise consult :func:`_resolve_in_memory_tabular`
            #   for the concrete holder class
            #   (:class:`ArrowTabular` for arrow / polars / pandas /
            #   row-list / column-dict shapes, :class:`Dataset`
            #   for spark — kept lazy on the executors).
            if isinstance(data, Tabular) and not isinstance(data, IO):
                return data
            in_memory_target = _resolve_in_memory_tabular(data)
            if in_memory_target is not None:
                # ArrowTabular / Dataset aren't subclasses of
                # :class:`IO`, so ``isinstance(returned, cls)`` is
                # False at the top of ``type.__call__`` — Python
                # skips the post-``__new__`` ``IO.__init__`` re-entry
                # automatically. No idempotency flag dance.
                return in_memory_target(data, **kwargs)

            # Storage construction — pick the concrete storage subclass
            # for the given seed (scheme → URL registry, binary →
            # Memory, path → LocalPath / remote, …). When a parent
            # ``holder`` is supplied this branch is skipped and the
            # caller falls through to plain ``IO`` allocation below
            # (cursor over the bound parent).
            #
            # Before scheme dispatch, give format dispatch a shot —
            # ``IO(path="x.csv")`` should land on :class:`CSVFile`
            # rather than the underlying storage subclass. The format
            # leaf itself runs the storage scheme dispatch for its
            # own ``_parent`` on the recursive ``__new__`` call.
            fmt_target = _resolve_format_target(
                cls,
                media_type=kwargs.get("media_type"),
                path=path,
                data=data,
                holder=holder,
            )
            if fmt_target is not None and issubclass(fmt_target, IO):
                instance = fmt_target.__new__(
                    fmt_target,
                    data=data, stat=stat, scheme=scheme, url=url,
                    binary=binary, path=path, holder=holder,
                    owns_holder=owns_holder, **kwargs,
                )
                if not isinstance(instance, cls):
                    type(instance).__init__(
                        instance,
                        data=data, path=path, binary=binary, url=url,
                        holder=holder, owns_holder=owns_holder, **kwargs,
                    )
                return instance

            target = _resolve_subclass(
                scheme=scheme, url=url, binary=binary, path=path, data=data,
            )
            return target.__new__(
                target,
                data=data,
                stat=stat,
                scheme=scheme,
                url=url,
                binary=binary,
                path=path,
                holder=holder,
                owns_holder=owns_holder,
                **kwargs,
            )

        # Format dispatch — route to the registered format leaf when
        # the construction hints (``media_type``, path extension, …)
        # name one different from ``cls``.
        #
        # Storage subclasses (those with a ``scheme`` class-level
        # marker — :class:`Memory`, :class:`LocalPath`, remote paths)
        # skip this branch: format leaves are cursor-shaped (no
        # scheme) and a path holder needs to stay a path holder even
        # when its URL extension names a registered format.
        target = (
            _resolve_format_target(
                cls,
                media_type=kwargs.get("media_type"),
                path=path,
                data=data,
                holder=holder,
            )
            if not getattr(cls, "scheme", None)
            else None
        )
        if target is not None and target is not cls and issubclass(target, IO):
            instance = target.__new__(
                target,
                data=data,
                stat=stat,
                scheme=scheme,
                url=url,
                binary=binary,
                path=path,
                holder=holder,
                owns_holder=owns_holder,
                **kwargs,
            )
            # When target isn't a subclass of cls (sideways routes
            # like ``BytesIO(path="x.parquet")`` → :class:`ParquetFile`,
            # which inherits :class:`IO` directly), Python won't
            # auto-invoke ``__init__`` on the returned instance —
            # do it ourselves so the instance is fully set up.
            if not isinstance(instance, cls):
                type(instance).__init__(
                    instance,
                    data=data,
                    path=path,
                    binary=binary,
                    url=url,
                    holder=holder,
                    owns_holder=owns_holder,
                    **kwargs,
                )
            return instance

        # Forward construction args to :class:`Singleton.__new__` so the
        # default ``_singleton_key`` (or a subclass override) can read
        # ``url`` / ``data`` / ``client`` off them. Concrete leaves that
        # opt out of caching (``_SINGLETON_TTL = ...`` on
        # :class:`IO` itself) short-circuit before this matters.
        instance = super().__new__(
            cls,
            data=data,
            stat=stat,
            scheme=scheme,
            url=url,
            binary=binary,
            path=path,
            **kwargs,
        )
        # Pre-stamp the cursor's parent / ownership before ``__init__``
        # so the parent-probe (``try: self._parent``) sees the bound
        # holder instead of clobbering it. When *cls* is a cursor-only
        # leaf (e.g. :class:`BytesIO`, :class:`ParquetFile`) and the
        # caller handed us a ``data`` / ``path`` / ``binary`` / ``url``
        # seed without an explicit ``holder=``, auto-build a storage
        # parent via abstract ``IO(...)`` scheme dispatch so the cursor
        # has something to delegate byte primitives to. Storage leaves
        # (:class:`Memory`, :class:`LocalPath`, …) own their bytes
        # directly and seed via ``__init__`` instead — detected by the
        # ``scheme`` class-level marker on URL-based subclasses.
        if holder is not None:
            instance._parent = holder
            instance._owns_parent = bool(owns_holder)
        elif isinstance(data, IO):
            # Cursor-over-IO: borrow the parent's storage rather than
            # reconstructing one. ``BytesIO(other_bytes_io)`` shares the
            # same byte substrate; the new cursor owns nothing of its
            # own beyond the position. ``_resolve_subclass(data=IO)``
            # would return ``type(data)`` and route us back into this
            # branch — infinite recursion.
            instance._parent = data._parent if data._parent is not None else data
            instance._owns_parent = False
        elif not getattr(cls, "scheme", None):
            # Cursor-only leaf without an explicit ``holder=`` — mint
            # the storage parent via the scheme registry directly so
            # we don't re-enter the format-dispatch path on ``cls``.
            try:
                parent_target = _resolve_subclass(
                    scheme=scheme, url=url, binary=binary,
                    path=path, data=data,
                )
                instance._parent = parent_target(
                    data=data, path=path, binary=binary, url=url,
                )
                instance._owns_parent = True
            except (TypeError, ValueError):
                # Subclass __init__ may have richer drain logic
                # (file-like ``data``, backend-specific shapes); leave
                # the slots at their defaults for it to populate.
                pass
        return instance

    def __init__(
        self,
        data: Any = None,
        *,
        stat: IOStats | None = None,
        url: URL | None = None,
        binary: bytes | bytearray | memoryview | None = None,
        path: PathLike | None = None,
        holder: "IO | None" = None,
        owns_holder: bool = False,
        mode: ModeLike = "rb+",
        media_type: Any = None,
        temporary: bool = False,
        singleton_ttl: Any = ...,
        **kwargs,
    ):
        """Initialize the IO.

        Exactly one of ``url`` / ``binary`` / ``path`` / ``data`` /
        ``holder`` determines the seed; the rest are mutually
        exclusive (validated in :meth:`__new__`).

        ``holder=`` (alias: ``parent=``) borrows an existing IO as
        backing storage — every byte primitive then delegates through
        :meth:`_active`. ``owns_holder=True`` transfers close-ownership
        so closing this IO also closes the parent.

        ``temporary=True`` marks the IO for self-cleanup on release:
        :meth:`_release` calls :meth:`clear` so the payload is dropped
        when the IO closes. Default ``False`` — clears only happen
        when the caller asks.

        ``mode`` follows stdlib :func:`open` semantics, normalized to
        a :class:`Mode` enum. Side effects fire on :meth:`_acquire`,
        not here: cursor stays at byte 0 until then.

        ``stat`` lets callers seed the metadata cache (size / mtime /
        media_type) when they already know it — saves a backend probe
        on the first :meth:`stat` call.
        """
        super().__init__(singleton_ttl=singleton_ttl, **kwargs)

        self._url: URL | None = None
        if url is not None:
            self.url = url
        # IO owns its own size + mtime + media_type. Subclasses
        # update these via :meth:`_touch_stat` (or direct mutation
        # on hot paths); :meth:`_stat` snapshots them into a fresh
        # :class:`IOStats` on demand.
        self._size: int = int(stat.size) if stat is not None else 0
        self._mtime: float = float(stat.mtime) if stat is not None else 0.0
        # Lazy xxh3_64 digest cache — paid on first call, valid until
        # ``_size`` or ``_mtime`` shifts (every write goes through
        # :meth:`_touch_stat`, which updates one or the other).
        self._xxh3_64_cached: int = 0
        self._xxh3_64_size: int = -1
        self._xxh3_64_mtime: float = -1.0
        # Cursor + mode — opt-in seekable surface. ``_pos`` only moves
        # when a caller passes ``cursor=True`` to a read/write primitive
        # or drives the IO through stdlib-style ``read()`` / ``write()``.
        self._pos: int = 0
        self._mode: Mode = Mode.from_(mode)
        # Parent / wrapping defaults — top-level storage IOs own
        # their bytes (no parent). Cursor / format-leaf subclasses set
        # ``_parent`` in :meth:`__new__` (before ``__init__`` runs),
        # so respect any pre-set value here instead of clobbering it.
        try:
            self._parent  # noqa: B018  -- slot probe
        except AttributeError:
            self._parent = None
            self._owns_parent = False
        if stat is not None and stat.media_type is not None:
            self._media_type = stat.media_type
        else:
            self._media_type = ...
        self.temporary: bool = bool(temporary)

        for prio in (binary, path, data):
            if prio is not None:
                self._init_from(prio)
                break

        # If this IO has no parent yet but the caller handed us a
        # ``data``-shaped input the scheme dispatch could not drain
        # (file-like objects, backend-specific shapes), fall back to
        # :meth:`from_` to build a fresh in-memory holder for the
        # cursor to wrap.
        if self._parent is None and data is not None and not (
            isinstance(data, (bytes, bytearray, memoryview, str, pathlib.PurePath, URL))
            or isinstance(data, IO)
        ):
            try:
                tmp = type(self).from_(data, mode=mode)
                self._parent = tmp._parent
                self._owns_parent = True
            except (TypeError, ValueError):
                # Subclass may have richer drain logic in a custom
                # ``__init__``; leave the slots at defaults.
                pass

        # Stamp media type onto the bound holder's IOStats when this
        # is a cursor — gives the codec auto-handling path something
        # to inspect, and makes the buffer self-describing.
        if media_type is not None and self._parent is not None:
            try:
                mt = MediaType.from_(media_type, default=None)
                if mt is not None:
                    self._parent.media_type = mt
            except Exception:
                pass

    def _init_from(self, data: Any) -> None:
        if isinstance(data, IO):
            self._init_from_holder(data)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            self._init_from_bytes(data)
        elif isinstance(data, str):
            self._init_from_str(data)
        elif isinstance(data, pathlib.PurePath):
            self._init_from_pathlib(data)
        elif isinstance(data, URL):
            self._init_from_url(data)
        else:
            raise TypeError(
                f"Cannot initialize {type(self).__name__} from "
                f"{type(data).__name__}: {data!r}"
            )

    def _init_from_holder(self, holder: "IO") -> None:
        if not self._url:
            self.url = holder.url

        if not self._url_matches(holder.url):
            self.write_bytes(holder.read_bytes())

    def _init_from_bytes(self, data: bytes | bytearray | memoryview) -> None:
        self.write_bytes(data)

    def _init_from_local_path(self, path: PathLike) -> None:
        url = URL.from_(path)
        if not self._url:
            self.url = url

        # Path-shaped seed on a path-shaped holder is identity only —
        # there's nothing to copy, and the file may not exist yet.
        # The cross-backend case (Memory seeded from a local file)
        # still routes through write_local_path.
        if self._url_matches(url):
            return
        # A cross-backend seed (e.g. Memory seeded from a local path)
        # only has bytes to copy when the file actually exists. ``IO(data=
        # str(tmp_path / "x.csv"))`` is the common "I'll write to this
        # location later" pattern — the holder should land empty but
        # carrying the URL + format dispatch from the suffix, not crash
        # with ``FileNotFoundError`` inside ``write_local_path``.
        if not os.path.exists(os.fspath(path)):
            return
        self.write_local_path(path)

    def _init_from_pathlib(self, path: pathlib.PurePath) -> None:
        self._init_from_local_path(os.fspath(path))

    def _init_from_str(self, value: str) -> None:
        if URL.is_urlish(value):
            self._init_from_url(URL.from_(value))
            return

        raise ValueError(
            f"Cannot initialize {type(self).__name__} from string {value!r}: "
            "not a recognized URL"
        )

    def _init_from_url(self, url: URL) -> None:
        if not self._url:
            self.url = url

        if self._url_matches(url):
            return

        self._init_from_local_path(url.__fspath__())

    def _url_matches(self, candidate: URL) -> bool:
        """True when *candidate* points at the same place as :attr:`_url`.

        Compares both the canonical URL and the local fspath so the
        check survives a setter that adds/strips a scheme. Returns
        ``False`` when no URL is bound yet.
        """
        if self._url is None:
            return False
        if self._url == candidate:
            return True
        try:
            return self._url.__fspath__() == candidate.__fspath__()
        except Exception:
            return False

    def _init_from_file_like(self, data: "_StdlibIO[bytes]") -> None:
        offset = 0
        while True:
            chunk = data.read(_COPY_CHUNK)
            if not chunk:
                break
            self.write_bytes(chunk, offset=offset)
            offset += len(chunk)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_(
        cls,
        obj: Any,
        *,
        url: URL | None = None,
        mode: ModeLike = "rb+",
        **kwargs,
    ) -> "IO":
        """Auto-route *obj* to the right storage / cursor, return an owning IO.

        Two shapes share the method:

        - **Storage subclasses** (``cls`` has a :attr:`scheme` —
          :class:`IO` itself, :class:`Memory`, :class:`LocalPath`,
          remote paths). The result is a storage IO that owns its
          bytes — ``IO.from_(b"x")`` → :class:`Memory`,
          ``IO.from_("file://...")`` → :class:`LocalPath`.
        - **Cursor / format-leaf subclasses** (``cls`` has no
          ``scheme`` — :class:`BytesIO`, :class:`ParquetFile`,
          :class:`CSVFile`, …). The result is an owning cursor over
          a fresh storage parent built from *obj*.

        Recognised input shapes:

        - :class:`IO` of ``cls`` — pass through (idempotent).
        - :class:`IO` of a different class — for storage ``cls``,
          return the underlying parent; for cursor ``cls``, borrow
          the same parent into a fresh cursor.
        - bytes-like (``bytes`` / ``bytearray`` / ``memoryview``) —
          back with a fresh :class:`Memory`.
        - path-like (``str`` / ``pathlib.Path`` / ``URL``) — back
          with the path-shaped storage class for the scheme.
        - local file handle — back with :class:`LocalPath`; lazy
          read from disk (no drain).
        - other file-like — drain into a fresh :class:`MemoryStream`.
        """
        if isinstance(obj, cls):
            return obj

        is_storage = bool(getattr(cls, "scheme", None)) or cls is IO

        if isinstance(obj, IO):
            if is_storage:
                # Caller wants a storage handle; return the underlying
                # storage parent when ``obj`` is a cursor, else the
                # storage instance itself.
                target = obj._parent if obj._parent is not None else obj
                if isinstance(target, cls):
                    return target
                return cls(data=target, url=url, **kwargs)
            # Different cursor / format-leaf class over the same byte
            # substrate — borrow the holder rather than drain.
            return cls(
                holder=obj._parent if obj._parent is not None else obj,
                owns_holder=False, mode=mode, url=url, **kwargs,
            )

        if isinstance(obj, (bytes, bytearray, memoryview)):
            if is_storage:
                return cls(binary=obj, url=url, **kwargs)
            from .memory import Memory
            return cls(
                holder=Memory(binary=obj), owns_holder=True, mode=mode,
                url=url, **kwargs,
            )

        if hasattr(obj, "read") and not isinstance(obj, (str, bytes)):
            # Live local file handle (``open("path", "rb")``,
            # ``pathlib.Path.open()``) carries a string ``.name``
            # pointing at the on-disk file. Route through LocalPath
            # so the holder reads from disk on demand instead of
            # draining into memory. Anonymous streams fall through
            # to :class:`MemoryStream`.
            local_path = _local_path_for_handle(obj)
            if local_path is not None:
                from yggdrasil.io.path.local_path import LocalPath
                if is_storage:
                    return LocalPath(local_path) if cls in (IO, LocalPath) else cls(path=local_path, url=url, **kwargs)
                return cls(
                    holder=LocalPath(local_path),
                    owns_holder=True, mode=mode, url=url, **kwargs,
                )
            from yggdrasil.io.memory_stream import MemoryStream
            if is_storage:
                return MemoryStream(obj) if cls in (IO,) else cls(data=obj, url=url, **kwargs)
            return cls(
                holder=MemoryStream(obj),
                owns_holder=True, mode=mode, url=url, **kwargs,
            )

        # Path-like — route through ``cls(path=...)`` so scheme-aware
        # dispatch (file → LocalPath, s3 → S3Path, …) fires.
        if isinstance(obj, (str, pathlib.PurePath, URL)):
            try:
                if is_storage:
                    return cls(path=obj, url=url, **kwargs)
                return cls(path=obj, mode=mode, url=url, **kwargs)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Cannot wrap {type(obj).__name__} as a "
                    f"{cls.__name__} via path dispatch. Got {obj!r}."
                ) from exc

        # No recognised shape — bail out loudly. Routing an
        # arbitrary value through the ``data=`` channel silently
        # creates an empty IO (the constructor's seed loop only
        # touches bytes-like / path / URL / IO inputs), which masks
        # caller bugs at the boundary.
        raise TypeError(
            f"Cannot wrap {type(obj).__name__} as a {cls.__name__}. "
            "Accepted: IO, bytes-like, str / PathLike / URL, or a "
            f"file-like object with a ``read`` method. Got {obj!r}."
        )

    @classmethod
    def from_url(cls, url: URL, **kwargs) -> "IO":
        """Create a new IO from a URL.

        When *cls* is abstract (has subclasses but isn't itself
        constructible — e.g. :class:`Path`), the URL scheme is
        resolved through the :class:`URLBased` registry to a concrete
        subclass; an unknown scheme raises :class:`ValueError`
        instead of producing the obscure "Can't instantiate abstract
        class" :class:`TypeError`.
        """
        u = URL.from_(url)
        # Abstract dispatch: when ``cls`` has subclasses and its own
        # ``scheme`` marker is empty, route via the URLBased registry.
        if cls.__subclasses__() and not getattr(cls, "scheme", None):
            scheme = u.scheme
            if scheme:
                try:
                    target = URLBased.for_scheme(scheme)
                except (ValueError, ImportError) as exc:
                    raise ValueError(
                        f"Unknown scheme {scheme!r} for "
                        f"{cls.__name__}.from_url({url!r})."
                    ) from exc
                return target(url=u, **kwargs)
        return cls(url=u, **kwargs)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> "IO":
        """Create a new IO from bytes."""
        return cls(binary=data, **kwargs)

    @classmethod
    def from_holder(
        cls,
        holder: "IO",
        *,
        owns_holder: bool = False,
        mode: ModeLike = "rb+",
        media_type: Any = None,
        auto_open: bool = True,
        **kwargs: Any,
    ) -> "IO":
        """Construct a cursor over *holder*, dispatching to the format leaf.

        Resolves the format-specific :class:`IO` leaf via *media_type*
        (when given) or the holder's stamped ``stat().media_type``, and
        returns an instance of that leaf bound to *holder*. When no
        leaf can be resolved, falls back to ``cls`` itself.

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

    def _from_url(self, url: URL, **kwargs: Any) -> "IO":
        """Build a sibling :class:`IO` for *url* of this one's class.

        Reuses the existing :attr:`_parent` when set (cursor case —
        the new sibling shares the same backing storage and just
        re-points the URL); otherwise builds the sibling from *url*
        directly (top-level storage case — :class:`LocalPath`,
        :class:`Memory`, …, where the URL *is* the addressing).
        """
        if self._parent is not None:
            return type(self)(parent=self._parent, url=url, **kwargs)
        return type(self)(url=url, **kwargs)

    # ------------------------------------------------------------------
    # Format registry — MediaType → Holder subclass dispatch
    # ------------------------------------------------------------------

    @classmethod
    def class_for_media_type(
        cls,
        media_type: "MediaType | MimeType | str | Any",
        *,
        default: Any = ...,
    ) -> "type":
        """Resolve a :class:`MediaType` (or coercible) to its format leaf.

        Looks up :attr:`MediaType.mime_type`'s name in
        :data:`_HOLDER_FORMAT_REGISTRY`. Codec is orthogonal — Parquet
        compressed with zstd or snappy still resolves to
        :class:`ParquetFile`; the codec layer is the holder's concern.

        The returned class is a :class:`Tabular` subclass — typically a
        :class:`Holder` byte-backed leaf, occasionally a non-Holder
        leaf (:class:`FolderIO`, :class:`DeltaFolder`). Returns *default*
        on miss when supplied; otherwise raises :class:`KeyError` with
        the list of registered names.
        """
        mt = MediaType.from_(media_type, default=None)
        if mt is None:
            if default is ...:
                raise KeyError(
                    f"Cannot coerce {media_type!r} to a MediaType "
                    "for IO format-registry lookup."
                )
            return default

        hit = _HOLDER_FORMAT_REGISTRY.get(mt.mime_type.name)
        if hit is not None:
            return hit

        # Miss may just mean the leaf package hasn't been imported
        # yet — force the side-effect bootstrap once and retry. This
        # is what catches nested leaves (ZipFile / FolderIO / DeltaFolder)
        # for callers that never touched ``yggdrasil.io.nested``.
        if not _HOLDER_FORMAT_REGISTRY_BOOTSTRAPPED:
            _bootstrap_holder_format_registry()
            hit = _HOLDER_FORMAT_REGISTRY.get(mt.mime_type.name)
            if hit is not None:
                return hit

        if default is ...:
            raise KeyError(
                f"No IO registered for {mt.mime_type.value!r}. "
                f"Registered: {sorted(_HOLDER_FORMAT_REGISTRY)}."
            )
        return default

    @classmethod
    def for_holder(
        cls,
        holder: "IO",
        *,
        media_type: "MediaType | MimeType | str | None" = None,
        default: Any = ...,
        **kwargs: Any,
    ) -> "Tabular":
        """Build the right format leaf for *holder*.

        Resolution order for the format discriminator:

        1. The explicit *media_type* kwarg, when supplied.
        2. ``holder.stat().media_type`` — set by the holder from its
           URL extension, magic-byte sniff, or content-type header.

        The resolved class is instantiated as ``Cls(holder=holder,
        **kwargs)``. On lookup miss, falls back to *default* when
        supplied; otherwise raises :class:`KeyError`.
        """
        mt = media_type
        if mt is None:
            stats = getattr(holder, "stat", None)
            if callable(stats):
                mt = getattr(stats(), "media_type", None)

        if mt is None:
            if default is ...:
                raise KeyError(
                    f"No media_type on {holder!r}; pass media_type= "
                    "explicitly or seed the holder's IOStats."
                )
            return default

        target = cls.class_for_media_type(mt, default=default)
        if target is default and default is not ...:
            return default
        return target(holder=holder, **kwargs)

    @classmethod
    def registered_classes(cls) -> "dict[str, type]":
        """Snapshot of the registry — debugging / introspection only."""
        return dict(_HOLDER_FORMAT_REGISTRY)

    # ------------------------------------------------------------------
    # Abstract primitives
    # ------------------------------------------------------------------

    def read_mv(
        self,
        size: int = -1,
        offset: int = 0,
        *,
        cursor: bool = False,
    ) -> memoryview:
        """Slice ``size`` bytes from ``offset`` as a :class:`memoryview`.

        ``cursor=True`` ignores the explicit *offset* and reads from
        the holder's internal cursor (:attr:`tell`), advancing it past
        the bytes returned. ``cursor=False`` (default) keeps the
        cursor-less positional contract — the cursor is untouched.

        Cursor IOs (those wrapping a :attr:`parent` storage) delegate
        the whole call through :meth:`_active` so the parent's
        bounds-check uses its own size — avoids a redundant ``stat``
        probe on remote backings when the cursor has no local size
        cache, and routes through any subclass ``_active`` override
        (lazy materialization on :class:`ZipEntryFile`, …).
        """
        if self._parent is not None:
            if cursor:
                offset = self._pos
            out = self._active().read_mv(size, offset)
            if cursor:
                self._pos = offset + len(out)
            return out
        if cursor:
            offset = self._pos
        total = self.size
        offset = _resolve_pos(offset, total)
        if offset < 0 or offset > total:
            raise ValueError(
                f"Offset {offset} is out of bounds for "
                f"{type(self).__name__} of size {total}"
            )
        if size < 0:
            size = total - offset
        if size < 0 or offset + size > total:
            raise ValueError(
                f"Range [{offset}, {offset + size}) is out of bounds for "
                f"{type(self).__name__} of size {total}"
            )

        out = self._read_mv(size, offset)
        if cursor:
            self._pos = offset + size
        return out

    def _read_mv(self, n: int, pos: int) -> memoryview:
        """Return a memoryview over ``n`` bytes starting at ``pos``.

        Bounds and negative-index normalization happen in :meth:`read_mv`;
        this hook receives non-negative, in-range ``(n, pos)`` with
        ``0 <= pos <= size`` and ``0 <= n <= size - pos``. The append
        point ``pos == size`` is reachable via ``pos = -1`` and always
        pairs with ``n == 0`` — return an empty view in that case.

        Cursor / format-leaf IOs inherit a delegating default that
        forwards to ``self._active()._read_mv(n, pos)``. Storage
        subclasses (:class:`Memory`, :class:`Path`) override this to
        slice their own buffer. The view's lifetime tracks the
        underlying storage; subclasses MAY return a view that backs
        onto a transient buffer (e.g. a remote download) — the
        caller must consume / copy the view before any other I/O.
        """
        if self._parent is not None:
            return self._active()._read_mv(n, pos)
        raise NotImplementedError(
            f"{type(self).__name__} has no _read_mv implementation and "
            "no bound parent IO. Storage subclasses must override "
            "_read_mv; cursor / format-leaf subclasses must be "
            "constructed with a parent IO."
        )

    def write_mv(
        self,
        data: memoryview,
        offset: int = 0,
        *,
        size: int = -1,
        overwrite: bool = False,
        update_stat: bool = True,
        cursor: bool = False,
    ) -> int:
        """Splice ``data`` at ``offset``, pre-growing the holder as needed.

        ``size`` caps the byte count written — ``size=-1`` (default)
        writes all of ``data``; ``size>=0`` writes
        ``min(len(data), size)`` bytes. Caps via a slice of
        ``data`` (zero-copy on ``memoryview`` / ``bytes``), so
        downstream pipelines that only need the first N bytes of
        a larger buffer skip the trailing tail.

        ``overwrite`` declares that this write replaces the
        holder's tail past ``offset + size`` — after the splice,
        :attr:`size` is set to ``offset + size``. Callers that
        currently do ``truncate(0)`` followed by ``write_bytes(...)``
        collapse to a single ``write_bytes(..., overwrite=True)``,
        which on whole-blob remote backends saves a SDK round
        trip (the atomic upload at ``offset == 0`` already
        replaces the object — no preceding truncate needed).

        Pipeline:

        1. Slice ``data`` to ``size`` if capped.
        2. Normalize ``offset`` (``-1`` → append, ``-N`` →
           ``self.size - N``).
        3. Pre-grow visible :attr:`size` to cover the splice via
           :meth:`resize`.
        4. Hand the normalized ``(data, offset)`` to
           :meth:`_write_mv`.
        5. Truncate tail past ``offset + n`` when ``overwrite``.
        6. Mark dirty + bump cached mtime if anything was written.

        ``update_stat=False`` skips the post-write
        :meth:`_touch_stat` and :meth:`mark_dirty` calls. Use it for
        bulk loops that want a single stat refresh at the end (one
        :func:`time.time` call instead of one per write); the caller
        is then responsible for calling :meth:`_touch_stat` (or
        re-statting via the path-side ``_stat`` for filesystem
        backends) once the loop finishes.

        Cursor IOs (those wrapping a :attr:`parent` storage) delegate
        the whole call through :meth:`_active` so the parent's
        resize / bounds-check / dirty-marking fires once, on the
        backing storage — the cursor only advances its own ``_pos``.
        """
        if self._parent is not None:
            if cursor:
                offset = self._pos
            written = self._active().write_mv(
                data, offset, size=size, overwrite=overwrite,
                update_stat=update_stat,
            )
            if cursor:
                self._pos = offset + written
            return written
        if cursor:
            offset = self._pos
        if size >= 0 and len(data) > size:
            data = data[:size]
        total = self.size
        offset = _resolve_pos(offset, total)
        if offset < 0:
            raise ValueError(
                f"Offset {offset} is out of bounds for "
                f"{type(self).__name__} of size {total}"
            )

        n = len(data)
        end = offset + n
        if n == 0:
            if overwrite and end < total:
                self.truncate(end)
                if update_stat:
                    self._touch_stat(size=self.size)
                    self.mark_dirty()
            return 0

        # Pre-grow the visible size so _write_mv just lays bytes down
        # at a known-valid range. resize() is a no-op when offset+n
        # <= size (in-place overwrite case), so the fast path stays fast.
        if end > total:
            self.resize(end)

        written = self._write_mv(data, offset)
        # ``overwrite`` drops any tail beyond the spliced range —
        # collapses ``truncate(0) + write_bytes(...)`` into one call
        # and lets whole-blob remote backends skip the preceding
        # truncate SDK round trip (the atomic upload at ``offset
        # == 0`` already replaces the object).
        if overwrite and end < self.size:
            self.truncate(end)

        if written > 0 and update_stat:
            self._touch_stat(size=max(end, self.size))
            self.mark_dirty()
        if cursor:
            self._pos = offset + written
        return written

    def _write_mv(self, data: memoryview, pos: int) -> int:
        """Splice ``data`` at ``pos``. Returns bytes actually written.

        Receives a normalized non-negative ``pos`` and an IO that's
        already been grown (via :meth:`resize`) to cover ``pos +
        len(data)``. Subclasses just put bytes down — no size
        management, no negative-index normalization. Dirty marking and
        stat-cache updates happen in :meth:`write_mv`. Cursor /
        format-leaf IOs inherit a delegating default forwarding to
        ``self._active()._write_mv(data, pos)``.
        """
        if self._parent is not None:
            return self._active()._write_mv(data, pos)
        raise NotImplementedError(
            f"{type(self).__name__} has no _write_mv implementation."
        )

    def reserve(self, n: int) -> None:
        """Pre-grow capacity to *at least* ``n`` bytes.

        Capacity-only — does NOT change :attr:`size`. Idempotent
        when capacity ≥ ``n`` already. Subclasses with no growable
        capacity layer may treat this as a no-op. Cursor / format-leaf
        IOs delegate to the bound parent.
        """
        if self._parent is not None:
            self._active().reserve(n)
            return
        raise NotImplementedError(
            f"{type(self).__name__} has no reserve implementation."
        )

    def resize(self, n: int) -> int:
        """Grow visible :attr:`size` to at least ``n`` bytes (one-way).

        Sister of :meth:`truncate`, but never shrinks. Used by
        :meth:`write_mv` to pre-allocate a known target before the
        splice so :meth:`_write_mv` doesn't have to manage size.

        - ``n <= size`` → no-op, returns current :attr:`size`.
        - ``n  > size`` → extends with zero-padding via
          :meth:`truncate`, returns ``n``.

        Subclasses with a native grow-only primitive (capacity hint to
        a remote upload session, ``posix_fallocate`` on local fd)
        override for the cheaper path; the default works on every
        backend.
        """
        if n < 0:
            raise ValueError(f"resize size must be >= 0, got {n!r}")
        current = self.size
        if n <= current:
            return current
        return self.truncate(n)

    def truncate(self, size: "int | None" = None) -> int:
        """Set the visible :attr:`size` to exactly ``size`` bytes.

        Shrinks drop the tail; extends zero-pad. Returns the new size.

        On a cursor (``self._parent is not None``), ``size=None``
        truncates at the current cursor position and the cursor is
        clamped if it would exceed the post-truncate size. On a
        storage IO ``size=None`` is invalid — pass an explicit byte
        count.
        """
        if self._parent is not None:
            if size is None:
                size = self._pos
            size = int(size)
            n = self._active().truncate(size)
            if self._pos > n:
                self._pos = n
            return n
        if size is None:
            raise TypeError(
                f"{type(self).__name__}.truncate requires an explicit "
                "size on a storage IO (no cursor to default from)."
            )
        raise NotImplementedError(
            f"{type(self).__name__} has no truncate implementation."
        )

    def clear(self) -> None:
        """Drop the IO's payload entirely.

        :class:`Memory` resets the underlying ``bytearray`` to zero
        bytes (capacity drops too). :class:`yggdrasil.io.path.Path`
        unlinks the backing file with ``missing_ok=True`` so the
        operation is idempotent. After :meth:`clear`, :attr:`size`
        reads ``0`` and the IO is still usable — subsequent writes
        grow it from scratch.
        """
        self._clear()

    def _clear(self) -> None:
        """Drop the IO's payload entirely.

        Cursor / format-leaf IOs delegate to the bound parent;
        storage subclasses override to drop their own backing.
        """
        if self._parent is not None:
            self._active()._clear()
            return
        raise NotImplementedError(
            f"{type(self).__name__} has no _clear implementation."
        )

    @property
    def size(self) -> int:
        """Current visible size in bytes.

        Cursor / format-leaf IOs read the bound parent's size;
        storage subclasses override directly.
        """
        if self._parent is not None:
            return self._active().size
        raise NotImplementedError(
            f"{type(self).__name__} has no size implementation."
        )

    @property
    def size_known(self) -> bool:
        """``True`` when reading :attr:`size` won't trigger a backend probe.

        Always true for in-memory IOs (size is a slot). Path IOs
        override to ``True`` only when their stat cache is warm —
        callers that want to short-circuit on an empty buffer
        (parquet / arrow IPC / CSV readers checking ``size == 0``)
        can guard the check on this predicate so a cold remote path
        doesn't pay a ``HeadObject`` / ``get_status`` /
        ``get_metadata`` round trip just to discover the file is
        non-empty. Cursor / format-leaf IOs delegate to the parent.
        """
        if self._parent is not None:
            return self._active().size_known
        return True

    def is_empty(self):
        return self.size == 0

    # ------------------------------------------------------------------
    # IOStats — built fresh from holder-owned slots
    # ------------------------------------------------------------------
    #
    # The holder itself owns the canonical ``_size`` / ``_mtime`` /
    # ``_media_type`` fields. :meth:`_stat` is the abstract hook
    # subclasses implement to snapshot those (plus any backend-derived
    # fields like ``kind``) into a fresh :class:`IOStats`. Callers that
    # need to mutate metadata go through the typed surfaces
    # (``holder.media_type = ...``, :meth:`_touch_stat`) — mutating the
    # returned ``IOStats`` no longer round-trips, since each call
    # produces a fresh instance.

    def stat(self) -> IOStats:
        """Snapshot the holder's metadata into a fresh :class:`IOStats`.

        Delegates to :meth:`_stat` for the backend-specific fields
        (``kind`` and the live size for path-bound holders); mutating
        the returned instance does NOT round-trip onto the holder.
        Use the holder's own setters / :meth:`_touch_stat` when you
        need to update metadata.
        """
        return self._stat()

    def _stat(self) -> IOStats:
        """Snapshot the holder's metadata into a fresh :class:`IOStats`.

        Subclasses build the :class:`IOStats` from their authoritative
        state — ``self._size`` / ``self._mtime`` for in-memory
        holders, a backend round-trip for path holders. The base
        :meth:`stat` always routes through this hook so callers don't
        need to know which backend they're against. Cursor /
        format-leaf IOs delegate to the bound parent.
        """
        if self._parent is not None:
            return self._active()._stat()
        raise NotImplementedError(
            f"{type(self).__name__} has no _stat implementation."
        )

    def _touch_stat(
        self,
        *,
        size: int | None = None,
        mtime: float | None = None,
        media_type: Any = None,
    ) -> None:
        """Update the holder-owned metadata fields after a successful write.

        Centralized so :meth:`write_mv` (and any subclass with a
        cheaper write path that bypasses :meth:`write_mv`) can keep
        ``size`` / ``media_type`` fresh without duplicating the
        bookkeeping.

        ``mtime`` is **only** updated when the caller passes it
        explicitly. The previous behavior — bumping ``mtime`` to
        ``time.time()`` on every write — added a syscall-equivalent
        clock read to every byte-level call and dominated tight
        write loops; callers that actually want the freshness should
        either pass ``mtime=`` or call :meth:`touch_mtime` once at
        the end of the operation.
        """
        if size is not None:
            self._size = int(size)
        if mtime is not None:
            self._mtime = float(mtime)
        if media_type is not None:
            self._media_type = media_type

    def touch_mtime(self, when: float | None = None) -> None:
        """Stamp the holder's mtime with the current time.

        Bulk-write helper — call once after a write loop instead of
        letting every :meth:`write_mv` call sample the clock. ``when``
        accepts an explicit timestamp (e.g. an upstream "Last-Modified"
        header); ``None`` defaults to :func:`time.time`.
        """
        self._mtime = float(when) if when is not None else time.time()

    @property
    def mtime(self) -> float:
        """Last-modified time stamp."""
        return self._mtime

    @property
    def media_type(self):
        """The holder's :class:`MediaType`, or ``None`` if unset.

        Resolves lazily on first read: a fresh holder bound only by URL
        carries the sentinel ``...`` in :attr:`_media_type` and runs
        :meth:`URL.infer_media_type` here once, caching the result back
        onto the slot. Subsequent reads (and pickling, IOStats
        snapshots, codec dispatch, …) hit the cached value.

        Cursor IOs (those wrapping a :attr:`parent` storage) defer to
        the parent's stamped media type when their own slot is unset
        — the codec / format dispatch on a :class:`JSONFile` bound to
        a gzip-stamped :class:`Memory` parent needs to see the parent's
        media type, not its own (the cursor was constructed bare).
        """
        mt = self._media_type
        if mt is ... and self._parent is not None:
            try:
                return self._parent.media_type
            except Exception:
                pass
        if mt is ...:
            url = self._url
            try:
                mt = url.infer_media_type(default=None) if url is not None else None
            except Exception:
                mt = None
            self._media_type = mt
        return mt

    @media_type.setter
    def media_type(self, value: Any) -> None:
        """Stamp a :class:`MediaType` onto the holder.

        Accepts anything :meth:`MediaType.from_` can coerce (a
        :class:`MediaType`, a :class:`MimeType`, a string mime form,
        or ``None`` to clear).
        """
        if value is None:
            self._media_type = None
            return
        try:
            from yggdrasil.data.enums.media_type import MediaType
            mt = MediaType.from_(value, default=None)
        except Exception:
            mt = value
        self._media_type = mt

    # ------------------------------------------------------------------
    # Per-open lifecycle — Path overrides; Memory and other always-live
    # holders inherit no-ops so :class:`BytesIO` can call them blind.
    # ------------------------------------------------------------------

    def acquire(self) -> "IO":
        """Bring the IO's backing into the acquired state.

        Lifecycle primitive — idempotent. Returns ``self``.
        :meth:`__enter__` calls this; so does :meth:`open` before
        constructing its cursor IO.
        """
        if not self._acquired:
            Disposable.open(self)
        return self

    def open(
        self,
        mode: ModeLike = "rb+",
        *,
        media_type: "MediaType | None" = None,
        owns_holder: bool = False,
        auto_open: bool = True,
        **kwargs: Any,
    ) -> "IO":
        """Acquire the IO and return a fresh :class:`IO` cursor over it.

        Dispatches to the format-specific :class:`IO` leaf via the
        IO's stamped media type (or *media_type* override), so
        ``LocalPath("data.parquet").open()`` lands on
        :class:`ParquetFile`, ``LocalPath("data.csv").open()`` on
        :class:`CSVFile`, and an unknown / no-media holder falls back
        to a plain :class:`IO`.

        Pattern::

            with LocalPath("/tmp/x.bin").open("wb") as bio:
                bio.write(b"hello")
            # path released here.

            with LocalPath("data.parquet").open() as bio:
                table = bio.read_arrow_table()  # Tabular surface
            # path released here.

        The default ``owns_holder=False`` returns a non-owning
        cursor — closing the cursor leaves the parent open, so the
        caller can mint multiple cursors against the same parent.
        Pass ``owns_holder=True`` to transfer close-ownership of the
        parent to the cursor (the cursor's close then also closes
        the parent).
        """
        self.acquire()
        # Cursor pattern: a fresh IO bound to ``self`` as its parent,
        # format-dispatched by :meth:`IO.__new__` based on
        # ``media_type`` (explicit override or this IO's stamped one).
        # ``cursor=True`` is a marker the construction path consumes;
        # the parent slot already encodes the cursor↔parent relationship.
        cursor = IO(
            parent=self,
            cursor=True,
            owns_holder=owns_holder,
            mode=mode,
            media_type=self.media_type if media_type is None else media_type,
            **kwargs,
        )
        if auto_open:
            cursor.acquire()
        return cursor

    def __enter__(self) -> "IO":
        """``with holder:`` yields the holder, not a cursor.

        Override of :class:`Disposable.__enter__` (which would
        otherwise call :meth:`open` and hand back a cursor). Use
        ``with holder.open() as bio:`` to get a cursor bound to the
        with-block lifetime.
        """
        self.acquire()
        return self

    # ==================================================================
    # Disposable lifecycle — apply mode side effects, acquire the parent
    # ==================================================================

    def _acquire(self) -> None:
        """Acquire the parent IO and apply the mode side effects.

        Behaviour splits on whether this IO is a storage leaf (no
        parent) or a cursor (parent set):

        - **Storage leaf** — default :class:`Disposable._acquire`
          fires (no-op at this level).
        - **Cursor** — when this IO owns its parent, ``parent.acquire()``
          fires. Mode side effects then apply against the parent:

          - :data:`Mode.OVERWRITE` / :data:`Mode.TRUNCATE` — truncate
            the durable parent to zero bytes.
          - :data:`Mode.APPEND` — cursor parked at EOF.
          - :data:`Mode.ERROR_IF_EXISTS` — fail-fast
            :class:`FileExistsError` if the durable parent is non-empty.
          - :data:`Mode.READ_ONLY` / :data:`Mode.AUTO` / default —
            cursor at 0, durable bytes untouched.

        Must NOT call ``self._parent.open()`` — that's the
        IO-returning convenience and would recurse.
        """
        if self._parent is None:
            return

        if self._owns_parent:
            self._parent.acquire()

        if self._mode is Mode.ERROR_IF_EXISTS and self._parent.size > 0:
            raise FileExistsError(
                f"{type(self).__name__} opened with mode={self._mode!r} "
                f"but holder is non-empty ({self._parent.size} bytes)."
            )

        if self._mode in (Mode.OVERWRITE, Mode.TRUNCATE):
            self._parent.truncate(0)

        self._pos = self._parent.size if self._mode.appendable else 0

    def _active(self) -> "IO":
        """The IO this cursor reads / writes against.

        Returns ``self._parent`` when set (cursor case), else
        ``self`` (storage case). Subclasses that need a side effect
        before every byte-level access (lazy materialization in
        :class:`ZipEntryFile` / :class:`XLSXSheetFile`) override this
        hook to drive the side effect, then ``return super()._active()``.
        """
        return self._parent if self._parent is not None else self

    # ==================================================================
    # Tabular surface — open contextually, delegate to the dispatched leaf
    # ==================================================================

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Stream batches from a borrowed cursor on the dispatched leaf.

        On a storage IO (no parent) this routes through :meth:`open`
        so the same format-leaf dispatch (ParquetFile / XLSXFile /
        CSVFile / …) and ``acquire`` / ``release`` accounting that
        drives explicit ``with holder.open() as bio:`` usage handles
        the contextual read too. Options are re-homed onto the leaf's
        options class so format-specific knobs (sheet name, delimiter,
        …) survive the hop.

        On a plain :class:`IO` cursor (parent set, no format override)
        we have no decoder — format-specific subclasses
        (:class:`ParquetFile`, :class:`CSVFile`, …) override against
        the same byte buffer.
        """
        if self._parent is None:
            with self.open(mode="rb") as bio:
                leaf_options = type(bio).check_options(options=options)
                yield from bio._read_arrow_batches(leaf_options)
            return
        raise NotImplementedError(
            f"{type(self).__name__} has no tabular decoder. "
            "Construct via the format leaf (ParquetFile, CSVFile, …) "
            "to read Arrow record batches from this byte buffer."
        )

    def _write_arrow_batches(
        self,
        batches: "Iterable[pa.RecordBatch]",
        options: O,
    ) -> None:
        """Write batches via :meth:`open` on the dispatched leaf.

        Mirrors :meth:`_read_arrow_batches` — on a storage IO one
        open / dispatch / close cycle handles every format leaf. On
        a plain cursor without a format override raises
        :class:`NotImplementedError`.
        """
        if self._parent is None:
            with self.open(mode="wb") as bio:
                leaf_options = type(bio).check_options(options=options)
                bio._write_arrow_batches(batches, leaf_options)
            return
        raise NotImplementedError(
            f"{type(self).__name__} has no tabular encoder. "
            "Construct via the format leaf (ParquetFile, CSVFile, …) "
            "to write Arrow record batches into this byte buffer."
        )

    def flush(self) -> None:
        """Push buffered writes to the durable backing.

        Cursor IOs forward the flush to their bound parent; storage
        IOs go through :meth:`Disposable.commit` (default no-op
        unless a subclass overrides).
        """
        if self._parent is not None:
            try:
                self._parent.flush()
            except Exception:
                pass
            return
        return self.commit()

    def close(self, force: bool = False) -> None:
        """Release the IO; on :attr:`temporary`, discard pending
        writes instead of committing them.

        On a cursor with ``owns_holder=True`` the bound parent is
        closed too. Preserves the cursor position across the close
        — a reopen on the same instance lands at the byte the
        previous transaction left off.
        """
        super().close(force=force)

    def _release(self) -> None:
        """:class:`Disposable` release hook.

        On a cursor that owns its parent, close the parent. Drops
        the payload when :attr:`temporary` is set. Cursor IOs also
        clean up persisted-schema scratch via
        :meth:`Tabular._unpersist_schema`.
        """
        # Closing a cursor is the user's signal that the write
        # transaction is done — push any deferred writes through the
        # parent's :meth:`flush` before the ownership cascade. Buffered
        # remote paths (S3Path, VolumePath, DBFSPath, WorkspacePath)
        # batch writes inside an acquired window; without this hop the
        # ``with path.open("wb") as fh: fh.write(...)`` shape loses
        # data when the cursor borrows the parent (owns_holder=False).
        # Read cursors and unbuffered backends are unaffected — flush
        # is a no-op without dirty state.
        if self._parent is not None:
            try:
                self._parent.flush()
            except Exception:
                pass
        if self._parent is not None and self._owns_parent:
            try:
                self._parent.close()
            except Exception:
                pass

        if self.temporary:
            try:
                self.clear()
            except Exception:
                pass

        try:
            self._unpersist_schema()
        except AttributeError:
            pass

        super()._release()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def url(self) -> "URL":
        """Canonical URL identifying this holder."""
        if self._url is None:
            return URL.from_memory_address(self)
        return self._url

    @url.setter
    def url(self, value: "URL") -> None:
        # Fast path: when ``value`` is already a :class:`URL` with the
        # expected scheme, skip the ``with_scheme`` rebuild — Holder
        # construction off ``Path.parent`` / ``Path.joinpath`` always
        # produces same-schemed URLs and was paying for an unconditional
        # ``_replace`` copy here.
        if isinstance(value, URL) and value.scheme == self.scheme:
            self._url = value
            return
        self._url = URL.from_(value).with_scheme(self.scheme)

    # ------------------------------------------------------------------
    # Backing-shape predicates
    # ------------------------------------------------------------------

    @property
    def is_memory(self) -> bool:
        """True when the IO lives entirely in process memory.

        Cursor / format-leaf IOs delegate to the bound parent.
        Storage subclasses (:class:`Memory`) override directly.
        """
        if self._parent is not None:
            return self._parent.is_memory
        return False

    @property
    def is_local_path(self) -> bool:
        """True when the IO is a path on the local filesystem.

        Cursor / format-leaf IOs delegate to the bound parent.
        Storage subclasses (:class:`LocalPath`) override directly.
        """
        if self._parent is not None:
            return self._parent.is_local_path
        return False

    @property
    def is_remote_path(self) -> bool:
        """True when the IO is a path on a non-local backend.

        Cursor / format-leaf IOs delegate to the bound parent.
        Storage subclasses (remote paths) override directly.
        """
        if self._parent is not None:
            return self._parent.is_remote_path
        return False

    @property
    def is_local(self) -> bool:
        return self.is_memory or self.is_local_path

    @property
    def is_remote(self) -> bool:
        return self.is_remote_path

    @property
    def is_streaming(self) -> bool:
        """True when :attr:`size` reflects only the bytes pulled so far.

        Streaming holders (:class:`MemoryStream` over a live
        source) lazily pull bytes on read; their :attr:`size`
        grows as the cursor advances and may underreport the
        eventual total. Static holders (:class:`Memory`,
        :class:`Path`) know their full size up front so the
        default is ``False``.

        :class:`IO.read` checks this flag to decide whether to
        cap the requested byte count at :attr:`size` (static
        case — out-of-range reads would raise) or pass the
        request through unclamped (streaming case — the holder
        pulls until it has enough or EOF).
        """
        return False

    # ------------------------------------------------------------------
    # Cursorless I/O — the canonical surface :class:`BytesIO` consumes
    # ------------------------------------------------------------------

    def pread(self, n: int, pos: int, *, cursor: bool = False) -> bytes:
        """Positional read. Returns at most ``n`` bytes at *pos*.

        ``cursor=True`` reads from the internal cursor instead of *pos*
        and advances it past the bytes returned.
        """
        return bytes(self.read_mv(n, pos, cursor=cursor))

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        update_stat: bool = True,
        cursor: bool = False,
    ) -> int:
        """Positionally write. Returns bytes actually written.

        ``update_stat=False`` defers the post-write stat refresh to
        the caller — see :meth:`write_mv` for the bulk-write rationale.
        ``cursor=True`` writes at the internal cursor instead of *pos*
        and advances it by the bytes written.
        """
        return self.write_mv(
            _as_byte_mv(data), pos,
            update_stat=update_stat, cursor=cursor,
        )

    def memoryview(self) -> memoryview:
        """View over the holder's visible bytes."""
        return self.read_mv(-1, 0)

    # ------------------------------------------------------------------
    # Bytes / text convenience surface
    # ------------------------------------------------------------------

    def read_bytes(
        self,
        size: int = -1,
        offset: int = 0,
        *,
        cursor: bool = False,
    ) -> bytes:
        """Read ``size`` bytes starting at ``offset`` as :class:`bytes`.

        ``size=-1`` reads to EOF; ``offset`` accepts negative
        indices via :func:`_resolve_pos` (``-1`` → ``size``,
        ``-N`` → ``self.size - N``). ``cursor=True`` reads from the
        internal cursor and advances it past the bytes returned.
        """
        return bytes(self.read_mv(size, offset, cursor=cursor))

    def write_bytes(
        self,
        data: Any,
        offset: int = 0,
        *,
        size: int = -1,
        overwrite: bool = False,
        cursor: bool = False,
    ) -> int:
        """Splice ``data`` at ``offset``. Returns bytes written.

        ``size`` caps the byte count written — ``size=-1``
        (default) writes the entire source; ``size>=0`` writes at
        most ``size`` bytes. The cap is forwarded into each
        type-directed branch so a stream source stops reading
        after ``size`` bytes (no over-pull) and a bytes-like
        source slices its tail off before dispatching.

        ``overwrite`` declares that this write replaces every
        byte from ``offset`` onward. The holder ends at
        ``offset + bytes_written`` regardless of its prior size,
        and whole-blob remote backends collapse the implied
        ``truncate(...) + write(...)`` pair into one SDK call.

        Type-directed dispatch — bytes-like payloads
        (:class:`bytes`, :class:`bytearray`, :class:`memoryview`,
        and ``str`` after UTF-8 encoding) splice through
        :meth:`write_mv`; other :class:`Holder` instances route
        through :meth:`write_holder` (size-aware: small payloads
        write inline, large ones stream); file-like sources
        (anything exposing ``.read``) drain through
        :meth:`write_stream`. Subclasses override
        :meth:`_write_mv`, :meth:`_write_stream`, and / or
        :meth:`_write_holder` rather than this dispatch.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        if isinstance(data, IO):
            return self.write_holder(
                data, offset=offset, size=size, overwrite=overwrite,
                cursor=cursor,
            )
        if hasattr(data, "read"):
            return self.write_stream(
                data, offset=offset, size=size, overwrite=overwrite,
                cursor=cursor,
            )
        return self.write_mv(
            _as_byte_mv(data), offset, size=size, overwrite=overwrite,
            cursor=cursor,
        )

    def read_text(
        self,
        encoding: str = "utf-8",
        errors: str = "strict",
        *,
        size: int = -1,
        offset: int = 0,
        cursor: bool = False,
    ) -> str:
        """Decode ``size`` bytes at ``offset`` as text.

        ``cursor=True`` reads from the internal cursor and advances it.
        """
        return self.read_bytes(
            size, offset, cursor=cursor,
        ).decode(encoding, errors=errors)

    def write_text(
        self,
        text: str,
        encoding: str = "utf-8",
        errors: str = "strict",
        *,
        offset: int = 0,
        cursor: bool = False,
    ) -> int:
        """Encode ``text`` and splice at ``offset``. Returns bytes written.

        ``cursor=True`` writes at the internal cursor and advances it.
        """
        return self.write_bytes(
            text.encode(encoding, errors=errors), offset, cursor=cursor,
        )

    # ------------------------------------------------------------------
    # Cursorless read/write primitives — IO subclasses add cursor
    # ------------------------------------------------------------------

    def readinto(
        self, buffer: Any, *, offset: int = 0, cursor: bool = False,
    ) -> int:
        """Fill *buffer* with bytes starting at ``offset``.

        Returns the number of bytes written into *buffer* —
        ``min(len(buffer), self.size - offset)``. Matches the
        stdlib :meth:`io.RawIOBase.readinto` shape. ``cursor=True``
        reads from the internal cursor and advances it.

        On a cursor IO (``_parent is not None``) the default flips
        to cursor-anchored — stdlib ``readinto(buf)`` then matches
        the BinaryIO contract.
        """
        if not cursor and offset == 0 and self._parent is not None:
            cursor = True
        mv = memoryview(buffer)
        capacity = len(mv)
        if capacity == 0:
            return 0
        if cursor:
            offset = self._pos
        chunk = self.read_bytes(capacity, offset)
        n = len(chunk)
        if n:
            mv[:n] = chunk
        if cursor:
            self._pos = offset + n
        return n

    def readline(
        self, limit: int = -1, *, offset: int = 0, cursor: bool = False,
    ) -> bytes:
        """Read up to the next newline starting at ``offset``.

        Returns the line including the trailing ``\\n`` (or short
        when EOF lands first). ``limit >= 0`` caps the byte count.
        ``cursor=True`` reads from the internal cursor and advances
        it past the returned line. On a cursor IO the default flips
        to cursor-anchored.
        """
        if not cursor and offset == 0 and self._parent is not None:
            cursor = True
        if cursor:
            offset = self._pos
        total = self.size
        if offset >= total:
            return b""
        chunk_len = total - offset
        if limit is not None and limit >= 0:
            chunk_len = min(limit, chunk_len)
        if chunk_len <= 0:
            return b""
        chunk = self.read_bytes(chunk_len, offset)
        nl = chunk.find(b"\n")
        line = chunk if nl == -1 else chunk[: nl + 1]
        if cursor:
            self._pos = offset + len(line)
        return line

    def readlines(
        self, hint: int = -1, *, offset: int = 0, cursor: bool = False,
    ) -> list[bytes]:
        """Read every line from ``offset`` to EOF (or until ``hint`` bytes).

        ``cursor=True`` reads from the internal cursor and advances it
        past the bytes consumed. On a cursor IO the default flips to
        cursor-anchored.
        """
        if not cursor and offset == 0 and self._parent is not None:
            cursor = True
        lines: list[bytes] = []
        scan = self._pos if cursor else offset
        total = 0
        while True:
            line = self.readline(offset=scan)
            if not line:
                break
            lines.append(line)
            total += len(line)
            scan += len(line)
            if hint is not None and hint > 0 and total >= hint:
                break
        if cursor:
            self._pos = scan
        return lines

    # ------------------------------------------------------------------
    # Cursor — opt-in seekable surface
    # ------------------------------------------------------------------

    def tell(self) -> int:
        """Current cursor position."""
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek the internal cursor to *offset* relative to *whence*.

        Mirrors :meth:`io.IOBase.seek` with two ergonomic deviations:

        * ``seek(-1, SEEK_SET)`` is a "go to end" sentinel — pairs
          with ``read(-1)`` / "read all". Any other negative
          ``SEEK_SET`` offset raises :class:`ValueError`.
        * ``SEEK_CUR`` / ``SEEK_END`` with a negative offset that
          would land before byte 0 clamps to 0 instead of raising.
        """
        offset = int(offset)
        size = self.size
        if whence == 0:  # SEEK_SET
            if offset == -1:
                self._pos = size
            elif offset < 0:
                raise ValueError(
                    f"Negative SEEK_SET offset {offset!r} is invalid; "
                    f"use SEEK_END to count from the end."
                )
            else:
                self._pos = offset
        elif whence == 1:  # SEEK_CUR
            self._pos = max(0, self._pos + offset)
        elif whence == 2:  # SEEK_END
            self._pos = max(0, size + offset)
        else:
            raise ValueError(f"Invalid whence: {whence!r}")
        return self._pos

    def seekable(self) -> bool:
        return True

    # ---- structured binary helpers — fixed-width little-endian ---------
    #
    # On a cursor IO (``_parent is not None``) every fixed-width
    # primitive defaults to cursor-anchored: it reads / writes at
    # ``self._pos`` and advances. On a storage IO (no parent) the
    # default is positional from byte 0; callers pass ``offset=`` /
    # ``cursor=True`` for non-default behaviour.

    def _read_struct(
        self, fmt: str, n: int, offset: int, *, cursor: bool = False,
    ) -> Any:
        if not cursor and offset == 0 and self._parent is not None:
            cursor = True
        return struct.unpack(
            fmt, self.read_bytes(n, offset, cursor=cursor),
        )[0]

    def _write_struct(
        self, fmt: str, value: Any, offset: int, *, cursor: bool = False,
    ) -> int:
        if not cursor and offset == 0 and self._parent is not None:
            cursor = True
        return self.write_bytes(
            struct.pack(fmt, value), offset, cursor=cursor,
        )

    def read_int8(self, *, offset: int = 0, cursor: bool = False) -> int: return self._read_struct("<b", 1, offset, cursor=cursor)
    def write_int8(self, v: int, *, offset: int = 0, cursor: bool = False) -> int: return self._write_struct("<b", int(v), offset, cursor=cursor)
    def read_uint8(self, *, offset: int = 0, cursor: bool = False) -> int: return self._read_struct("<B", 1, offset, cursor=cursor)
    def write_uint8(self, v: int, *, offset: int = 0, cursor: bool = False) -> int: return self._write_struct("<B", int(v), offset, cursor=cursor)
    def read_int16(self, *, offset: int = 0, cursor: bool = False) -> int: return self._read_struct("<h", 2, offset, cursor=cursor)
    def write_int16(self, v: int, *, offset: int = 0, cursor: bool = False) -> int: return self._write_struct("<h", int(v), offset, cursor=cursor)
    def read_uint16(self, *, offset: int = 0, cursor: bool = False) -> int: return self._read_struct("<H", 2, offset, cursor=cursor)
    def write_uint16(self, v: int, *, offset: int = 0, cursor: bool = False) -> int: return self._write_struct("<H", int(v), offset, cursor=cursor)
    def read_int32(self, *, offset: int = 0, cursor: bool = False) -> int: return self._read_struct("<i", 4, offset, cursor=cursor)
    def write_int32(self, v: int, *, offset: int = 0, cursor: bool = False) -> int: return self._write_struct("<i", int(v), offset, cursor=cursor)
    def read_uint32(self, *, offset: int = 0, cursor: bool = False) -> int: return self._read_struct("<I", 4, offset, cursor=cursor)
    def write_uint32(self, v: int, *, offset: int = 0, cursor: bool = False) -> int: return self._write_struct("<I", int(v), offset, cursor=cursor)
    def read_int64(self, *, offset: int = 0, cursor: bool = False) -> int: return self._read_struct("<q", 8, offset, cursor=cursor)
    def write_int64(self, v: int, *, offset: int = 0, cursor: bool = False) -> int: return self._write_struct("<q", int(v), offset, cursor=cursor)
    def read_uint64(self, *, offset: int = 0, cursor: bool = False) -> int: return self._read_struct("<Q", 8, offset, cursor=cursor)
    def write_uint64(self, v: int, *, offset: int = 0, cursor: bool = False) -> int: return self._write_struct("<Q", int(v), offset, cursor=cursor)
    def read_f32(self, *, offset: int = 0, cursor: bool = False) -> float: return self._read_struct("<f", 4, offset, cursor=cursor)
    def write_f32(self, v: float, *, offset: int = 0, cursor: bool = False) -> int: return self._write_struct("<f", float(v), offset, cursor=cursor)
    def read_f64(self, *, offset: int = 0, cursor: bool = False) -> float: return self._read_struct("<d", 8, offset, cursor=cursor)
    def write_f64(self, v: float, *, offset: int = 0, cursor: bool = False) -> int: return self._write_struct("<d", float(v), offset, cursor=cursor)
    def read_bool(self, *, offset: int = 0, cursor: bool = False) -> bool: return bool(self.read_uint8(offset=offset, cursor=cursor))
    def write_bool(self, v: bool, *, offset: int = 0, cursor: bool = False) -> int: return self.write_uint8(1 if v else 0, offset=offset, cursor=cursor)

    # ------------------------------------------------------------------
    # Local-path bridge
    # ------------------------------------------------------------------

    def write_local_path(
        self,
        path: PathLike,
        *,
        pos: int = 0,
        n: int = -1,
        chunk_size: int = _COPY_CHUNK,
        cursor: bool = False,
    ) -> int:
        """Load ``path``'s bytes into this holder at ``pos``.

        ``n < 0`` reads the whole file; ``n >= 0`` caps the source
        bytes pulled at *n*. Streams in ``chunk_size`` slices so a
        large file doesn't materialize into memory.

        Pre-allocates the holder via :meth:`resize` when the source
        size is known up front (``n >= 0`` or local stat available),
        so the inner loop only writes — no per-chunk grow.
        """
        if cursor:
            pos = self._pos
        if pos < 0:
            raise ValueError("write_local_path pos must be >= 0")
        os_path = os.fspath(path)

        # Pre-grow the holder when we know the target end position.
        # n < 0 → fall back to source stat; failure is non-fatal (the
        # write loop still grows incrementally via write_mv → resize).
        target_end: int | None = None
        if n >= 0:
            target_end = pos + n
        else:
            try:
                target_end = pos + os.path.getsize(os_path)
            except OSError:
                pass
        if target_end is not None and target_end > self.size:
            self.resize(target_end)

        total = 0
        write_pos = pos
        remaining = n if n >= 0 else None
        with open(os_path, "rb") as fh:
            while True:
                want = chunk_size
                if remaining is not None:
                    if remaining <= 0:
                        break
                    want = min(want, remaining)
                chunk = fh.read(want)
                if not chunk:
                    break
                written = self.write_mv(memoryview(chunk), write_pos)
                if written == 0:
                    break
                write_pos += written
                total += written
                if remaining is not None:
                    remaining -= written
        if cursor and total:
            self._pos = pos + total
        return total

    def write_stream(
        self,
        src: Any,
        *,
        offset: int = 0,
        size: int = -1,
        overwrite: bool = False,
        batch_size: int = _COPY_CHUNK,
        cursor: bool = False,
    ) -> int:
        """Drain a binary source into this holder at ``offset``.

        Public entry point: accepts a yggdrasil :class:`IO[bytes]`,
        a stdlib :class:`typing.BinaryIO` (``io.BytesIO``,
        ``open(..., "rb")``, urllib3 responses, …), or any file-like
        carrying a ``.read``. Non-:class:`IO` sources are coerced
        via :meth:`IO.from_` so subclass-side :meth:`_write_stream`
        always receives a real :class:`IO[bytes]`.

        ``size`` caps the byte count drained from *src* —
        ``size=-1`` (default) reads to EOF; ``size>=0`` stops at
        ``size`` bytes (no over-pull from the source).

        ``overwrite`` truncates the holder's tail past
        ``offset + bytes_written``; whole-blob remote backends
        get a single atomic PUT instead of an explicit truncate
        followed by a write.

        ``batch_size`` is the read/write chunk size for the
        default streaming path (:data:`_COPY_CHUNK`, 1 MiB).
        Tune up for high-throughput remote sinks where the
        per-call overhead dominates, or down to bound peak
        memory on a slow consumer.
        """
        if cursor:
            offset = self._pos
        if offset < 0:
            raise ValueError("write_stream offset must be >= 0")
        if batch_size <= 0:
            raise ValueError("write_stream batch_size must be > 0")
        from yggdrasil.io.base import IO as _IO
        from yggdrasil.io.bytes_io import BytesIO as _YggBytesIO

        io_src = src if isinstance(src, _IO) else _YggBytesIO.from_(src)
        n = self._write_stream(
            io_src,
            offset=offset,
            size=size,
            overwrite=overwrite,
            batch_size=batch_size,
        )
        if cursor:
            self._pos = offset + n
        return n

    def _write_stream(
        self,
        src: "IO[bytes]",
        *,
        offset: int,
        size: int = -1,
        overwrite: bool = False,
        batch_size: int = _COPY_CHUNK,
    ) -> int:
        """Splice ``src``'s bytes into this holder starting at ``offset``.

        Default implementation: real chunked streaming — read
        ``batch_size`` bytes at a time and splice each chunk
        through :meth:`write_bytes`, so multi-GB sources never
        materialise as a single :class:`bytes` object in Python.
        ``size>=0`` caps the byte count so the source stops
        being read once the limit is reached. ``overwrite=True``
        truncates the tail beyond the final cursor on the last
        chunk so the holder ends exactly where the stream did.

        Subclass override hook: backends with an atomic
        whole-object upload (Volumes ``files.upload``, Workspace
        ``workspace.upload``, S3 ``PutObject``) replace this with
        a single request that consumes *src* directly — the
        chunked default is a strict loss for those.

        *src* is always a real :class:`IO[bytes]`; the public
        :meth:`write_stream` does the coercion so subclass code
        gets a stable type.
        """
        write_pos = offset
        total = 0
        remaining = size if size >= 0 else None
        while True:
            if remaining is not None:
                if remaining <= 0:
                    break
                chunk_size = min(batch_size, remaining)
            else:
                chunk_size = batch_size
            chunk = src.read(chunk_size)
            if not chunk:
                break
            n = self.write_bytes(chunk, offset=write_pos)
            write_pos += n
            total += n
            if remaining is not None:
                remaining -= n
        if overwrite and write_pos < self.size:
            self.truncate(write_pos)
        return total

    def write_holder(
        self,
        src: "Holder",
        *,
        offset: int = 0,
        size: int = -1,
        overwrite: bool = False,
        batch_size: int = _COPY_CHUNK,
        cursor: bool = False,
    ) -> int:
        """Splice another :class:`Holder`'s bytes into this one at ``offset``.

        Public entry point: validates the inputs, then dispatches
        to :meth:`_write_holder`. ``size`` caps the byte count
        pulled from *src* — ``size=-1`` (default) writes the
        whole source; ``size>=0`` writes the first ``size`` bytes.
        ``overwrite`` truncates the tail past
        ``offset + bytes_written`` (collapses ``truncate(...) +
        write_holder(...)`` into one operation for whole-blob
        remote backends). ``batch_size`` is forwarded to the
        streaming path for above-threshold payloads.

        Subclasses override the private hook to swap in a
        backend-aware fast path (Workspace / Volumes / S3 can
        hand the source straight to their atomic-upload SDK call
        without ever materialising the bytes in Python).
        """
        if cursor:
            offset = self._pos
        if offset < 0:
            raise ValueError("write_holder offset must be >= 0")
        if not isinstance(src, IO):
            raise TypeError(
                f"write_holder: expected an IO source, got "
                f"{type(src).__name__}. Pass through `write_bytes` for "
                f"bytes-like / file-like / stream inputs."
            )
        n = self._write_holder(
            src,
            offset=offset,
            size=size,
            overwrite=overwrite,
            batch_size=batch_size,
        )
        if cursor:
            self._pos = offset + n
        return n

    def _write_holder(
        self,
        src: "Holder",
        *,
        offset: int,
        size: int = -1,
        overwrite: bool = False,
        batch_size: int = _COPY_CHUNK,
    ) -> int:
        """Splice *src*'s bytes into this holder starting at ``offset``.

        Routing:

        - *src* is a :class:`Path` and we're at ``offset == 0``
          with no slicing → defer to :meth:`Holder._transfer_to`
          on *src* so the source-side fast paths fire
          (:func:`shutil.copyfile` for local→local,
          :meth:`Holder.write_local_path` for local→remote).
        - Sub-threshold payloads (under
          :data:`_INLINE_WRITE_THRESHOLD`, currently 4 MiB)
          splice through :meth:`write_mv` in one shot.
        - Larger payloads open an :class:`IO[bytes]` cursor on
          *src* and hand it to :meth:`_write_stream`, so backends
          with an atomic streaming uploader can replace the
          default chunked drain.

        ``size>=0`` caps the byte count pulled from *src*. The
        decision boundary (inline vs stream) uses
        ``min(src.size, size)`` so a 5 MiB source with
        ``size=1024`` still goes through the inline fast path.
        ``overwrite`` truncates the tail past
        ``offset + bytes_written``; ``batch_size`` controls the
        streaming chunk size when the threshold path is taken.

        Override when a backend can splice a foreign holder
        without going through Python bytes (e.g. an S3
        ``CopyObject`` when ``self`` and *src* share an
        underlying bucket).
        """
        from yggdrasil.io.path.path import Path

        if (
            offset == 0
            and size < 0
            and isinstance(src, Path)
            and type(src)._transfer_to is not IO._transfer_to
        ):
            # Path source with a specialised ``_transfer_to``
            # (LocalPath, AWS S3Path, …) → defer to its fast
            # paths instead of the generic inline / stream loop.
            src._transfer_to(self)
            written = src.size
            if overwrite and written < self.size:
                self.truncate(written)
            return written

        src_size = src.size
        effective = src_size if size < 0 else min(src_size, size)
        if effective < _INLINE_WRITE_THRESHOLD:
            return self.write_mv(
                src.read_mv(effective, 0), offset, overwrite=overwrite,
            )
        from yggdrasil.io.bytes_io import BytesIO as _YggBytesIO

        with _YggBytesIO(holder=src, mode="rb") as io_src:
            return self._write_stream(
                io_src,
                offset=offset,
                size=effective,
                overwrite=overwrite,
                batch_size=batch_size,
            )

    # ------------------------------------------------------------------
    # Byte transfer — upload / download to any byte sink
    # ------------------------------------------------------------------

    def upload(
        self, src: Any, *, size: int = -1, offset: int = 0,
    ) -> "Holder":
        """Upload *src*'s bytes into this holder.

        Symmetric to :meth:`download` but indexed from the
        destination side — ``dst.upload(src)`` makes the
        destination's content equal to the source's.

        *src* accepts any of:

        - :class:`Holder` (incl. any :class:`Path` subclass) —
          its bytes are pulled starting at *offset*.
        - :class:`IO` cursor — *offset* (if non-zero) seeks
          before ``read()``; otherwise the cursor's current
          position is honoured.
        - ``str`` / :class:`os.PathLike` — coerced via
          ``Path.from_(src)`` and treated as a holder.

        *size* and *offset* slice the source: ``size=-1`` (default)
        reads to EOF, ``size>=0`` caps the byte count, ``offset``
        is the starting offset. Slicing forces the whole-payload
        fast path in :meth:`_transfer_to` to defer to a bytes
        copy (the backend-specific shortcuts —
        ``shutil.copyfile``, ``write_local_path`` — don't expose
        a window).

        When *self* is a :class:`Path` whose URL ends in a
        trailing ``/`` (directory shape), the source's filename
        (``src.url.name`` or ``"download"`` for nameless holders)
        is joined onto it. No remote ``stat`` is issued — the
        trailing slash is a purely local, ``cp``-style hint.

        Returns the resolved destination so chains like
        ``dst.upload(src).read_bytes()`` work.

        Subclasses with a faster move (e.g. local→local via
        ``sendfile``, local→remote chunked stream) override
        :meth:`_transfer_to`, not this method.
        """
        from yggdrasil.io.path.path import Path

        source = IO.from_(src)
        target = _join_dir_hint(self, source)
        if isinstance(source, Path) and source.is_dir():
            # Directory tree: only a :class:`Path` target can hold
            # it. ``size`` / ``offset`` slicing is a file-only knob.
            if size != -1 or offset != 0:
                raise IsADirectoryError(
                    f"IO.upload: source {source.full_path()!r} is "
                    f"a directory; size / offset slicing applies to "
                    f"file uploads only."
                )
            if not isinstance(target, Path):
                raise IsADirectoryError(
                    f"IO.upload: source {source.full_path()!r} is "
                    f"a directory; target must be a Path to hold the "
                    f"tree, got {type(target).__name__}."
                )
            target.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                (target / child.name).upload(child)
            return target
        if size < 0 and offset == 0:
            # Whole-source: defer to ``write_bytes`` type dispatch —
            # bytes → ``write_mv``, :class:`IO` → ``write_holder``
            # (Path target uses :meth:`Path._write_holder` to fire
            # the ``_transfer_to`` fast paths: ``shutil.copyfile``
            # for local→local, ``write_local_path`` for local→remote).
            # ``overwrite=True`` truncates any tail past the source's
            # length.
            target.write_bytes(source, overwrite=True)
        else:
            target.write_bytes(
                source.read_bytes(size=size, offset=offset),
                overwrite=True,
            )
        return target

    def download(
        self, to: Any = None, *, size: int = -1, offset: int = 0,
    ) -> "Holder | IO":
        """Copy this holder's bytes to a local target.

        When *to* is :data:`None`, bytes land in the user's
        ``~/Downloads`` folder under :attr:`url.name` (or
        ``"download"`` for nameless holders), with browser-style
        ``(1)`` / ``(2)`` / … suffixes appended on name conflict.
        Otherwise *to* accepts the same shapes as :meth:`upload`
        (:class:`Holder`, :class:`IO`, ``str`` / :class:`os.PathLike`).
        *size* and *offset* slice this holder: ``size=-1`` (default)
        reads to EOF, ``size>=0`` caps the byte count, ``offset``
        is the starting offset. Returns the resolved target.
        """
        from yggdrasil.io.path.path import Path

        if to is None:
            to = _default_download_target(self.url.name)
        target = _join_dir_hint(IO.from_(to), self)

        if isinstance(self, Path) and self.is_dir():
            if size != -1 or offset != 0:
                raise IsADirectoryError(
                    f"IO.download: source {self.full_path()!r} is "
                    f"a directory; size / offset slicing applies to "
                    f"file downloads only."
                )
            if not isinstance(target, Path):
                raise IsADirectoryError(
                    f"IO.download: source {self.full_path()!r} is "
                    f"a directory; target must be a Path to hold the "
                    f"tree, got {type(target).__name__}."
                )
            return target.upload(self)
        if size < 0 and offset == 0:
            # Whole-source: defer to ``target.write_bytes(self)``
            # so the byte-pump goes through the same dispatch as
            # ``upload`` — Path target picks up the local fast
            # paths via :meth:`Path._write_holder`.
            target.write_bytes(self, overwrite=True)
        else:
            target.write_bytes(
                self.read_bytes(size=size, offset=offset),
                overwrite=True,
            )
        return target

    def _transfer_to(self, target: "Holder | IO") -> None:
        """Default transfer: pull self's bytes, push into *target*.

        Subclasses override to take advantage of backend-side fast
        paths (e.g. :class:`Path` uses :func:`shutil.copyfile` for
        local-to-local and :meth:`write_local_path` for
        local-to-remote so neither path materialises the full
        payload).
        """
        target.write_bytes(self.read_bytes())

    def _transfer_filename(self) -> str:
        """Filename used when joining onto a directory-shaped target.

        :class:`Memory` IOs address themselves with auto-minted
        ``mem://<host>/<hex_addr>`` URLs whose ``name`` is the
        object address — useless as a download filename. Fall back
        to ``"download"`` for memory-backed IOs and any IO whose
        URL has no nameable segment.
        """
        if self.is_memory:
            return "download"
        return self.url.name or "download"

    # ------------------------------------------------------------------
    # Hashing — full-payload digests over the durable bytes.
    # ------------------------------------------------------------------
    #
    # Lives on the holder rather than only on :class:`BytesIO` because
    # callers that only have a holder shouldn't have to open a cursor
    # just to compute a digest — the holder owns the bytes.

    def to_bytes(self) -> bytes:
        """Full payload as :class:`bytes` — alias for ``read_bytes()``."""
        return self.read_bytes()

    def getvalue(self) -> bytes:
        """Stdlib :class:`io.BytesIO` parity — alias for :meth:`to_bytes`."""
        return self.to_bytes()

    def decode(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        """Decode the whole payload as text. Cursorless — does not seek."""
        return self.to_bytes().decode(encoding, errors=errors)

    def to_base64(self, urlsafe: bool = True) -> str:
        """Return the payload base64-encoded as an ASCII ``str``.

        ``urlsafe=True`` (default) uses :func:`base64.urlsafe_b64encode`
        — ``-`` / ``_`` in place of ``+`` / ``/`` so the result drops
        cleanly into a URL or filename. ``urlsafe=False`` falls back
        to the standard alphabet.
        """
        import base64

        b = self.to_bytes()
        if urlsafe:
            return base64.urlsafe_b64encode(b).decode("ascii")
        return base64.b64encode(b).decode("ascii")

    def xxh3_64(self):
        """Return an :class:`xxhash.xxh3_64` instance over the payload.

        Always rebuilds an updatable :class:`xxhash.xxh3_64` so callers
        can keep mixing more bytes in if they want. The expensive
        part — walking the payload — is short-circuited via the
        cached digest; we just seed a fresh hasher with the cached
        value's bytes when available.
        """
        import xxhash
        return xxhash.xxh3_64(self.read_bytes())

    def xxh3_int64(self) -> int:
        """64-bit xxh3 hash of the payload as a signed int64.

        ``xxh3_64`` produces an unsigned 64-bit value; downstream Arrow
        schemas pin the field as ``int64``, so the digest is wrapped
        into signed range ``[-2**63, 2**63)``. Memoized against
        ``(_size, _mtime)`` — which every write path bumps via
        :meth:`_touch_stat` — so repeated reads pay the walk once.
        """
        if (
            self._xxh3_64_size != -1
            and self._xxh3_64_size == self._size
            and self._xxh3_64_mtime == self._mtime
        ):
            return self._xxh3_64_cached
        import xxhash
        v = xxhash.xxh3_64(self.read_bytes()).intdigest()
        if v >= 2 ** 63:
            v -= 2 ** 64
        self._xxh3_64_cached = v
        self._xxh3_64_size = self._size
        self._xxh3_64_mtime = self._mtime
        return v

    @property
    def xxh3_64_digest(self) -> bytes:
        """8-byte big-endian payload digest — equivalent to
        ``xxh3_64().digest()`` but served from the cached
        :meth:`xxh3_int64` so callers mixing the digest into a parent
        hash don't re-walk the payload."""
        v = self.xxh3_int64()
        if v < 0:
            v += 2 ** 64
        return v.to_bytes(8, "big")

    # ==================================================================
    # Cursor IO surface — stdlib :class:`typing.BinaryIO` interop
    # ==================================================================
    #
    # The methods below give every IO instance the stdlib file-like
    # surface (``read`` / ``write`` / ``readline`` / iteration / mode
    # / closed / fileno / …) so pandas, pyarrow, zipfile, urllib3 and
    # friends accept us as a binary handle without a separate facade.
    # On a cursor IO (``_parent is not None``) they read / write at
    # ``self._pos`` and advance; on a storage IO they read / write
    # against the IO's own bytes.

    @property
    def holder(self) -> "IO":
        """The bound parent IO (cursor case) or ``self`` (storage case).

        Backwards-compatible alias preserved from the pre-merge
        ``IO.holder`` property — call sites that drilled through a
        cursor to reach its backing storage keep working.
        """
        return self._parent if self._parent is not None else self

    @property
    def owns_holder(self) -> bool:
        """Whether closing self also closes the bound parent."""
        return self._owns_parent

    def remaining_bytes(self) -> int:
        """Bytes from the cursor to EOF on the active payload."""
        return self._active().size - self._pos

    def view(
        self,
        *,
        pos: int = 0,
        size: Optional[int] = None,
        mode: ModeLike = "rb",
    ) -> "IO":
        """Return a fresh, non-owning cursor over this IO.

        With *size* unset the view shares the same backing storage —
        zero copy, cursor seeded at *pos*. Useful for Parquet footer
        probes, zip directory walks, magic-byte sniffs.

        With *size* set, the view holds an in-memory copy of bytes
        ``[pos, pos+size)``. That's the right shape for a *bounded*
        sub-view that should not race with later mutations of the
        parent buffer.
        """
        if size is None:
            target = self._parent if self._parent is not None else self
            v = type(self)(holder=target, owns_holder=False, mode=mode)
            v._pos = int(pos)
            return v
        if size < 0:
            raise ValueError(f"view size must be >= 0, got {size!r}")
        # Bounded view: snapshot the requested range.
        payload = self.pread(int(size), int(pos))
        return type(self)(payload)

    # ---- codec auto-handling -------------------------------------------

    def _codec(self):
        """The codec on this buffer's :class:`MediaType`, or ``None``.

        Path-bound IOs learn their media type from the URL suffix at
        construction (``data.csv.gz`` → CSV + GZIP); callers that
        build a :class:`Memory` IO by hand can seed
        ``stat().media_type`` to opt the buffer into codec
        round-tripping.
        """
        try:
            mt = self.media_type
        except Exception:
            return None
        return getattr(mt, "codec", None) if mt is not None else None

    def _format_view(self) -> "IO":
        """A read-only IO over the *format* bytes.

        Uncompressed → non-owning :meth:`view` of ``self``. Codec
        present → fresh in-memory IO whose bytes are the decompressed
        payload. Caller closes the returned buffer.
        """
        codec = self._codec()
        if codec is None:
            return self.view(pos=0)
        return codec.decompress(self)

    def _format_input(self) -> "_FormatInputContext":
        """Context manager yielding a pyarrow-friendly input source.

        Local-path holder + no codec → :func:`pyarrow.memory_map`;
        anything else falls back to :meth:`_format_view`. The yielded
        value is whichever NativeFile / file-like object won the
        resolution; the context manager closes it on exit.
        """
        from yggdrasil.io.base import _FormatInputContext
        return _FormatInputContext(self)

    def _format_buffer(self) -> "_FormatBufferContext":
        """Context manager yielding a buffer to write raw format bytes into.

        Uncompressed holder → yielded buffer is ``self`` (pre-truncated
        to zero). Codec-tagged holder → a fresh in-memory IO; on exit
        the bytes are compressed and committed to ``self``.
        """
        from yggdrasil.io.base import _FormatBufferContext
        return _FormatBufferContext(self)

    def arrow_input_stream(self) -> "_ArrowInputStreamContext":
        """Context manager yielding the cheapest :class:`pa.NativeFile` over the payload.

        Local-path holder + no codec → :func:`pyarrow.memory_map`
        (zero-copy). Codec-tagged holder → decompress, then wrap in a
        :class:`pa.BufferReader`. Anything else → snapshot and wrap.
        The yielded stream is always a real :class:`pa.NativeFile`,
        so the caller hands it directly to pyarrow readers.
        """
        from yggdrasil.io.base import _ArrowInputStreamContext
        return _ArrowInputStreamContext(self)

    def arrow_output_stream(
        self, *, append: bool = False,
    ) -> "_ArrowOutputStreamContext":
        """Context manager yielding a :class:`pa.BufferOutputStream` writer.

        ``with bio.arrow_output_stream() as sink: writer(sink)``. The
        yielded sink accepts the format encoder's writes against a
        pure-Arrow in-memory buffer. On a clean exit the encoded
        bytes are committed to ``self`` via
        :meth:`_commit_format_payload`, which handles codec
        compression and the overwrite-vs-append disposition.
        """
        from yggdrasil.io.base import _ArrowOutputStreamContext
        return _ArrowOutputStreamContext(self, append=append)

    def _commit_format_payload(
        self,
        payload: "Any",
        *,
        append: bool = False,
    ) -> int:
        """Bulk-commit a fully-encoded format payload to this buffer.

        ``payload`` is anything :func:`memoryview`-able — typically
        a :class:`pyarrow.Buffer` from a
        :class:`pyarrow.BufferOutputStream` after the format encoder
        finishes. The codec on :attr:`media_type` (when set) is
        applied here, then the bytes land in ``self`` with one
        ``truncate`` + one ``write`` (overwrite) or one seek-to-end
        + one ``write`` (append).
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
            self.write_bytes(view, cursor=True)

        if restore_pos is not None:
            self._pos = min(restore_pos, self.size)
        return n

    # ---- mode predicates / stdlib BinaryIO surface ---------------------

    @property
    def mode(self) -> str:
        """POSIX mode string — stdlib :class:`typing.BinaryIO` parity.

        pandas / pyarrow / zipfile inspect ``.mode`` for substrings
        like ``"b"`` to dispatch binary vs text reads, so this
        surface returns the os-mode form (``"rb+"`` / ``"wb+"`` /
        ``"ab+"`` / ``"xb+"``) rather than the typed :class:`Mode`
        enum. The typed value is available via ``self._mode``.
        """
        return self._mode.os_mode

    def readable(self) -> bool:
        return self._mode.readable

    def writable(self) -> bool:
        return self._mode.writable

    def appendable(self) -> bool:
        """True when writes append at EOF — :data:`Mode.APPEND` only."""
        return self._mode.appendable

    @property
    def name(self) -> str:
        return str(self.url)

    def with_media_type(self, media_type: Any, *, copy: bool = False) -> "IO":
        """Stamp *media_type* onto the bound IO's metadata.

        With ``copy=False`` (the default), mutates ``self`` and returns
        it. ``copy=True`` allocates a fresh holder over the same bytes
        and returns a new IO over it.
        """
        mt = MediaType.from_(media_type, default=None) if media_type is not None else None
        if copy:
            payload = self.to_bytes()
            return type(self)(payload, media_type=mt)
        if mt is not None:
            target = self._parent if self._parent is not None else self
            target.media_type = mt
        return self

    def as_media(self, media_type: Any = None) -> "IO":
        """Return a typed Tabular leaf bound to this buffer's holder.

        Resolution: explicit *media_type* wins; otherwise the buffer's
        stamped media type is used. The leaf borrows the same backing
        storage so durable bytes are shared without a copy. When
        ``self`` is already an instance of the resolved leaf class,
        returns ``self`` unchanged.

        Raises :class:`KeyError` when no media type can be resolved or
        the resolved type has no registered Tabular leaf.
        """
        mt = MediaType.from_(media_type, default=None) if media_type is not None else None
        if mt is None:
            try:
                mt = self.media_type
            except Exception:
                mt = None
        if mt is None:
            raise KeyError(
                f"No media_type available for {self!r}. "
                "Pass media_type= explicitly or stamp it on the "
                "holder's IOStats via with_media_type()."
            )

        target = IO.class_for_media_type(mt)
        if isinstance(self, target):
            return self
        return target(
            holder=self._parent if self._parent is not None else self,
            owns_holder=False,
            mode=self._mode,
            media_type=mt,
        )

    @property
    def closed(self) -> bool:
        """Stdlib ``IO[bytes]`` parity — ``False`` while the bound
        backing is reachable.

        Stdlib semantics: ``closed`` means "file unusable for I/O."
        On a cursor the predicate flips only when teardown has dropped
        the parent reference; on a storage IO it always reads
        ``False`` (the storage owns its own bytes). Matters for
        pyarrow / pandas / polars / zipfile, which guard every op
        with an ``assert not closed``.
        """
        if self._parent is None:
            return False
        return self._parent is None

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        """Underlying fd if the holder exposes one. Raises otherwise."""
        target = self._parent if self._parent is not None else self
        fileno = getattr(target, "fileno", None)
        if fileno is None or fileno is IO.fileno:
            raise OSError(
                f"{type(self).__name__} has no underlying file descriptor."
            )
        return fileno()

    # ---- stdlib-style cursor read / write -----------------------------

    def read(self, size: int = -1) -> bytes:
        """Read up to *size* bytes from the cursor, advancing past them.

        Stdlib :meth:`io.RawIOBase.read` semantic: ``size < 0`` /
        ``None`` reads to EOF; otherwise reads up to ``size`` bytes,
        returning fewer at EOF.

        Static IOs (:class:`Memory`, :class:`Path`) know their full
        size up front; cap the request at ``self.size - self._pos``
        before dispatching so the storage's strict ``read_bytes``
        doesn't trip on an out-of-range window. Streaming IOs
        (:class:`MemoryStream` — ``is_streaming``) lazily pull bytes;
        forward the request unclamped so the storage pulls until it
        has enough or signals EOF.
        """
        active = self._active()
        if size is None or size < 0:
            out = active.read_bytes(-1, offset=self._pos)
        elif active.is_streaming:
            if size == 0:
                return b""
            out = active.read_bytes(size, offset=self._pos)
        else:
            remaining = max(0, active.size - self._pos)
            capped = min(size, remaining)
            if capped == 0:
                return b""
            out = active.read_bytes(capped, offset=self._pos)
        self._pos += len(out)
        return out

    def readall(self) -> bytes:
        """Read from cursor to EOF, advancing the cursor."""
        return self.read(-1)

    def readinto1(self, b: Any) -> int:
        return self.readinto(b)

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
            return self.write_bytes(
                b.encode("utf-8"), cursor=True,
            )
        if isinstance(b, (bytes, bytearray, memoryview)):
            return self.write_bytes(b, cursor=True)
        if hasattr(b, "read"):
            return self.write_stream(b, cursor=True)
        return self.write_bytes(memoryview(b), cursor=True)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    # ---- iteration / structured prefixed I/O --------------------------

    def __iter__(self) -> "IO":
        return self

    def __next__(self) -> bytes:
        line = self.readline(cursor=True)
        if not line:
            raise StopIteration
        return line

    def read_bytes_u32(self) -> bytes:
        """Length-prefixed (uint32 LE) bytes blob."""
        n = self.read_uint32()
        data = self.read(n)
        if len(data) != n:
            raise EOFError(f"expected {n} bytes, got {len(data)}")
        return data

    def write_bytes_u32(self, data: BytesLike) -> int:
        mv = memoryview(data)
        return self.write_uint32(len(mv)) + self.write_bytes(mv, cursor=True)

    def read_str_u32(self, encoding: str = "utf-8") -> str:
        """Length-prefixed UTF-8 string."""
        return self.read_bytes_u32().decode(encoding)

    def write_str_u32(self, s: str, encoding: str = "utf-8") -> int:
        return self.write_bytes_u32(s.encode(encoding))

    # ---- parse / decompress -------------------------------------------

    def json_load(self, *, media_type: Any = None, orient: Any = None) -> Any:
        """Parse the buffer, auto-detecting media type and compression.

        Resolution order for the media type:

        1. Explicit *media_type* kwarg.
        2. Cached :attr:`media_type` on the IO.
        3. Magic-byte sniff via :meth:`MediaType.from_io` — when this
           fires and the IO had no cached media type, the sniffed
           value is stamped onto the IO so future callers (codec
           handling, tabular dispatch) see it without re-sniffing.

        If the resolved type carries a codec the buffer is
        decompressed first and the inner mime is stamped onto the
        decompressed buffer. JSON / NDJSON / opaque-bytes payloads go
        through ``json.loads`` (or ``pandas.read_json`` when *orient*
        is set); every other registered format dispatches to its
        :class:`Tabular` leaf and returns ``read_pylist()``.
        """
        import json as _json
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
            target = self._parent if self._parent is not None else self
            try:
                target.media_type = mt
            except Exception:
                pass

        if mt is not None and mt.codec is not None:
            buf = mt.codec.decompress(self)
            inner_mt = MediaType(mime_type=mt.mime_type, codec=None)
            target = buf._parent if buf._parent is not None else buf
            try:
                target.media_type = inner_mt
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

        leaf_cls = IO.class_for_media_type(mt, default=None)
        if leaf_cls is None:
            text = buf.to_bytes().decode("utf-8", errors="replace")
            if not text.strip():
                return None
            return _json.loads(text)
        leaf = (
            buf if isinstance(buf, leaf_cls)
            else leaf_cls(
                holder=buf._parent if buf._parent is not None else buf,
                owns_holder=False,
            )
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

    def _commit_metadata(self) -> None:
        """Refresh the holder's :class:`IOStats` after a bulk write.

        Bulk writers route through ``options.sync_metadata=False`` for
        the inner per-batch call so each ``write_mv`` skips its
        post-write ``_touch_stat``. This single call at the end stamps
        a fresh ``mtime`` and flushes any buffered backend state — one
        ``time.time()`` (and one optional flush) per write op instead
        of one per batch.
        """
        target = self._parent if self._parent is not None else self
        try:
            target.touch_mtime()
        except AttributeError:
            pass
        try:
            target.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __bytes__(self) -> bytes:
        return self.read_bytes()

    def __bool__(self) -> bool:
        return True


def _looks_like_directory(url: URL) -> bool:
    """Trailing-slash check: ``True`` iff *url*'s path ends in ``/``.

    Used by the upload/download directory-hint helpers to apply
    ``cp``-style "into this directory" semantics without a remote
    stat round trip. The canonical signal is an empty trailing
    element in :attr:`URL.parts`.
    """
    parts = url.parts
    return bool(parts) and parts[-1] == ""


def _join_dir_hint(
    dst: "IO", src: "IO",
) -> "IO":
    """Apply ``cp``-style directory hint when *dst* is a slash-terminated Path.

    ``dst_dir_slash.upload(src)`` lands at ``dst_dir/<src.name>``;
    a non-Path *dst* (:class:`Memory`, cursor IO) or a non-directory
    path is returned untouched. The source's filename is taken from
    :meth:`IO._transfer_filename` so :class:`Memory` / nameless IOs
    fall back to ``"download"``.
    """
    from yggdrasil.io.path.path import Path

    if isinstance(dst, Path) and _looks_like_directory(dst.url):
        return dst / src._transfer_filename()
    return dst


def _default_download_target(name: str) -> "IO":
    """Resolve a fresh :class:`LocalPath` under ``~/Downloads`` for *name*.

    Browser-style default: drop the file under the user's
    Downloads folder, and on a name clash append ``(1)``, ``(2)``,
    … before the suffix until a free slot is found. The directory
    is created on demand; the file itself is not.
    """
    from yggdrasil.io.path.local_path import LocalPath

    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    candidate = os.path.join(downloads_dir, name)
    if not os.path.exists(candidate):
        return LocalPath(candidate)

    stem, suffix = os.path.splitext(name)
    i = 1
    while True:
        candidate = os.path.join(downloads_dir, f"{stem} ({i}){suffix}")
        if not os.path.exists(candidate):
            return LocalPath(candidate)
        i += 1


def _as_byte_mv(data: Union[bytes, bytearray, memoryview]) -> memoryview:
    """Normalize a bytes-like to a 1-D, contiguous, unsigned-byte memoryview.

    Centralizes the pwrite/write_bytes prelude so callers don't repeat
    the cast/contiguity dance and the rules stay in one place.
    """
    mv = memoryview(data)
    if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
        mv = mv.cast("B")
    if not mv.c_contiguous:
        mv = memoryview(bytes(mv))
    return mv


def _local_path_for_handle(obj: Any) -> Optional[str]:
    """Return the on-disk path for a local file handle, or ``None``.

    Recognises real file handles (``open("...", "rb")``,
    ``pathlib.Path.open()``) by their string ``.name`` attribute —
    stdlib file objects expose the underlying path there. Filters
    out anonymous streams whose ``.name`` is an int fd (sockets,
    pipes) or a bracketed sentinel (``"<stdin>"``, ``"<fdopen>"``)
    and anything whose ``.name`` doesn't actually exist on disk.

    Used by :meth:`IO.from_` to scrap the drain-into-Memory step
    for live local files — the resulting :class:`LocalPath`
    reads from the file system on demand, so a multi-GB handle
    never gets materialised.
    """
    name = getattr(obj, "name", None)
    if not isinstance(name, str):
        return None
    if name.startswith("<") and name.endswith(">"):
        return None
    try:
        if not os.path.isfile(name):
            return None
    except (OSError, ValueError):
        return None
    return name


# Backwards-compat alias. The pre-merge code split storage (Holder)
# from cursor (IO) into two classes; the merge collapsed them. Call
# sites that still import ``Holder`` continue to work — every Holder
# IS now an IO.
Holder = IO