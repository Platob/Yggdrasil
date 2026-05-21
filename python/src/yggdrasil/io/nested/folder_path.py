"""Filesystem folder of tabular files.

:class:`FolderPath` is a :class:`Tabular` over a directory whose
entries are tabular files (parquet, csv, arrow IPC, ndjson, …) and /
or sub-directories. The class has no byte buffer of its own — its
state is the bound :attr:`path` plus the children walk.

It is :class:`Singleton`-cached with a 15-minute TTL keyed on the
folder's URL string + the ``yggmeta`` flag, so every
``FolderPath(path=p)`` and every recursive
``iter_children`` /  partition-candidate probe with the same URL
hands back the same live instance. The schema cache,
``free_columns`` memo, and Hive ``static_values`` cache stay warm
across the whole ``send_many`` cache-scan loop — see
``_singleton_key``.

Reads
-----

:meth:`iter_children` walks :attr:`path` and yields one child per
non-private entry:

* Files resolve through :class:`MediaType.from_` (extension first,
  magic-byte fallback) to a :class:`Tabular` leaf, or to a generic
  :class:`BytesIO` if the resolution fails.
* Directories come back as a fresh :class:`FolderPath` of the same
  concrete class, so a tree of folders flattens transparently into
  one batch stream.

When the folder's resolved schema (sidecar / target / first
batch) declares any :class:`Field` children tagged
``partition_by=True``, the listing is Hive-aware: subdirectory
names of the form ``<col>=<val>/`` seed the child folder's
:attr:`Tabular.static_values` with the parsed KV (typed via
the schema's field dtype) so the inherited
:meth:`Tabular._should_prune_by_predicate` skips the whole
subtree when ``options.predicate`` is provably false on that
partition value — no descent, no read. When the predicate
constrains the partition column to a finite value set (via
:func:`extract_partition_filters`), the listing **doesn't even
walk the filesystem**: the candidate child paths are built
directly from the predicate's accepted values and probed one at
a time, so a 1000-partition cache with three live keys does
three ``stat`` calls instead of one ``iterdir`` of 1000 entries.

Writes
------

:meth:`make_child` mints ``part-{epoch_ms}-{seed}.{ext}`` under
:attr:`path` and returns a closed :class:`Tabular` leaf bound to
the new path. The default writer extension is configurable on
:class:`FolderOptions` via ``child_media_type``; the default is
Arrow IPC — single-pass column-oriented encoding, no row-group
footer to rewrite on append, and a write-side that matches the
in-memory batch shape almost 1:1, so it's the cheapest format to
land a stream of small batches into. Callers that want parquet
supply ``FolderOptions(child_media_type=MediaTypes.PARQUET)``.

When the incoming data carries a schema with ``partition_by``-
tagged children, the folder aggregates every batch into one
table, groups by the leading partition column, and routes the
per-value subsets to ``<base>/<col>=<val>/`` sub-folders, each
receiving its own ``part-*.{ext}`` leaf — mirrors the Hive read
shape. There is no ``partition_columns`` knob on
:class:`FolderOptions`: the schema is the only source of truth.

What "private" means
--------------------

Entries whose name starts with ``.`` are skipped. That covers
``.schema`` sidecars, ``.ygg/`` directories, ``.tmp`` fragments —
the dot-prefixed metadata convention without the class needing to
enumerate them.
"""

from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import Mapping
from threading import RLock
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.constants import MEDIA_TYPE_METADATA_KEY
from yggdrasil.data.options import CastOptions
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.data.enums.media_type import MediaType, MediaTypes
from yggdrasil.dataclasses import ExpiringDict
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.holder import IO
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.io.url import hive_cast_value, hive_encode, hive_split

if TYPE_CHECKING:
    from yggdrasil.io.path import Path


__all__ = ["FolderPath", "FolderOptions"]


# 15-minute TTL for the FolderPath singleton cache. The cache holds
# the schema, free_columns memo, and Hive static_values lookup that
# every send_many cache-scan loop walks; collapsing repeat
# constructors onto one instance keeps those warm across calls
# without leaking across the lifetime of long-running workers.
_FOLDER_PATH_SINGLETON_TTL: float = 15.0 * 60.0


@dataclasses.dataclass(frozen=True, slots=True)
class FolderOptions(CastOptions):
    """:class:`CastOptions` extended with folder-write knobs."""

    #: Media type of newly minted child files. Drives both the
    #: filename extension and the :class:`Tabular` leaf class
    #: (``ParquetFile`` / ``ArrowIPCFile`` / ``CSVFile`` / …) the folder
    #: dispatches to. Defaults to Arrow IPC — matches the in-memory
    #: batch shape, no row-group footer to rewrite, cheapest format
    #: to land a stream of small batches into. Pass
    #: ``MediaTypes.PARQUET`` (or any registered :class:`MediaType`)
    #: to override; a bare string (``"parquet"``) / extension
    #: (``"csv"``) / mime value (``"application/json"``) is coerced
    #: through :meth:`MediaType.from_`.
    #:
    #: Partition layout has no knob here — the schema owns it. Every
    #: :class:`yggdrasil.data.data_field.Field` child tagged
    #: ``partition_by=True`` becomes a Hive partition level, with the
    #: outermost-first ordering taken from the schema's child order.
    #: Source priority (most specific wins) when the folder needs to
    #: resolve the layout: the incoming batch's
    #: ``Schema.from_arrow(batch.schema)`` on writes, then
    #: :meth:`FolderPath.collect_schema` (the persisted ``.ygg/`` sidecar
    #: or the first-batch fallback), then ``options.target``.
    child_media_type: MediaType = MediaTypes.ARROW_IPC

    def __post_init__(self) -> None:
        CastOptions.__post_init__(self)
        coerced = MediaType.from_(self.child_media_type, default=None)
        if coerced is None:
            raise ValueError(
                f"FolderOptions.child_media_type must coerce to a MediaType; "
                f"got {self.child_media_type!r}. Pass one of "
                f"MediaTypes.ARROW_IPC / .PARQUET / a registered extension "
                f"string, or a MediaType instance."
            )
        if coerced is not self.child_media_type:
            object.__setattr__(self, "child_media_type", coerced)


class FolderPath(IO[bytes, FolderOptions]):
    """:class:`Tabular` over a directory of tabular files.

    Inherits :class:`Holder` so it registers in the cross-cutting
    media-type registry alongside byte-backed leaves. The byte
    primitives raise :class:`NotImplementedError` — a folder is a
    logical container, not a positional buffer; navigate via
    :meth:`iter_children` / :meth:`read_arrow_batches` instead.

    :class:`Singleton`-cached with a 15-minute TTL, keyed on the
    backing URL string + the ``yggmeta`` flag (see
    :meth:`_singleton_key`). Every ``FolderPath(path=p)`` and every
    recursive partition-candidate probe with the same URL collapses
    to one live instance, so the on-instance schema cache,
    ``free_columns`` memo, and Hive ``static_values`` cache stay
    warm across the whole ``send_many`` cache-scan loop without
    leaking across long-running workers.
    """

    mime_type: ClassVar[MimeTypes] = MimeTypes.FOLDER

    __slots__ = (
        "path",
        "_yggmeta_enabled",
        "_predicate_free_cols",
        "_static_values_cache",
        "_initialized",
    )

    #: Sidecar directory name under :attr:`path` where folder-level
    #: metadata (the dumped schema, future per-format hints) lives.
    #: Dot-prefixed so :meth:`iter_children`'s skip-private filter
    #: drops it from data listings without an explicit exclude.
    YGGMETA_DIRNAME: ClassVar[str] = ".ygg"

    #: Filename of the schema sidecar inside :attr:`YGGMETA_DIRNAME`.
    #: Zero-row Arrow IPC file — the schema (with every Field tag —
    #: ``partition_by``, ``cluster_by``, primary keys, comments, …)
    #: round-trips byte-for-byte through pyarrow.
    YGGMETA_SCHEMA_FILENAME: ClassVar[str] = "schema.arrow"

    # 15-minute cache shared across every FolderPath. The default
    # Singleton ``_INSTANCES`` is process-lifetime — overriding to
    # an :class:`ExpiringDict` with TTL + bounded capacity stops
    # long-running ingestion workers from accumulating one entry
    # per ever-seen cache root.
    _SINGLETON_TTL: ClassVar[Any] = _FOLDER_PATH_SINGLETON_TTL
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=_FOLDER_PATH_SINGLETON_TTL, max_size=10_000,
    )
    _INSTANCES_LOCK: ClassVar[RLock] = RLock()

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        """Key by ``(cls, url_str, yggmeta_enabled)``.

        The default ``(cls, args, sorted_kwargs)`` would split
        instances by every cosmetic difference in how the caller
        spelled the path (``"foo"`` vs ``URL.from_("foo")`` vs a
        live :class:`Path`) and by transient kwargs like
        ``tabular_parent`` — the singleton would never coalesce in
        practice. Two FolderPaths backed by the same URL serve the
        same on-disk state, so they share one instance regardless
        of how the caller phrased the constructor.
        """
        raw = kwargs.get("path")
        if raw is None and args:
            raw = args[0]
        if raw is None:
            raw = kwargs.get("data")
        if raw is None:
            raw = kwargs.get("url")
        # Canonicalise through :class:`Path` so every spelling of
        # the same on-disk root resolves to one key — str, pathlib,
        # URL, and live :class:`Path` instances all flow through the
        # backend-specific ``full_path``. Live :class:`Path` short-
        # circuits the construction so the singleton-cached Path
        # itself does the dispatch.
        if hasattr(raw, "full_path"):
            url_key = raw.full_path()
        else:
            from yggdrasil.io.path.path import Path as _Path
            try:
                url_key = _Path.from_(raw).full_path()
            except Exception:
                # Last resort: fall back to the literal value so we
                # at least share instances within the misuse case
                # rather than building a new one per call.
                url_key = repr(raw)
        yggmeta = kwargs.get("yggmeta", True)
        return (cls, url_key, bool(yggmeta))

    @classmethod
    def options_class(cls):
        return FolderOptions

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        tabular_parent: "Tabular | None" = None,
        yggmeta: bool = True,
        **kwargs: Any,
    ) -> None:
        """Bind to a folder path. No I/O.

        ``data`` and ``path`` accept the same shape; ``path`` wins
        when both are supplied. ``tabular_parent`` rides through to
        the :class:`Tabular` slot — set by the enclosing folder when
        it yields this one as a child.

        Partition KV doesn't need a constructor seed: a Hive-shaped
        sub-folder at ``<base>/partition_key=42/`` reports
        ``{"partition_key": 42}`` via :attr:`static_values` directly
        from the bound :attr:`path` URL — see :attr:`static_values`
        for the parsing contract.

        ``yggmeta`` (default ``True``) enables the dot-prefixed
        sidecar at ``<path>/.ygg/``. :meth:`_persist_schema` (the
        Tabular write hook) dumps the schema there so
        :meth:`_collect_schema` returns the typed :class:`Schema`
        (with every :class:`Field` tag — ``partition_by``,
        ``cluster_by``, primary keys, …) on the next read without
        re-opening a data leaf. Disable for ad-hoc folders where the
        sidecar would be noise (the sidecar folder itself constructs
        with ``yggmeta=False`` so we never recurse).
        """
        # Singleton-cached re-entry: :meth:`Singleton.__new__` returns
        # the live instance when the (url, yggmeta) key already
        # resolved one — Python still calls ``__init__`` after that,
        # but a second pass through here would reset every
        # on-instance cache (schema, free_cols memo, static_values)
        # we built earlier. Bail out so the cached state stays warm.
        if getattr(self, "_initialized", False):
            if tabular_parent is not None:
                # Late parent linkage from ``adopt_child`` — record it
                # so the schema / free_cols walk can find an
                # already-warm ancestor without overwriting cleaned
                # caches.
                self.tabular_parent = tabular_parent
            return

        # Resolve the path first; we hand the folder's URL up to
        # :class:`Holder` so the URL-keyed surfaces (singleton key,
        # repr, equality) line up with the underlying path.
        raw = path if path is not None else data
        if raw is None:
            raise ValueError(
                f"{type(self).__name__} requires a path; got None. "
                "Pass path=... or a path-ish positional."
            )

        from yggdrasil.io.path.path import Path as _Path
        self.path: "Path" = raw if isinstance(raw, _Path) else _Path.from_(raw)
        self._yggmeta_enabled: bool = bool(yggmeta)
        # Predicate-id-keyed cache of :func:`free_columns` results.
        # Walks the AST once per predicate; the recursive child reads
        # share the same predicate instance and walk the parent chain
        # via :meth:`_free_cols_for` to find this slot — so a 64-OR
        # batch lookup pays the walk once instead of 64+ times.
        self._predicate_free_cols: "tuple[int, tuple[str, ...]] | None" = None
        # Hive partition KV cache — :attr:`static_values` walks the
        # URL segments and casts each value to the schema-declared
        # dtype on every call. The URL is immutable, the schema is
        # itself cached, and the call sits in the hot prune loop
        # (one ``static_values`` per ``matches_static`` per child),
        # so we materialise once and hand back the same dict. Reset
        # by :meth:`_persist_schema` when a write changes the
        # partition dtypes.
        self._static_values_cache: "Mapping[str, Any] | None" = None

        # Don't forward ``data`` / ``path`` / ``binary`` — Holder
        # would try to seed bytes from the directory path. The folder
        # is identity-only at this layer.
        super().__init__(
            url=self.path.url,
            tabular_parent=tabular_parent,
            **kwargs,
        )
        self._initialized = True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self.path!r})"

    # ==================================================================
    # Holder byte primitives — folder is a directory, not a buffer
    # ==================================================================

    def _read_mv(self, n: int, pos: int) -> memoryview:
        raise NotImplementedError(
            f"{type(self).__name__} is a directory — not a positional "
            "byte buffer. Use iter_children() / read_arrow_batches() "
            "to walk its tabular leaves."
        )

    def _write_mv(self, data: "memoryview", pos: int) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} is a directory. Use make_child() / "
            "write_arrow_batches() to mint a tabular leaf inside it."
        )

    def reserve(self, n: int) -> None:
        raise NotImplementedError(f"{type(self).__name__} is a directory.")

    def truncate(self, n: int) -> int:
        raise NotImplementedError(f"{type(self).__name__} is a directory.")

    def _clear(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} is a directory.")

    @property
    def size(self) -> int:
        # Folders don't have a size in the Holder sense; the byte
        # primitives all raise, so report 0 for stat-like callers.
        return 0

    def _stat(self) -> "IOStats":
        # Delegate to the underlying path's stat — a folder's
        # metadata (existence, mtime, kind=DIRECTORY) lives on the
        # backing :class:`Path`. Override of :class:`Holder._stat`.
        return self.path.stat()

    @property
    def is_local_path(self) -> bool:
        return self.path.is_local_path

    @property
    def is_remote_path(self) -> bool:
        return self.path.is_remote_path

    @property
    def is_memory(self) -> bool:
        return False

    # ==================================================================
    # Context-manager protocol — folder leaves are stateless w.r.t.
    # open/close. Provide a no-op ``with`` block so call sites that
    # do ``with cache:`` (e.g. the session lookup helper) work
    # against either a BytesIO (real Disposable) or a folder.
    # ==================================================================

    def __enter__(self) -> "FolderPath":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    # ==================================================================
    # Static values — Hive partition KV parsed off the bound URL
    # ==================================================================

    @property
    def static_values(self) -> "Mapping[str, Any]":
        """Partition KV parsed from the folder's bound :attr:`path` URL.

        Walks the URL's segments (``self.path.url.parts``) and picks
        every Hive-shaped ``<col>=<val>`` chunk — a folder at
        ``<base>/year=2024/month=05/`` reports
        ``{"year": 2024, "month": 5}`` without any constructor seed.
        Values are cast to their declared dtype when a parent
        :class:`FolderPath` (or this folder itself) already has the
        schema cached, otherwise the decoded string passes through;
        the downstream :meth:`matches_static` prune is conservative
        and just degrades to a row-level filter on undecidable
        compares.

        Schema lookup walks ``tabular_parent`` first so partition
        children minted during a read reuse the root folder's
        cached schema instead of paying a per-child sidecar load.
        Only when no ancestor has the schema cached does the read
        fall through to :meth:`collect_schema`'s sidecar / first-
        batch path.

        The parsed mapping is memoised on the instance — URL parts
        are immutable for the bound path and the schema only flips
        when :meth:`_persist_schema` writes a new layout, which
        invalidates the cache. Calling once per ``matches_static``
        per child (the prune hot path) costs one ``dict`` read after
        the first invocation.
        """
        cached = self._static_values_cache
        if cached is not None:
            return cached
        parts = getattr(self.path.url, "parts", None) or ()
        if not parts:
            self._static_values_cache = {}
            return self._static_values_cache
        out: "dict[str, Any]" = {}
        schema: Any = ...
        for segment in parts:
            parsed = hive_split(segment)
            if parsed is None:
                continue
            col, raw = parsed
            if schema is ...:
                try:
                    schema = self.collect_schema()
                except Exception:
                    schema = None
            out[col] = hive_cast_value(
                raw, self._partition_dtype(col, schema=schema),
            )
        self._static_values_cache = out
        return out

    # ==================================================================
    # Yggmeta sidecar — folder-level metadata under ``<path>/.ygg/``
    # ==================================================================
    #
    # The sidecar tree is private to :meth:`_collect_schema` /
    # :meth:`_persist_schema` — they build the ``<path>/.ygg/...``
    # paths inline because the sidecar folder is opened at most twice
    # per FolderPath lifetime (one read on first ``_collect_schema``,
    # one write per persisted schema), and the recursion guard is just
    # ``self._yggmeta_enabled`` propagating ``yggmeta=False`` to the
    # sidecar's own FolderPath. No public surface for fishing it out
    # currently has a caller — add one back if a real consumer shows
    # up.

    def _collect_schema(self, options: FolderOptions) -> Any:
        """Ancestor-cache / sidecar / first-batch schema collection.

        Resolution order (most-specific wins):

        1. Walks :attr:`tabular_parent` and returns the first cached
           schema reached — partition children minted during a
           partitioned read share their root's schema, so the
           per-child sidecar load is redundant. The root pays for
           its sidecar once on first read; every nested child after
           that hits a process-local in-memory mapping.
        2. Reads ``.ygg/schema.arrow`` when present — the canonical
           record of what the folder's last writer persisted, with
           every :class:`Field` tag intact.
        3. Falls through to :meth:`Tabular._collect_schema` (first
           batch's schema) when no sidecar exists.

        Sidecar I/O is best-effort: unreadable / corrupt entries
        degrade to the next step, never raise.
        """
        parent = self.tabular_parent
        while parent is not None:
            cached = getattr(parent, "_schema_cache", ...)
            if cached is not ...:
                return cached
            parent = getattr(parent, "tabular_parent", None)

        if self._yggmeta_enabled:
            schema_path = (
                self.path / self.YGGMETA_DIRNAME / self.YGGMETA_SCHEMA_FILENAME
            )
            try:
                payload = schema_path.read_bytes()
            except Exception:
                payload = b""
            if payload:
                try:
                    from yggdrasil.data.schema import Schema
                    reader = pa.ipc.open_file(pa.BufferReader(payload))
                    return Schema.from_arrow(reader.schema)
                except Exception:
                    pass  # corrupt sidecar — fall through to first-batch
        # Missing-folder short-circuit: the base ``_collect_schema``
        # would recurse through ``_read_arrow_batches`` →
        # ``iter_children`` which itself stat's the path and returns
        # empty; collapse that whole round trip into a single
        # ``Schema.empty()`` here. Pays one ``exists`` syscall per
        # cold cache miss but avoids the ``next(iter(batches), None)``
        # detour the base fallback otherwise takes.
        if not self.path.exists():
            from yggdrasil.data.schema import Schema
            return Schema.empty()
        return super()._collect_schema(options)

    def _schema_for_arrow(self, arrow_schema: Any) -> Any:
        """Return a :class:`Schema` matching *arrow_schema*, reusing the cache.

        ``Schema.from_arrow`` walks every field of the pa.Schema and
        builds a fresh :class:`Field` per column — ~30 columns for
        :data:`RESPONSE_SCHEMA`, ~1 ms per call. The steady state of
        a hot cache-write loop is "every response carries the same
        shape", so rebuilding the Schema on every write is wasted
        work. This helper short-circuits to the cached
        :class:`Schema` (in :attr:`_schema_cache`, or any ancestor
        :attr:`tabular_parent`'s cache) when its arrow projection
        compares equal to *arrow_schema* — :meth:`pa.Schema.equals`
        is a C++-native comparison, so the win is the saved
        per-field rebuild.
        """
        node: Any = self
        while node is not None:
            cached = getattr(node, "_schema_cache", ...)
            if cached is not ... and cached is not None:
                try:
                    if cached.to_arrow_schema().equals(arrow_schema):
                        return cached
                except Exception:
                    pass
                break  # only the nearest cache holder; don't walk past it
            node = getattr(node, "tabular_parent", None)
        try:
            from yggdrasil.data.schema import Schema
            return Schema.from_arrow(arrow_schema)
        except Exception:
            return None

    def _persist_schema(
        self,
        schema: Any,
        *,
        media_type: "MediaType | None" = None,
    ) -> None:
        """Cache *schema* in memory AND dump to ``.ygg/schema.arrow``.

        Overrides :meth:`Tabular._persist_schema` so writers that
        call the standard hook automatically also stamp the sidecar
        — every :class:`Field` tag (``partition_by``, primary keys,
        comments) round-trips through the zero-row Arrow IPC file
        the next read picks up via :meth:`_collect_schema`.

        ``media_type`` is the on-disk format the writer just landed
        in (``options.child_media_type`` — Arrow IPC, Parquet, …).
        When supplied, it's stamped onto the persisted arrow schema
        under the ``b"media_type"`` metadata key as the canonical
        mime string; :attr:`Field.media_type` reads it back so a
        consumer loading the schema can tell which format the rows
        live in without scanning the part files. Defaults to ``None``
        (no stamp) for callers that don't carry a media-type — the
        sidecar then only carries the structural schema, exactly
        like before.

        The disk write is short-circuited when the in-memory cache
        already holds an equal schema **and** the persisted media
        type hasn't changed — the steady state of a hot cache-write
        loop is "every batch carries the same shape", so re-stamping
        ``.ygg/schema.arrow`` on every call would burn IO for no
        observable change. Side-car I/O stays best-effort: a failed
        write never poisons the data write that just landed.
        """
        if schema is None:
            return
        # Stamp the media-type onto a copy of the schema so the
        # in-memory cache, the sidecar bytes, and the next
        # ``Field.media_type`` read all see the same value. Building
        # a copy keeps the caller's Field untouched (it may be
        # singleton-shared via ``_schema_for_arrow``'s ancestor walk
        # or owned by an unrelated consumer). ``with_metadata``
        # replaces the dict wholesale, so merge with the existing
        # metadata first to preserve every schema-level tag
        # (``primary_key``, ``partition_by``, ``comment``, …).
        if media_type is not None and hasattr(schema, "with_metadata"):
            merged_meta = dict(schema.metadata or {})
            merged_meta[MEDIA_TYPE_METADATA_KEY] = (
                media_type.mime_type.value.encode("utf-8")
            )
            stamped = schema.with_metadata(merged_meta, inplace=False)
        else:
            stamped = schema
        prior = self._schema_cache
        super()._persist_schema(stamped)
        # Invalidate the partition-KV cache: ``static_values`` casts
        # raw URL values through the schema's partition dtypes, so a
        # schema flip can change the cast result.
        self._static_values_cache = None
        if not self._yggmeta_enabled:
            return
        # Skip the sidecar rewrite when the schema is unchanged from
        # the in-memory cache. ``Schema.__eq__`` compares structure +
        # tags + metadata so two ``Schema.from_arrow`` round trips of
        # the same source compare equal; the media_type lives in
        # metadata so a flip there already invalidates the equality.
        if prior is not ... and prior == stamped:
            return
        arrow_schema = (
            stamped.to_arrow_schema() if hasattr(stamped, "to_arrow_schema")
            else stamped
        )
        if arrow_schema is None:
            return
        sink = pa.BufferOutputStream()
        with pa.ipc.new_file(sink, arrow_schema):
            pass  # zero-row file, schema-only
        try:
            (
                self.path
                / self.YGGMETA_DIRNAME
                / self.YGGMETA_SCHEMA_FILENAME
            ).write_bytes(sink.getvalue().to_pybytes())
        except Exception:
            # Sidecar is opportunistic — never fail the data write.
            pass

    # ==================================================================
    # Children — read
    # ==================================================================

    def iter_children(
        self, options: "FolderOptions | None" = None,
    ) -> "Iterator[Tabular]":
        """Yield every non-private direct entry of :attr:`path`.

        Sub-directories come back as a fresh :class:`FolderPath`. File
        entries route through :class:`MediaType.from_` (extension
        first, magic-byte fallback) to a registered :class:`Tabular`
        leaf — :class:`ParquetFile` for ``.parquet``,
        :class:`ArrowIPCFile` for ``.arrow``, etc. Files that don't
        resolve fall back to a plain :class:`BytesIO`, which is
        useful for the children-surface walk but raises on the
        Tabular hooks (so they're transparently skipped by
        :meth:`_read_arrow_batches`).

        A missing folder yields nothing — no error. A stat failure
        mid-listing (race with a delete) silently skips the entry
        rather than aborting the whole walk.

        When *options* has ``partition_columns`` set, the listing is
        Hive-aware:

        - the leading column is consumed at this level; sub-folders
          matching ``<col>=<val>/`` are minted with the parsed value
          seeded into their :attr:`Tabular.static_values` (and the
          ``partition_columns`` tuple shrunk by one so the recursion
          consumes the next level next time);
        - when ``options.predicate`` constrains that column to a
          finite value set (via :func:`extract_partition_filters`),
          candidate sub-paths are built directly from the accepted
          values and probed — :meth:`Path.iterdir` is **not** called,
          so a wide partition tree stays cheap;
        - otherwise we ``iterdir()``-walk and discard any
          ``<col>=<val>/`` whose value is rejected by the predicate
          via the static-values prune.

        Non-partitioned folders and unset ``options`` keep the
        legacy flat-listing behaviour.
        """
        # Cheap early exit on a folder that hasn't been created yet
        # — without this :meth:`partition_columns` would call
        # :meth:`collect_schema`, which falls back to
        # ``super()._collect_schema`` → ``_read_arrow_batches`` →
        # back here, cycling forever on the very first write to a
        # fresh partition child whose sidecar doesn't exist yet.
        # ``Path.exists`` is one stat probe and we'd pay the same on
        # the unguarded ``iterdir`` anyway.
        if not self.path.exists():
            return
        partition_columns = self.partition_columns(options)
        if partition_columns:
            yield from self._iter_partition_children(options, partition_columns)
            return

        for entry in self.path.iterdir():
            if entry.name.startswith("."):
                continue

            try:
                is_dir = entry.is_dir()
            except Exception:
                continue

            if is_dir:
                yield self.adopt_child(type(self)(path=entry))
                continue

            child = self._leaf_for(entry)
            if child is None:
                continue
            yield self.adopt_child(child)

    def partition_columns(
        self,
        options: "FolderOptions | None" = None,
        *,
        schema: Any = None,
    ) -> "tuple[str, ...]":
        """Partition column names declared on the folder's schema.

        ``schema`` overrides the lookup when supplied (write path
        passes the just-peeked batch schema). Otherwise the resolved
        schema is :meth:`collect_schema` — which itself reads the
        ``.ygg/schema.arrow`` sidecar when present and falls back to
        the first-batch shape — with ``options.target`` as a final
        fallback when even the sidecar / first batch is empty (a
        fresh folder asked for its layout before any write).

        Columns already pinned by :attr:`static_values` get filtered
        out: a partition-key=v leaf would otherwise re-infer
        ``partition_key`` from its own data and recurse forever into
        ``partition_key=v/partition_key=v/…``. :class:`Field` parses
        every tag off the Arrow schema, so the helper just walks
        ``schema.children`` and picks the ``partition_by=True`` ones.
        """
        if schema is None:
            try:
                schema = self.collect_schema(options)
            except Exception:
                schema = None
        if schema is None or not getattr(schema, "children", None):
            schema = getattr(options, "target", None) if options is not None else None
        children = getattr(schema, "children", None) or ()
        consumed = set(self.static_values)
        return tuple(
            f.name for f in children
            if getattr(f, "partition_by", False) and f.name not in consumed
        )

    def _iter_partition_children(
        self,
        options: "FolderOptions",
        partition_columns: "tuple[str, ...]",
    ) -> "Iterator[Tabular]":
        """Hive-aware listing variant — see :meth:`iter_children`.

        ``partition_columns[0]`` is the column this level resolves;
        descendants re-resolve the layout from their own
        :attr:`static_values` (the consumed head gets filtered out
        of the schema-driven lookup, so the next level naturally
        picks up the remaining partition columns). Leaf files at
        this level (plain ``part-*.<ext>``, no ``<col>=<val>``
        prefix) still pass through so a partially-populated tree
        (e.g. a freshly minted folder where the writer happened to
        land a non-partitioned leaf alongside the partition dirs)
        reads cleanly.
        """
        head = partition_columns[0]
        accepted = self._accepted_partition_values(options, head)

        if accepted is not None:
            yield from self._iter_partition_candidates(head, accepted)
            return

        for entry in self.path.iterdir():
            if entry.name.startswith("."):
                continue
            try:
                is_dir = entry.is_dir()
            except Exception:
                continue
            if is_dir:
                parsed = hive_split(entry.name)
                if parsed is not None and parsed[0] == head:
                    value = hive_cast_value(
                        parsed[1], self._partition_dtype(head),
                    )
                    yield self._mint_partition_child(entry, head, value)
                else:
                    # Non-Hive sub-folder mixed in (older layout,
                    # operator scratch dir, …): yield it as a plain
                    # child so the walker still descends, but don't
                    # seed any partition KV — the predicate prune
                    # falls back to the row-level filter for it.
                    yield self.adopt_child(type(self)(path=entry))
                continue

            child = self._leaf_for(entry)
            if child is None:
                continue
            yield self.adopt_child(child)

    def _iter_partition_candidates(
        self,
        column: str,
        values: "frozenset[Any]",
    ) -> "Iterator[Tabular]":
        """Probe ``<base>/<column>=<val>/`` directly for each accepted value.

        Skips :meth:`Path.iterdir` entirely — N stat probes against a
        deterministic path layout instead of a full directory walk.

        The existence check is taken here, before minting the child:
        on a 64-OR batch lookup against a partially-populated cache
        the hot shape is "most candidates miss". Skipping the
        FolderPath allocation + adoption + downstream ``_read_arrow_batches``
        for misses cuts ~95 us per non-existent partition. When the
        candidate does exist we'd have paid a stat inside the
        child's ``iter_children`` anyway, so the early stat doesn't
        regress the hit path.

        ``hive_cast_value`` is **not** re-applied to the accepted
        values: :func:`extract_partition_filters` already hands back
        values typed against the predicate's literal types — running
        them through another ``pa.compute.cast`` for every of the N
        candidates wastes a kernel call per probe (and was the
        single biggest cost in the prune hot path before the
        skip).
        """
        for value in sorted(values, key=lambda v: (v is None, v)):
            encoded = hive_encode(value)
            candidate = self.path / f"{column}={encoded}"
            if not candidate.exists():
                continue
            yield self._mint_partition_child(candidate, column, value)

    def _mint_partition_child(
        self,
        path: "Path",
        column: str,
        value: Any,
    ) -> "Tabular":
        """Mint a child FolderPath bound to the partition sub-path.

        ``column`` / ``value`` are accepted for the call-site
        readability of "this child carries the partition <col>=<val>"
        — but no seed is stashed on the child. :attr:`static_values`
        parses the same KV straight off the bound :attr:`path` URL,
        so the next call to :meth:`partition_columns` on the child
        filters that column out of the schema-driven lookup
        naturally.
        """
        del column, value  # Parsed by :attr:`static_values` from path.
        return self.adopt_child(type(self)(path=path))

    def _accepted_partition_values(
        self,
        options: "FolderOptions | None",
        column: str,
    ) -> "frozenset[Any] | None":
        """Return the finite accepted-value set for *column*, or ``None``.

        Looks first at ``options.prune_values`` (caller-supplied
        explicit set), then falls back to walking
        ``options.predicate`` via :func:`extract_partition_filters`.
        Returns ``None`` when neither source pins the column to a
        finite set — the caller falls back to a full ``iterdir`` +
        per-child prune. ``options=None`` (the unpopulated default
        for :meth:`_has_tabular_children` and other "is there
        anything here?" probes) is treated the same way: no prune
        set, no predicate, walk everything.
        """
        if options is None:
            return None
        prune_values = options.prune_values
        if prune_values:
            forced = prune_values.get(column)
            if forced is not None:
                return frozenset(forced)
        predicate = options.predicate
        if predicate is None:
            return None
        try:
            from yggdrasil.io.tabular.execution.expr import (
                extract_partition_filters,
            )
            extracted = extract_partition_filters(predicate, (column,))
        except Exception:
            return None
        return extracted.get(column)

    def _partition_dtype(
        self, column: str, *, schema: Any = ...,
    ) -> "pa.DataType | None":
        """Best-effort partition dtype lookup for *column*.

        Uses *schema* when supplied (caller may have already paid
        the :meth:`collect_schema` cost — pass it through to avoid
        re-collecting once per segment), else calls
        :meth:`collect_schema` itself. Returns the column's arrow
        dtype, or ``None`` when the schema is empty / doesn't know
        the column. The static-value prune is conservative on
        undecidable predicates so a missing schema doesn't break
        correctness — it just leaves the partition value as the
        URL-decoded string and turns a partition-level skip into
        a row-level filter.
        """
        if schema is ...:
            try:
                schema = self.collect_schema()
            except Exception:
                schema = None
        if schema is None:
            return None
        try:
            field = schema[column]
        except Exception:
            return None
        if field is None:
            return None
        try:
            return field.arrow_type
        except Exception:
            return None

    def _leaf_for(self, entry: "Path") -> "Tabular | None":
        """Resolve a file entry to a :class:`Tabular` leaf.

        Returns ``None`` when the entry doesn't have a registered
        media type — the caller skips it. This is the contract
        :meth:`_read_arrow_batches` relies on to ignore non-tabular
        siblings without forcing the user to clean the directory.
        """
        # Side-effect import: ensures every primitive leaf (parquet /
        # csv / arrow / ndjson / json / xlsx) has registered itself
        # in the Tabular registry, so ``class_for_media_type`` can
        # actually find them.
        import yggdrasil.io.primitive  # noqa: F401

        try:
            mt = MediaType.from_(entry.url, default=None)
        except Exception:
            mt = None
        if mt is None:
            return None
        try:
            cls = IO.class_for_media_type(mt, default=None)
        except Exception:
            cls = None
        if cls is None:
            return None
        return cls(holder=entry, owns_holder=False)

    # ==================================================================
    # Children — write
    # ==================================================================

    def make_child(
        self, *, options: FolderOptions | None = None,
    ) -> "Tabular":
        """Mint a fresh tabular leaf bound to a fresh path under :attr:`path`.

        Filename shape: ``part-{epoch_ms}-{seed}.{ext}`` where ``ext``
        is ``options.child_media_type.full_extension``. The
        millisecond timestamp gives lexical-time ordering; a 2-byte
        seed (~65k-value space) breaks ties between writes that land
        in the same millisecond.

        The :class:`Tabular` leaf is dispatched directly from the
        media type via :meth:`IO.class_for_media_type`, so the
        write path doesn't go through the path-extension reverse-
        lookup. A media type with no registered leaf falls back to
        a raw :class:`BytesIO` so non-tabular extensions still get a
        working write.

        Returns a closed leaf. Caller opens it inside a ``with``
        block to write bytes.
        """
        opts = options or FolderOptions()
        # No mkdir pre-check: the leaf's first write goes through
        # Path's auto-create-parents path, so a missing folder
        # materializes on the byte landing instead of an extra
        # round trip we'd pay on every part mint.
        ext = opts.child_media_type.full_extension
        suffix = f".{ext}" if ext else ""
        epoch_ms = int(time.time() * 1000)
        seed = os.urandom(2).hex()
        name = f"part-{epoch_ms}-{seed}{suffix}"

        child_path = self.path / name
        cls = IO.class_for_media_type(opts.child_media_type, default=None)
        if cls is None:
            leaf: "Tabular" = BytesIO(holder=child_path, owns_holder=False)
        else:
            leaf = cls(holder=child_path, owns_holder=False)
        return self.adopt_child(leaf)

    # ==================================================================
    # Tabular hooks — derived from children
    # ==================================================================

    def _read_arrow_batches(
        self, options: FolderOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Chain :meth:`iter_children` into one Arrow batch stream.

        Sub-folders recurse through their own
        :meth:`_read_arrow_batches`; leaf children read in turn.

        Self + per-child predicate pruning runs through
        :meth:`Tabular._should_prune_by_predicate`: when
        ``options.predicate`` is provably false against the bound
        :attr:`static_values` (own seed + inherited from
        :attr:`tabular_parent`), the whole read is skipped without
        opening the directory; per-child the same check skips
        sub-folders / leaf files whose static surface decides the
        predicate negatively. Children without a static surface fall
        through unchanged (undecidable → read), so a vanilla folder
        without partition KV behaves exactly as before.

        When the folder's resolved schema declares any partition
        columns (via :class:`Field` ``partition_by`` tags), the
        listing is Hive-aware (see :meth:`iter_children`). Each
        yielded child carries the consumed partition KV on its
        :attr:`static_values`; the recursive read just forwards
        ``predicate`` / ``prune_values``, and the child's
        :meth:`partition_columns` filters the consumed column out
        of its own schema-driven lookup so the next level naturally
        drops to the remaining partitions (or to a flat write when
        the schema's partition set is exhausted).
        """
        predicate = options.predicate
        # Mutualise the predicate AST walk across the whole read
        # subtree. :meth:`_free_cols_for` checks this folder and
        # every ``tabular_parent`` for a cached entry keyed by
        # ``id(predicate)`` — the candidate-probe loop yields fresh
        # child FolderPaths, but they all share the same root and the
        # same predicate instance, so the walk happens exactly once
        # per ``read_arrow_batches`` call instead of N times per
        # iter_children iteration.
        free_cols = self._free_cols_for(predicate)
        if self._should_prune_by_predicate(options, free_cols=free_cols):
            return
        for child in self.iter_children(options=options):
            if child._should_prune_by_predicate(options, free_cols=free_cols):
                continue
            child_options = self._child_read_options(child, options)
            stream = child._read_arrow_batches(child_options)
            # Sub-folders recurse and apply the predicate at their
            # own leaf level; flat-format leaves (parquet / arrow IPC
            # / csv) don't filter rows themselves, so we run the
            # row-level predicate here in pyarrow's C++ kernels via
            # :meth:`Predicate.filter_arrow_batches`. The static-value
            # prune above already eliminated whole sub-trees the
            # predicate rejects on a partition column; this is the
            # residual non-partition filter.
            if predicate is not None and not isinstance(child, FolderPath):
                yield from predicate.filter_arrow_batches(stream)
            else:
                yield from stream

    def _free_cols_for(self, predicate: Any) -> "tuple[str, ...] | None":
        """Memoised :func:`free_columns` lookup keyed by ``id(predicate)``.

        Walks ``tabular_parent`` looking for an already-cached
        ``(id(predicate), free_cols)`` tuple; on miss, computes the
        free-column tuple once and stashes it on the *topmost*
        :class:`FolderPath` ancestor so every recursive child read
        finds it without re-walking the AST.

        Predicate AST nodes are immutable and short-lived (their
        lifetime is bounded by the read pipeline call stack), so
        id-keying is safe: the cached entry can't outlive the
        predicate instance that produced it.
        """
        if predicate is None:
            return None
        pid = id(predicate)
        # Search self + ancestors for a cached entry. The top-level
        # FolderPath is where we stash, but any ancestor along the way
        # may already hold the cache.
        node: Any = self
        topmost: "FolderPath" = self
        while node is not None:
            cached = getattr(node, "_predicate_free_cols", None)
            if cached is not None and cached[0] == pid:
                return cached[1]
            if isinstance(node, FolderPath):
                topmost = node
            node = getattr(node, "tabular_parent", None)
        try:
            from yggdrasil.io.tabular.execution.expr.nodes import free_columns
            cols = free_columns(predicate)
        except Exception:
            return None
        # Stash on the topmost FolderPath so every recursive child read
        # finds the cache on its first parent walk.
        try:
            topmost._predicate_free_cols = (pid, cols)
        except Exception:
            pass
        return cols

    def _child_read_options(
        self, child: "Tabular", options: FolderOptions,
    ) -> Any:
        """Forward predicate / prune-values to *child*'s read.

        Predicates and explicit prune-value sets need to ride down
        the tree — otherwise a partition-aware caller would have to
        re-seed them at every level, and a row-level predicate on a
        nested folder would silently no-op. Partition layout itself
        is **not** forwarded: each level re-resolves it from its
        own schema (sidecar / target / batch), with the consumed
        columns filtered out via :attr:`Tabular.static_values`. Leaf
        formats (parquet / arrow IPC / csv) accept ``predicate`` /
        ``prune_values`` via :class:`CastOptions`, the shared base.
        """
        opts_cls = child.options_class()
        kwargs: "dict[str, Any]" = {
            "predicate": options.predicate,
            "prune_values": options.prune_values,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            return opts_cls(**kwargs)
        except TypeError:
            # The leaf options class doesn't know about one of our
            # kwargs (e.g. an older subclass without ``prune_values``).
            # Drop the unknown bits and retry; the per-child predicate
            # still wins via row-level filtering after read.
            safe = {
                k: v for k, v in kwargs.items()
                if k in getattr(opts_cls, "__dataclass_fields__", ())
            }
            return opts_cls(**safe)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: FolderOptions,
    ) -> None:
        """Mint one child per rechunked group and drain into it.

        Mode dispatch:

        - **OVERWRITE / TRUNCATE** — drop every tabular sibling first,
          then write incoming as fresh part(s).
        - **AUTO / APPEND** (the default for tabular folders) — just
          add a new part file; existing parts are untouched.
        - **UPSERT / MERGE** — only meaningful with
          ``options.match_by``; see below.
        - **IGNORE** — no-op when the folder already holds tabular
          parts; otherwise behaves as APPEND.
        - **ERROR_IF_EXISTS** — raises when the folder is non-empty.

        Merge semantics (``options.match_by`` set):

        - **APPEND** — drop incoming rows whose key tuple already
          exists on disk; write only the survivors into a new part.
          Existing parts are not rewritten.
        - **UPSERT / MERGE** — collect incoming key tuples, walk
          existing parts and keep only rows whose key is *not* in
          that set, then write the survivors plus all incoming as
          fresh part(s). Old parts are dropped at the end so a
          failed write leaves them in place.

        ``options.byte_size`` / ``options.row_size`` route the actual
        bytes-to-disk write through
        :func:`rechunk_arrow_batches`, so the
        bin-packing applies regardless of which mode picked the
        rows. Setting both knobs unset keeps the legacy
        "one part file per write call" shape.

        Partition layout is auto-detected from the incoming data:
        when ``options.partition_columns`` is empty, the first
        batch's schema is parsed via :meth:`Schema.from_arrow` and
        any child :class:`Field` with ``partition_by=True`` becomes
        a partition level. ``Response.to_arrow_batch`` ships that
        metadata on its emitted schema (straight from
        ``RESPONSE_SCHEMA``), so the cache write path doesn't have
        to repeat ``partition_columns=`` on every call — the data
        carries its own partition shape. After every successful
        write the schema is persisted to ``.ygg/schema.arrow`` so
        subsequent reads round-trip the typed layout via
        :meth:`collect_schema` without re-opening a data leaf.
        """
        import pyarrow.compute as pc

        # Materialise the input once. The partition split needs to
        # see every batch at once (per-batch splitting scatters one
        # part file per (batch, value) pair into the same partition
        # directory), and even the flat-write branch needs the first
        # batch's schema for the ``.ygg/`` sidecar — buffering up
        # front is the simplest path that serves both.
        materialised = [b for b in batches if b.num_rows > 0]
        if not materialised:
            return
        first = materialised[0]
        first_schema = self._schema_for_arrow(first.schema)

        partition_columns = self.partition_columns(options, schema=first_schema)
        if partition_columns:
            # Partition split: union the materialised batches, group
            # by the leading partition column, and recurse into each
            # ``<self.path>/<col>=<val>/``. The recursion re-resolves
            # the partition layout from the same schema; the child's
            # ``static_values`` pins the consumed column so
            # :meth:`partition_columns` filters it out and only the
            # remaining levels fire. Multi-level layouts unwind
            # automatically, single-column layouts land in the flat
            # branch on the second hop. The partition column stays
            # in the written payload — Hive convention drops it,
            # but the cache layout keeps it so the row-level
            # predicate on read runs against the same in-payload
            # column without re-stamping from the directory name.
            head = partition_columns[0]
            if head not in first.schema.names:
                raise ValueError(
                    f"FolderPath partition write: batches are missing the "
                    f"partition column {head!r}. Schema has "
                    f"{first.schema.names!r}; partition_columns="
                    f"{partition_columns!r}."
                )

            # Apply mode semantics at the **folder level** before
            # recursing — without this, a partitioned OVERWRITE
            # would leave untouched partitions on disk (children
            # only see the partition values that exist in the
            # incoming batch, not the ones that previously lived
            # there) and IGNORE / ERROR_IF_EXISTS would silently
            # become "append at partition level".
            action = self._resolve_action(options.mode)
            if action is Mode.IGNORE and self._has_tabular_children():
                return
            if action is Mode.ERROR_IF_EXISTS and self._has_tabular_children():
                raise FileExistsError(
                    f"{type(self).__name__} already contains tabular files; "
                    f"refusing to write under mode={options.mode!r}."
                )
            if action is Mode.OVERWRITE:
                # Drop every existing child (files **and** partition
                # subtrees — see :meth:`_clear_tabular_children`) so
                # the new write fully replaces the old layout. The
                # recursive child writes below then land as a fresh
                # APPEND on the empty parent.
                self._clear_tabular_children()

            # Children get plain APPEND: the parent has already
            # honoured the mode (cleared for OVERWRITE, no-op for
            # APPEND / AUTO). UPSERT / MERGE pass through unchanged
            # so per-partition dedup against existing rows still
            # fires inside each child's flat branch.
            if action is Mode.OVERWRITE or action in (
                Mode.IGNORE, Mode.ERROR_IF_EXISTS,
            ):
                child_options = dataclasses.replace(options, mode=Mode.APPEND)
            else:
                child_options = options

            table = pa.Table.from_batches(materialised)
            column = table.column(head)
            for scalar in pc.unique(column.combine_chunks()):
                value = scalar.as_py()
                mask = (
                    pc.equal(column, scalar) if value is not None
                    else pc.is_null(column)
                )
                subset = table.filter(mask)
                if subset.num_rows == 0:
                    continue
                encoded = hive_encode(value)
                child = type(self)(path=self.path / f"{head}={encoded}")
                self.adopt_child(child)
                child._write_arrow_batches(
                    iter(subset.to_batches()), child_options,
                )
            self._persist_schema(first_schema, media_type=options.child_media_type)
            return

        action = self._resolve_action(options.mode)

        if action is Mode.IGNORE and self._has_tabular_children():
            return
        if action is Mode.ERROR_IF_EXISTS and self._has_tabular_children():
            raise FileExistsError(
                f"{type(self).__name__} already contains tabular files; "
                f"refusing to write under mode={options.mode!r}."
            )
        if action is Mode.OVERWRITE:
            self._clear_tabular_children()
            self._write_parts(materialised, options)
            self._persist_schema(first_schema, media_type=options.child_media_type)
            return

        match_by = list(options.match_by_keys or ())
        is_upsert = options.mode in (Mode.UPSERT, Mode.MERGE)

        if match_by and self._has_tabular_children():
            if is_upsert:
                self._merge_upsert(materialised, match_by, options)
            else:
                self._merge_append(materialised, match_by, options)
            self._persist_schema(first_schema, media_type=options.child_media_type)
            return

        # Plain APPEND (or empty folder): mint a fresh part and drain.
        self._write_parts(materialised, options)
        self._persist_schema(first_schema, media_type=options.child_media_type)

    def _write_parts(
        self,
        batches: Iterable[pa.RecordBatch],
        options: FolderOptions,
    ) -> None:
        """Mint one or more part files and drain *batches* into them.

        Honors ``options.byte_size`` / ``options.row_size`` for
        per-part rechunking; with neither set, drains the whole
        stream into a single part.
        """
        byte_size = getattr(options, "byte_size", None) or 0
        row_size = getattr(options, "row_size", None) or 0

        if byte_size > 0 or row_size > 0:
            from yggdrasil.arrow.cast import rechunk_arrow_batches

            rechunked = rechunk_arrow_batches(
                batches,
                byte_size=byte_size or None,
                row_size=row_size or None,
            )
            for batch in rechunked:
                if batch.num_rows == 0:
                    continue
                child = self.make_child(options=options)
                child.write_arrow_batches(
                    [batch], options=child.options_class()(),
                )
            return

        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            return

        child = self.make_child(options=options)
        child.write_arrow_batches(
            _chain_first(first, batch_iter),
            options=child.options_class()(),
        )

    # ==================================================================
    # Merge helpers — used when options.match_by is set
    # ==================================================================

    def _merge_append(
        self,
        batches: Iterable[pa.RecordBatch],
        match_by: "list[str]",
        options: FolderOptions,
    ) -> None:
        """APPEND with key-aware dedup.

        Rows whose ``match_by`` tuple already exists on disk are
        dropped from the incoming stream; survivors land in a new
        part file. Existing parts are not rewritten.
        """
        existing = self._collect_existing_keys(match_by)
        survivors = self._filter_batches_drop_keys(batches, match_by, existing)
        self._write_parts(survivors, options)

    def _merge_upsert(
        self,
        batches: Iterable[pa.RecordBatch],
        match_by: "list[str]",
        options: FolderOptions,
    ) -> None:
        """UPSERT / MERGE with key-aware rewrite.

        Drains incoming into memory once to capture the set of
        incoming keys, walks existing parts, drops every row whose
        key matches an incoming key, then writes the (filtered
        existing + incoming) stream into fresh parts and unlinks the
        old ones.
        """
        incoming = list(batches)
        if not incoming:
            return

        incoming_keys = self._collect_keys_from_batches(incoming, match_by)
        survivors_existing = self._iter_existing_filtered(match_by, incoming_keys)

        # Snapshot old part files before we touch anything new — we
        # only delete them after the rewrite has succeeded.
        old_files = self._tabular_files()

        merged_iter = _chain_iter(survivors_existing, iter(incoming))
        self._write_parts(merged_iter, options)

        for f in old_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    def _tabular_files(self) -> "list[Path]":
        out: "list[Path]" = []
        for entry in self.path.iterdir():
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    continue
            except Exception:
                continue
            out.append(entry)
        return out

    def _collect_existing_keys(
        self, match_by: "list[str]",
    ) -> "set[tuple]":
        keys: "set[tuple]" = set()
        for child in self.iter_children():
            if isinstance(child, FolderPath):
                continue
            try:
                for batch in child._read_arrow_batches(child.options_class()()):
                    self._extend_keys_from_batch(keys, batch, match_by)
            except Exception:
                continue
        return keys

    @staticmethod
    def _collect_keys_from_batches(
        batches: "Iterable[pa.RecordBatch]", match_by: "list[str]",
    ) -> "set[tuple]":
        keys: "set[tuple]" = set()
        for batch in batches:
            FolderPath._extend_keys_from_batch(keys, batch, match_by)
        return keys

    @staticmethod
    def _extend_keys_from_batch(
        keys: "set[tuple]",
        batch: pa.RecordBatch,
        match_by: "list[str]",
    ) -> None:
        if not all(c in batch.schema.names for c in match_by):
            return
        cols = [batch.column(c).to_pylist() for c in match_by]
        for row in zip(*cols):
            keys.add(row)

    def _filter_batches_drop_keys(
        self,
        batches: "Iterable[pa.RecordBatch]",
        match_by: "list[str]",
        drop_keys: "set[tuple]",
    ) -> "Iterator[pa.RecordBatch]":
        if not drop_keys:
            yield from batches
            return
        for batch in batches:
            yield from self._batch_filter_drop(batch, match_by, drop_keys)

    @staticmethod
    def _batch_filter_drop(
        batch: pa.RecordBatch,
        match_by: "list[str]",
        drop_keys: "set[tuple]",
    ) -> "Iterator[pa.RecordBatch]":
        if batch.num_rows == 0:
            return
        if not all(c in batch.schema.names for c in match_by):
            yield batch
            return
        cols = [batch.column(c).to_pylist() for c in match_by]
        mask = [row not in drop_keys for row in zip(*cols)]
        if all(mask):
            yield batch
            return
        if not any(mask):
            return
        keep_idx = [i for i, m in enumerate(mask) if m]
        table = pa.Table.from_batches([batch]).take(keep_idx).combine_chunks()
        for inner in table.to_batches():
            if inner.num_rows > 0:
                yield inner

    def _iter_existing_filtered(
        self,
        match_by: "list[str]",
        drop_keys: "set[tuple]",
    ) -> "Iterator[pa.RecordBatch]":
        """Walk existing leaves, yielding only rows whose key isn't in *drop_keys*."""
        for child in self.iter_children():
            if isinstance(child, FolderPath):
                continue
            try:
                stream = child._read_arrow_batches(child.options_class()())
            except Exception:
                continue
            yield from self._filter_batches_drop_keys(stream, match_by, drop_keys)

    def _has_tabular_children(self) -> bool:
        for _ in self.iter_children():
            return True
        return False

    def _clear_tabular_children(self) -> None:
        """Remove every non-private child — files **and** partition dirs.

        OVERWRITE semantics require the folder to be empty of
        tabular data before the new write lands, so a previous
        Hive-partitioned layout (any ``<col>=<val>/`` subtree)
        has to go too; skipping directories would leave stale
        partitions on disk that a subsequent read would still
        surface via :meth:`iter_children`. Dot-prefixed entries
        (the ``.ygg/`` sidecar, transient ``.tmp`` fragments) stay
        — :meth:`_persist_schema` immediately re-stamps the
        sidecar after the parent finishes the write.
        """
        for entry in self.path.iterdir():
            if entry.name.startswith("."):
                continue
            try:
                # ``Path.remove(recursive=True)`` handles both files
                # and directories (rmtree-style for dirs); the call
                # is best-effort so a transient race with a sibling
                # writer doesn't poison the overwrite.
                entry.remove(recursive=True, missing_ok=True)
            except Exception:
                pass

    # ==================================================================
    # Row-level delete
    # ==================================================================

    def _delete(self, predicate: Any, options: FolderOptions) -> int:
        """Walk children, filter each leaf in isolation, rewrite survivors.

        Streams leaf-by-leaf so a single match in one part file doesn't
        trigger a folder-wide rewrite — only the leaves that actually
        hold matched rows are rewritten. Sub-folders recurse. Files
        the predicate fully drains are unlinked outright; leaves with
        a mix of survivors and matches are rewritten as a fresh part
        and the original is unlinked once the new file is on disk.

        Per-batch filtering goes through
        :meth:`Predicate.filter_arrow_batches`, so the row work runs
        in pyarrow's C++ kernels — no Python row iteration.
        """
        not_pred = ~predicate
        deleted = 0
        for child in self.iter_children():
            if isinstance(child, FolderPath):
                deleted += child._delete(predicate, child.options_class()())
                continue
            deleted += self._delete_leaf(child, not_pred, options)
        return deleted

    def _delete_leaf(
        self,
        child: "Tabular",
        not_pred: Any,
        options: FolderOptions,
    ) -> int:
        """Filter rows in *child*; rewrite as a fresh part or unlink it."""
        survivors: "list[pa.RecordBatch]" = []
        kept_rows = 0
        total_rows = 0

        def _counted() -> "Iterator[pa.RecordBatch]":
            nonlocal total_rows
            for b in child._read_arrow_batches(child.options_class()()):
                total_rows += b.num_rows
                yield b

        try:
            for kept in not_pred.filter_arrow_batches(_counted()):
                kept_rows += kept.num_rows
                survivors.append(kept)
        except Exception:
            return 0

        deleted = total_rows - kept_rows
        if deleted == 0:
            return 0

        leaf_path = getattr(child, "_parent", None)
        if survivors:
            # Mixed: write survivors first, then drop the original. A
            # failed rewrite leaves the original intact.
            self._write_parts(iter(survivors), options)
        if leaf_path is not None:
            try:
                leaf_path.unlink(missing_ok=True)
            except Exception:
                pass
        return deleted

    # ==================================================================
    # Compaction — bin-pack small parts towards a target byte size
    # ==================================================================

    #: Default ``±`` tolerance band around *byte_size* for the
    #: "already close enough to the target" check. A 25 % cushion is
    #: wide enough that a Parquet file written from a slightly
    #: smaller-than-target Arrow table doesn't get rewritten the next
    #: pass (saving the read+write round-trip), and tight enough that
    #: a 50 %-of-target file still gets folded into a peer.
    OPTIMIZE_TOLERANCE: "ClassVar[float]" = 0.25

    def optimize(
        self,
        byte_size: "int | None" = None,
        *,
        target_media_type: "MediaType | str | Any" = MediaTypes.ARROW_IPC,
        tolerance: float = OPTIMIZE_TOLERANCE,
        **kwargs: Any,
    ) -> int:
        """Compact small part files into ``byte_size``-shaped bundles.

        Walks the tree under :attr:`path` and at every directory that
        holds part files, groups them by combined size and rewrites
        each group as a single fresh part. Two flavors of the pass:

        - ``byte_size=None`` (the default and the shape the local-cache
          compaction loop in :class:`Session` calls with) — collapses
          every directory with more than one part into a single file.
        - ``byte_size=N`` — first-fit-decreasing bin pack into bins of
          capacity ``N`` bytes. Parts whose size is within
          ``±tolerance`` of *N* (or already larger) are skipped: they
          are already "close enough" and rewriting them would just
          burn IO.

        ``target_media_type`` (a :class:`MediaType` or anything
        :meth:`MediaType.from_` accepts) selects the format the
        rewritten parts are encoded in. Defaults to Arrow IPC.

        Returns the number of new part files created. Idempotent: a
        second call on a tree that's already at target leaves
        nothing to do and returns ``0``.
        """
        media = MediaType.from_(target_media_type, default=MediaTypes.ARROW_IPC)
        return self._optimize_walk(
            self.path,
            byte_size=byte_size,
            target_media_type=media,
            tolerance=tolerance,
        )

    def _optimize_walk(
        self,
        directory: "Path",
        *,
        byte_size: "int | None",
        target_media_type: MediaType,
        tolerance: float,
    ) -> int:
        """Recurse into *directory*, compacting part files at each level."""
        try:
            entries = list(directory.iterdir())
        except FileNotFoundError:
            return 0

        subdirs: "list[Path]" = []
        files: "list[Path]" = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    subdirs.append(entry)
                else:
                    files.append(entry)
            except Exception:
                continue

        compacted = 0
        for sub in subdirs:
            compacted += self._optimize_walk(
                sub,
                byte_size=byte_size,
                target_media_type=target_media_type,
                tolerance=tolerance,
            )

        parts = [f for f in files if f.name.startswith("part-")]
        compacted += self._compact_parts(
            directory,
            parts,
            byte_size=byte_size,
            target_media_type=target_media_type,
            tolerance=tolerance,
        )
        return compacted

    def _compact_parts(
        self,
        directory: "Path",
        parts: "list[Path]",
        *,
        byte_size: "int | None",
        target_media_type: MediaType,
        tolerance: float,
    ) -> int:
        """Group *parts* by size and rewrite each group as one file."""
        if len(parts) < 2:
            return 0

        groups: "list[list[Path]]"
        if byte_size is None:
            # No size knob — keep the legacy "everything into one"
            # shape the cache compaction loop expects.
            groups = [list(parts)]
        else:
            sized: "list[tuple[int, Path]]" = []
            for p in parts:
                try:
                    size = int(p.size)
                except Exception:
                    continue
                # Already at target (within ``±tolerance``) or already
                # larger — leave it alone. Splitting an oversized part
                # is a different operation and out of scope here.
                if size >= byte_size * (1.0 - tolerance):
                    continue
                sized.append((size, p))

            if len(sized) < 2:
                return 0

            sized.sort(key=lambda t: t[0], reverse=True)
            groups = []
            bin_sizes: "list[int]" = []
            for size, path in sized:
                placed = False
                for idx, current in enumerate(bin_sizes):
                    if current + size <= byte_size:
                        groups[idx].append(path)
                        bin_sizes[idx] = current + size
                        placed = True
                        break
                if not placed:
                    groups.append([path])
                    bin_sizes.append(size)

        compacted = 0
        leaf_folder = FolderPath(path=directory)
        write_options = FolderOptions(
            mode=Mode.APPEND, child_media_type=target_media_type,
        )
        for group in groups:
            if len(group) < 2:
                continue
            tables: "list[pa.Table]" = []
            for f in group:
                leaf = leaf_folder._leaf_for(f)
                if leaf is None:
                    continue
                try:
                    tables.append(leaf.read_arrow_table())
                except Exception:
                    # Unreadable part — leave it on disk; another
                    # writer might still be flushing it.
                    tables = []
                    break
            if not tables:
                continue

            try:
                merged = pa.concat_tables(tables, promote_options="default")
            except TypeError:
                # pyarrow < 14 had no ``promote_options`` kwarg.
                merged = pa.concat_tables(tables, promote=True)

            # Write the merged table first; only after the new part
            # is on disk do we drop the originals. A failed write
            # leaves the source files intact.
            leaf_folder.write_arrow_table(merged, options=write_options)
            for f in group:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
            compacted += 1
        return compacted

    def _resolve_action(self, mode: Mode) -> Mode:
        # AUTO maps to APPEND for tabular folders: each write adds a
        # fresh part file alongside the existing ones, the way the
        # response cache and any "drop another batch into the
        # partition" workflow expects. OVERWRITE / TRUNCATE stay
        # destructive and are reserved for the explicit
        # "rewrite the whole folder" call.
        if mode is Mode.OVERWRITE or mode is Mode.TRUNCATE:
            return Mode.OVERWRITE
        if mode is Mode.AUTO or mode is Mode.APPEND:
            return Mode.APPEND
        if mode is Mode.IGNORE:
            return Mode.IGNORE
        if mode is Mode.ERROR_IF_EXISTS:
            return Mode.ERROR_IF_EXISTS
        if mode is Mode.UPSERT or mode is Mode.MERGE:
            return Mode.APPEND
        return Mode.APPEND


def _chain_first(
    first: pa.RecordBatch,
    rest: "Iterator[pa.RecordBatch]",
) -> "Iterator[pa.RecordBatch]":
    yield first
    yield from rest


def _chain_iter(*iters: "Iterable[pa.RecordBatch]") -> "Iterator[pa.RecordBatch]":
    for it in iters:
        yield from it
