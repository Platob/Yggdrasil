"""Filesystem folder of tabular files.

:class:`FolderIO` is a :class:`Tabular` over a directory whose
entries are tabular files (parquet, csv, arrow IPC, ndjson, …) and /
or sub-directories. The class has no byte buffer of its own — its
state is the bound :attr:`path` plus the children walk.

Reads
-----

:meth:`iter_children` walks :attr:`path` and yields one child per
non-private entry:

* Files resolve through :class:`MediaType.from_` (extension first,
  magic-byte fallback) to a :class:`Tabular` leaf, or to a generic
  :class:`BytesIO` if the resolution fails.
* Directories come back as a fresh :class:`FolderIO` of the same
  concrete class, so a tree of folders flattens transparently into
  one batch stream.

When :attr:`FolderOptions.partition_columns` is set the listing is
Hive-aware: subdirectory names of the form ``<col>=<val>/`` seed
the child folder's :attr:`Tabular.static_values` with the parsed
KV (typed via :attr:`FolderOptions.schema` when present) so the
inherited :meth:`Tabular._should_prune_by_predicate` skips the
whole subtree when ``options.predicate`` is provably false on
that partition value — no descent, no read. When the predicate
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
:class:`FolderOptions` via ``child_extension``; the default is
``"arrow"`` (Arrow IPC) — single-pass column-oriented encoding,
no row-group footer to rewrite on append, and a write-side that
matches the in-memory batch shape almost 1:1, so it's the
cheapest format to land a stream of small batches into. Callers
that want parquet supply ``FolderOptions(child_extension="parquet")``.

With ``partition_columns`` set on the write call, the folder
splits each incoming batch by the partition column values and
routes the per-value subsets to ``<base>/<col>=<val>/`` sub-folders,
each receiving its own ``part-*.{ext}`` leaf — mirrors the Hive
read shape so the same options round-trip.

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
import urllib.parse
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.data.enums.media_type import MediaType, MediaTypes
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.holder import IO
from yggdrasil.io.tabular.base import Tabular

if TYPE_CHECKING:
    from yggdrasil.io.path import Path


__all__ = ["FolderIO", "FolderOptions"]


# ---------------------------------------------------------------------------
# Hive partition layout helpers — ``<col>=<val>/`` directory encoding
# ---------------------------------------------------------------------------


def _hive_encode(value: Any) -> str:
    """Encode *value* as a filesystem-safe Hive partition value.

    ``None`` → ``"__HIVE_DEFAULT_PARTITION__"`` matching the Hive /
    Spark / Delta convention. Everything else is ``str(value)`` URL-
    quoted with the path-separator + ``=`` characters reserved so
    the encoded value can be split back unambiguously on a single
    ``=`` and never collides with a directory boundary.
    """
    if value is None:
        return "__HIVE_DEFAULT_PARTITION__"
    return urllib.parse.quote(str(value), safe="")


def _hive_decode(raw: str) -> Any:
    """Inverse of :func:`_hive_encode` — returns the URL-decoded string.

    The caller is responsible for casting the result to the partition
    column's declared dtype (the folder layer doesn't know the schema
    at parse time; the read pipeline casts once the batch lands).
    """
    if raw == "__HIVE_DEFAULT_PARTITION__":
        return None
    return urllib.parse.unquote(raw)


def _hive_split_name(name: str) -> "tuple[str, Any] | None":
    """Parse a Hive-encoded directory name into ``(column, value)``.

    Returns ``None`` when *name* doesn't match the ``<col>=<val>``
    convention so the caller can treat the entry as a plain (non-
    Hive) sub-folder.
    """
    if "=" not in name:
        return None
    col, _, raw = name.partition("=")
    if not col:
        return None
    return col, _hive_decode(raw)


def _cast_partition_value(value: Any, dtype: "pa.DataType | None") -> Any:
    """Cast a decoded Hive value to *dtype*, falling back to the raw string.

    Used when the folder is read with a typed schema in scope (cache
    layouts where ``partition_key`` is ``int64``, time partitions
    encoded as strings need to land as :class:`datetime`, …). When
    *dtype* is ``None`` or the cast raises (un-castable value), the
    decoded string passes through unchanged — the static-value prune
    is conservative on undecidable predicates so a no-op cast just
    forces the row-level filter to run.
    """
    if value is None or dtype is None:
        return value
    try:
        arr = pa.array([value]).cast(dtype, safe=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError, NotImplementedError):
        return value
    return arr[0].as_py()


def _partition_columns_from_schema(schema: Any) -> "tuple[str, ...]":
    """Return field names tagged ``partition_by=True`` on *schema*'s children.

    *schema* is a :class:`yggdrasil.data.schema.StructField` — :class:`Field`
    already parses every per-field tag (``partition_by``,
    ``cluster_by``, ``primary_key``, …) from the Arrow metadata, so
    we just walk ``schema.children`` and pick the ones whose
    :attr:`Field.partition_by` flag is set. Empty / schema-less
    inputs produce an empty tuple — the caller falls back to a
    flat-folder layout.
    """
    if schema is None:
        return ()
    children = getattr(schema, "children", None)
    if not children:
        return ()
    return tuple(
        f.name for f in children if getattr(f, "partition_by", False)
    )


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
    child_media_type: MediaType = MediaTypes.ARROW_IPC

    #: Hive-style partition column names, outermost first.
    #: ``("partition_key",)`` lays children out as
    #: ``<base>/partition_key=<val>/part-*.<ext>``;
    #: ``("year", "month")`` lays them out as
    #: ``<base>/year=<v1>/month=<v2>/part-*.<ext>``.
    #:
    #: On **read** the listing parses each ``<col>=<val>/`` directory
    #: name, seeds the parsed KV on the child's
    #: :attr:`Tabular.static_values`, and the inherited
    #: :meth:`Tabular._should_prune_by_predicate` skips the whole
    #: subtree when ``options.predicate`` is provably false against
    #: that value. When the predicate constrains the leading
    #: partition column to a finite value set (via
    #: :func:`yggdrasil.io.tabular.execution.expr.extract_partition_filters`),
    #: the listing builds candidate sub-paths from the accepted
    #: values and probes them directly instead of walking
    #: ``iterdir`` — N probes instead of one full scan.
    #:
    #: On **write** the folder splits incoming batches by the listed
    #: column values and routes each subset to its
    #: ``<col>=<val>/`` sub-folder.
    #:
    #: Empty tuple keeps the legacy flat-folder behaviour. The order
    #: matters: it dictates the directory nesting on disk.
    partition_columns: "tuple[str, ...]" = ()

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
        if self.partition_columns and not isinstance(self.partition_columns, tuple):
            object.__setattr__(
                self, "partition_columns", tuple(self.partition_columns),
            )


class FolderIO(IO[bytes, FolderOptions]):
    """:class:`Tabular` over a directory of tabular files.

    Inherits :class:`Holder` so it registers in the cross-cutting
    media-type registry alongside byte-backed leaves. The byte
    primitives raise :class:`NotImplementedError` — a folder is a
    logical container, not a positional buffer; navigate via
    :meth:`iter_children` / :meth:`read_arrow_batches` instead.
    """

    mime_type: ClassVar[MimeTypes] = MimeTypes.FOLDER

    __slots__ = ("path", "_partition_remainder", "_yggmeta_enabled")

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
        static_values: "Mapping[str, Any] | None" = None,
        yggmeta: bool = True,
        **kwargs: Any,
    ) -> None:
        """Bind to a folder path. No I/O.

        ``data`` and ``path`` accept the same shape; ``path`` wins
        when both are supplied. ``tabular_parent`` rides through to
        the :class:`Tabular` slot — set by the enclosing folder when
        it yields this one as a child. ``static_values`` rides
        through too: an aggregator (e.g. :class:`YGGFolderIO`)
        minting a per-partition leaf seeds the kv here so every
        descendant inherits the partition constants via the
        :attr:`Tabular.static_values` parent chain — no extra
        per-batch stamping needed to assert the column equality.

        ``yggmeta`` (default ``True``) enables the dot-prefixed
        sidecar at ``<path>/.ygg/`` — see :attr:`yggmeta`. Writes
        dump the schema there via :meth:`_persist_yggmeta_schema`
        so :meth:`collect_schema` can return the typed
        :class:`Schema` (with every :class:`Field` tag —
        ``partition_by``, ``cluster_by``, primary keys, …) on the
        next read without re-opening a data leaf. Disable for
        ad-hoc folders where the sidecar would be noise (the
        sidecar folder itself constructs with ``yggmeta=False``
        so we never recurse).
        """
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
        # Outer read's ``partition_columns`` tail, stashed by the
        # parent when it yields this child via
        # :meth:`_mint_partition_child` so :meth:`_read_arrow_batches`
        # can forward the remaining levels on its recursive read
        # without re-reading the parent's options. Empty tuple for
        # non-Hive children and top-level instances.
        self._partition_remainder: "tuple[str, ...]" = ()
        self._yggmeta_enabled: bool = bool(yggmeta)

        # Don't forward ``data`` / ``path`` / ``binary`` — Holder
        # would try to seed bytes from the directory path. The folder
        # is identity-only at this layer.
        super().__init__(
            url=self.path.url,
            tabular_parent=tabular_parent,
            static_values=static_values,
            **kwargs,
        )

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

    def __enter__(self) -> "FolderIO":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    # ==================================================================
    # Yggmeta sidecar — folder-level metadata under ``<path>/.ygg/``
    # ==================================================================

    @property
    def yggmeta(self) -> "FolderIO | None":
        """Sidecar :class:`FolderIO` at ``<self.path>/.ygg/``.

        ``None`` when this folder was constructed with
        ``yggmeta=False`` (the sidecar disables itself recursively
        so the ``.ygg/`` folder never builds its own ``.ygg/.ygg/``).
        Same concrete type as the parent so a custom
        :class:`FolderIO` subclass keeps its overrides inside the
        sidecar tree.

        :meth:`iter_children` skips ``.``-prefixed entries by
        default so the sidecar is invisible to data reads — calling
        ``self.yggmeta`` is the only way to surface it.
        """
        if not self._yggmeta_enabled:
            return None
        return type(self)(path=self.path / self.YGGMETA_DIRNAME, yggmeta=False)

    def _yggmeta_schema_holder(self) -> "Tabular | None":
        """Return the Arrow IPC leaf bound to the sidecar schema file, or ``None``."""
        ygg = self.yggmeta
        if ygg is None:
            return None
        schema_path = ygg.path / self.YGGMETA_SCHEMA_FILENAME
        return ygg._leaf_for(schema_path)

    def _persist_yggmeta_schema(self, schema: Any) -> None:
        """Write *schema* to ``.ygg/schema.arrow`` as a zero-row IPC file.

        Persists the whole :class:`yggdrasil.data.schema.StructField`
        (every per-field tag, every metadata entry) so the next
        :meth:`collect_schema` returns the typed schema without
        re-opening a data leaf — :class:`Field` already parses
        everything off the Arrow schema, so the round trip is
        lossless. Best-effort: a failed sidecar write logs and
        moves on; the data write is already on disk and the next
        read can still fall back to the first-batch shape.
        """
        if not self._yggmeta_enabled or schema is None:
            return
        arrow_schema = (
            schema.to_arrow_schema() if hasattr(schema, "to_arrow_schema")
            else schema
        )
        if arrow_schema is None:
            return
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            sidecar_dir = self.path / self.YGGMETA_DIRNAME
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sink = pa.BufferOutputStream()
            with pa.ipc.new_file(sink, arrow_schema):
                pass  # zero-row file, schema-only
            (sidecar_dir / self.YGGMETA_SCHEMA_FILENAME).write_bytes(
                sink.getvalue().to_pybytes(),
            )
        except Exception:
            # Sidecar is opportunistic — never fail the data write.
            pass

    def _load_yggmeta_schema(self) -> Any:
        """Read the sidecar schema back as a :class:`Schema`, or ``None``.

        Reverse of :meth:`_persist_yggmeta_schema`. Returns ``None``
        when yggmeta is disabled, the sidecar doesn't exist, or the
        file is unreadable — caller falls back to the standard
        first-batch :meth:`_collect_schema` path.
        """
        if not self._yggmeta_enabled:
            return None
        sidecar_dir = self.path / self.YGGMETA_DIRNAME
        schema_path = sidecar_dir / self.YGGMETA_SCHEMA_FILENAME
        try:
            if not schema_path.exists():
                return None
            payload = schema_path.read_bytes()
        except Exception:
            return None
        if not payload:
            return None
        try:
            reader = pa.ipc.open_file(pa.BufferReader(payload))
            arrow_schema = reader.schema
        except Exception:
            return None
        try:
            from yggdrasil.data.schema import Schema
            return Schema.from_arrow(arrow_schema)
        except Exception:
            return None

    def _collect_schema(self, options: FolderOptions) -> Any:
        """Sidecar-first schema collection.

        Returns the dumped ``.ygg/schema.arrow`` schema when present
        — that's the canonical record of what the folder writer
        last persisted, complete with every :class:`Field` tag.
        Falls through to :meth:`Tabular._collect_schema` (first
        batch's schema) when the sidecar is missing or unreadable.
        """
        sidecar = self._load_yggmeta_schema()
        if sidecar is not None:
            return sidecar
        return super()._collect_schema(options)

    # ==================================================================
    # Children — read
    # ==================================================================

    def iter_children(
        self, options: "FolderOptions | None" = None,
    ) -> "Iterator[Tabular]":
        """Yield every non-private direct entry of :attr:`path`.

        Sub-directories come back as a fresh :class:`FolderIO`. File
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
        if not self.path.exists():
            return

        partition_columns = self._resolve_partition_columns(options)
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

    def _resolve_partition_columns(
        self,
        options: "FolderOptions | None",
        *,
        batch_schema: Any = None,
    ) -> "tuple[str, ...]":
        """Resolve the partition layout from the most specific source available.

        Priority:

        1. ``options.partition_columns`` — explicit caller override.
        2. *batch_schema* — when the write path peeks the first
           incoming batch, the schema (via :meth:`Schema.from_arrow`)
           already carries every :class:`Field` tag, so we just walk
           ``schema.children`` for partition-tagged ones.
        3. :meth:`collect_schema` — for the read path, returns the
           sidecar schema persisted on the last write (preferred) or
           the first batch's schema (fallback).
        4. ``options.target`` — the schema-bearing slot the caller
           may pass explicitly.

        Columns already pinned by :attr:`static_values` (i.e. ones
        the enclosing partition layer already consumed when it
        minted this child) are filtered out — without that, a
        partition-key=v leaf would re-infer ``partition_key`` from
        its own data and recurse forever into
        ``partition_key=v/partition_key=v/…``.

        Returns an empty tuple when no source pins the layout —
        the caller's flat-folder branch then applies. :class:`Field`
        already parses every tag, so the helper just hands off to it.
        """
        if options is None:
            return ()
        consumed = set(self.static_values)

        def _filter(cols: "tuple[str, ...]") -> "tuple[str, ...]":
            return tuple(c for c in cols if c not in consumed)

        if options.partition_columns:
            return _filter(tuple(options.partition_columns))
        cols = _filter(_partition_columns_from_schema(batch_schema))
        if cols:
            return cols
        try:
            schema = self.collect_schema(options)
        except Exception:
            schema = None
        cols = _filter(_partition_columns_from_schema(schema))
        if cols:
            return cols
        return _filter(_partition_columns_from_schema(
            getattr(options, "target", None),
        ))

    def _iter_partition_children(
        self,
        options: "FolderOptions",
        partition_columns: "tuple[str, ...]",
    ) -> "Iterator[Tabular]":
        """Hive-aware listing variant — see :meth:`iter_children`.

        ``partition_columns[0]`` is the column this level resolves;
        descendants see the tail. Leaf files at this level (plain
        ``part-*.<ext>``, no ``<col>=<val>`` prefix) still pass
        through so a partially-populated tree (e.g. a freshly minted
        folder where the writer happened to land a non-partitioned
        leaf alongside the partition dirs) reads cleanly.
        """
        head, *tail = partition_columns
        remaining = tuple(tail)
        accepted = self._accepted_partition_values(options, head)

        if accepted is not None:
            yield from self._iter_partition_candidates(
                head, accepted, remaining,
            )
            return

        for entry in self.path.iterdir():
            if entry.name.startswith("."):
                continue
            try:
                is_dir = entry.is_dir()
            except Exception:
                continue
            if is_dir:
                parsed = _hive_split_name(entry.name)
                if parsed is not None and parsed[0] == head:
                    value = _cast_partition_value(
                        parsed[1], self._partition_dtype(head),
                    )
                    yield self._mint_partition_child(
                        entry, head, value, remaining,
                    )
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
        remaining: "tuple[str, ...]",
    ) -> "Iterator[Tabular]":
        """Probe ``<base>/<column>=<val>/`` directly for each accepted value.

        Skips :meth:`Path.iterdir` entirely — N stat probes against a
        deterministic path layout instead of a full directory walk.
        Existence check is delegated to the child's first read
        (``exists()`` would still cost a stat, and a missing
        directory yields zero children anyway), so this stays at one
        ``Path.from_`` per candidate value.
        """
        dtype = self._partition_dtype(column)
        for value in sorted(values, key=lambda v: (v is None, v)):
            encoded = _hive_encode(value)
            candidate = self.path / f"{column}={encoded}"
            typed_value = _cast_partition_value(value, dtype)
            yield self._mint_partition_child(
                candidate, column, typed_value, remaining,
            )

    def _mint_partition_child(
        self,
        path: "Path",
        column: str,
        value: Any,
        remaining: "tuple[str, ...]",
    ) -> "Tabular":
        """Mint a child FolderIO seeded with the parsed partition KV.

        ``remaining`` is the *outer* read's ``partition_columns``
        tail, kept as bookkeeping for the read pipeline — when the
        child's ``_read_arrow_batches`` recurses, it forwards a
        ``FolderOptions`` whose ``partition_columns`` is this tail
        so the next level can keep consuming.
        """
        child = type(self)(
            path=path,
            static_values={column: value},
        )
        # Stash the remaining tail on the child via an attribute the
        # read pipeline reads back out. Plain attribute (not a slot)
        # keeps the FolderOptions surface immutable / hashable while
        # giving the read recursion a cheap handoff.
        child._partition_remainder = remaining  # type: ignore[attr-defined]
        return self.adopt_child(child)

    def _accepted_partition_values(
        self,
        options: "FolderOptions",
        column: str,
    ) -> "frozenset[Any] | None":
        """Return the finite accepted-value set for *column*, or ``None``.

        Looks first at ``options.prune_values`` (caller-supplied
        explicit set), then falls back to walking
        ``options.predicate`` via :func:`extract_partition_filters`.
        Returns ``None`` when neither source pins the column to a
        finite set — the caller falls back to a full ``iterdir`` +
        per-child prune.
        """
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

    def _partition_dtype(self, column: str) -> "pa.DataType | None":
        """Best-effort partition dtype lookup for *column*.

        Reads :attr:`Tabular.schema` when the subclass exposes one;
        falls back to ``None`` (meaning "leave the value as a
        decoded string"). The static-value prune is conservative on
        undecidable predicates so a missing schema doesn't break
        correctness — it just turns a partition-level skip into a
        row-level filter.
        """
        try:
            schema = getattr(self, "schema", None)
        except Exception:
            return None
        if schema is None:
            return None
        try:
            field = schema[column]
        except Exception:
            return None
        if field is None:
            return None
        try:
            return field.arrow_field.type
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
        self.path.mkdir(parents=True, exist_ok=True)

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

        When ``options.partition_columns`` is set, the listing is
        Hive-aware (see :meth:`iter_children`). Each yielded child
        carries the leftover partition-columns tail on
        :attr:`_partition_remainder`; the recursive read forwards
        ``predicate`` / ``prune_values`` plus that shrunken tail so
        the next level can keep consuming partition columns until
        the leaves are reached.
        """
        if self._should_prune_by_predicate(options):
            return
        predicate = options.predicate
        for child in self.iter_children(options=options):
            if child._should_prune_by_predicate(options):
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
            if predicate is not None and not isinstance(child, FolderIO):
                yield from predicate.filter_arrow_batches(stream)
            else:
                yield from stream

    def _child_read_options(
        self, child: "Tabular", options: FolderOptions,
    ) -> Any:
        """Forward predicate / prune / partition-tail to *child*'s read.

        Predicates and explicit prune-value sets need to ride down
        the tree — otherwise a partition-aware caller would have to
        re-seed them at every level, and a row-level predicate on a
        nested folder would silently no-op. Hive children carry the
        consumed-tail on :attr:`_partition_remainder` so the next
        level only sees columns it hasn't already pinned via
        :attr:`Tabular.static_values`. Non-FolderIO children fall
        back to their own default options — leaf formats
        (parquet / arrow IPC / csv) accept ``predicate`` /
        ``prune_values`` via :class:`CastOptions`, the shared base.
        """
        opts_cls = child.options_class()
        kwargs: dict[str, Any] = {
            "predicate": options.predicate,
            "prune_values": options.prune_values,
        }
        if isinstance(child, FolderIO):
            kwargs["partition_columns"] = getattr(
                child, "_partition_remainder", (),
            )
        try:
            return opts_cls(**{k: v for k, v in kwargs.items() if v is not None or k == "partition_columns"})
        except TypeError:
            # The leaf options class doesn't know about one of our
            # kwargs (e.g. an older subclass without ``prune_values``).
            # Drop the unknown bits and retry; the per-child predicate
            # still wins via row-level filtering after read.
            safe = {
                k: v for k, v in kwargs.items()
                if v is not None
                and k in getattr(opts_cls, "__dataclass_fields__", ())
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
        from yggdrasil.data.schema import Schema

        # Peek the first batch — even with ``options.partition_columns``
        # pinned we still capture the schema to persist into the
        # ``.ygg/`` sidecar so the next read picks up the typed
        # layout. ``_chain_first`` puts the peeked batch back in front
        # of the iterator so the downstream branches see the same
        # stream the caller handed in.
        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            return
        try:
            first_schema = Schema.from_arrow(first.schema)
        except Exception:
            first_schema = None
        batches = _chain_first(first, batch_iter)

        partition_columns = self._resolve_partition_columns(
            options, batch_schema=first_schema,
        )
        if partition_columns:
            options = dataclasses.replace(
                options, partition_columns=partition_columns,
            )
            self._write_partitioned_batches(batches, options)
            self._persist_yggmeta_schema(first_schema)
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
            self._write_parts(batches, options)
            self._persist_yggmeta_schema(first_schema)
            return

        match_by = list(options.match_by_keys or ())
        is_upsert = options.mode in (Mode.UPSERT, Mode.MERGE)

        if match_by and self._has_tabular_children():
            if is_upsert:
                self._merge_upsert(batches, match_by, options)
            else:
                self._merge_append(batches, match_by, options)
            self._persist_yggmeta_schema(first_schema)
            return

        # Plain APPEND (or empty folder): mint a fresh part and drain.
        self._write_parts(batches, options)
        self._persist_yggmeta_schema(first_schema)

    def _write_partitioned_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: FolderOptions,
    ) -> None:
        """Split incoming *batches* by partition values and recurse per partition.

        For each incoming batch, splits rows by the leading
        ``options.partition_columns`` value (so a single batch with
        rows spread across N partitions becomes N batches, one per
        partition value), and feeds each subset to a fresh
        :class:`FolderIO` rooted at
        ``<self.path>/<col>=<val>/``. The recursive write inherits
        ``options`` with ``partition_columns`` shrunk by one so the
        nesting unwinds naturally — when the tail is empty the
        recursion lands in the flat-write branch above and mints a
        ``part-*.<ext>`` leaf at the partition folder.

        The partition column stays in the written payload (Hive
        convention is to drop it, but the cache layout deliberately
        keeps it — the row-level predicate at the read side runs
        against the same in-payload column so the read doesn't have
        to re-stamp from the directory name).
        """
        import pyarrow.compute as pc

        head, *tail = options.partition_columns
        tail_tuple: "tuple[str, ...]" = tuple(tail)
        # Build the child write options once — same shape minus the
        # consumed partition column, same mode / match_by / media
        # type so the per-partition append behaves identically to the
        # flat case.
        child_options = dataclasses.replace(
            options, partition_columns=tail_tuple,
        )

        # Group per-partition-value batches without materialising the
        # full stream — a single pass over each incoming batch is
        # enough to bucket its rows by the partition column value.
        for batch in batches:
            if batch.num_rows == 0:
                continue
            if head not in batch.schema.names:
                raise ValueError(
                    f"FolderIO partition write: batch is missing the "
                    f"partition column {head!r}. Schema has "
                    f"{batch.schema.names!r}; partition_columns="
                    f"{options.partition_columns!r}."
                )
            column = batch.column(head)
            distinct = pc.unique(column)
            for scalar in distinct:
                value = scalar.as_py()
                mask = pc.equal(column, scalar) if value is not None else pc.is_null(column)
                subset = batch.filter(mask)
                if subset.num_rows == 0:
                    continue
                encoded = _hive_encode(value)
                child_path = self.path / f"{head}={encoded}"
                child = type(self)(
                    path=child_path,
                    static_values={head: value},
                    # Per-partition child stays sidecar-less: the
                    # top-level folder owns the canonical
                    # ``<root>/.ygg/schema.arrow`` record. Recursing
                    # into partition children with ``yggmeta=True``
                    # would scatter one ``.ygg/`` per partition
                    # directory for zero added value.
                    yggmeta=False,
                )
                self.adopt_child(child)
                child._write_arrow_batches(iter((subset,)), child_options)

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
        if not self.path.exists():
            return []
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
            if isinstance(child, FolderIO):
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
            FolderIO._extend_keys_from_batch(keys, batch, match_by)
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
            if isinstance(child, FolderIO):
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
        if not self.path.exists():
            return
        for entry in self.path.iterdir():
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    continue
            except Exception:
                continue
            try:
                entry.unlink(missing_ok=True)
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
        if not self.path.exists():
            return 0
        not_pred = ~predicate
        deleted = 0
        for child in self.iter_children():
            if isinstance(child, FolderIO):
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
        if not self.path.exists():
            return 0
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
        leaf_folder = FolderIO(path=directory)
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
