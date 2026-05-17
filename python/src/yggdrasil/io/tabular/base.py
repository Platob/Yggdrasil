"""Pure interface for tabular I/O over Apache Arrow record batches.

:class:`Tabular[O]` declares "I yield and consume Arrow record
batches" — nothing else. No state, no lifecycle, no holder, no
cursor. Two abstract hooks plus an options-class declaration; the
engine fan-out (Arrow / Polars / Pandas / Spark / Python-native) is
derived from those hooks.

Implementers compose :class:`Tabular` with whatever concrete
substrate they need:

- :class:`yggdrasil.io.buffer.bytes_io.BytesIO` mixes Tabular with
  :class:`Disposable` and a :class:`Holder` cursor — the default
  byte-backed implementation.
- A hypothetical :class:`SparkCatalogTabular` could mix Tabular with
  a Spark-session reference and skip byte-level concerns entirely.

The contract
------------

A leaf implements three things:

- :meth:`options_class`        — :class:`CastOptions` subtype.
- :meth:`_read_arrow_batches`  — yield :class:`pa.RecordBatch`.
- :meth:`_write_arrow_batches` — consume an iterable of batches.

Public methods route caller kwargs through :meth:`check_options` to
a single resolved options instance, then dispatch to one of the
hooks.

Why not inherit from Holder
---------------------------

Tabular is an interface, not a substrate. Most things that satisfy
"yield Arrow batches" don't have a meaningful byte representation
(a Spark catalog table, a JDBC cursor, a remote dataset service).
Forcing Holder's byte primitives into the contract would either
require those backends to fake a buffer or split them off into a
parallel hierarchy. Keeping Tabular pure lets every backend
satisfy it without lying.

Format registry
---------------

Concrete byte-backed leaves (ParquetIO, CsvIO, ArrowIPCIO, …)
declare :attr:`Tabular.mime_type` at the class level and the
:meth:`__init_subclass__` hook auto-registers them in
:data:`_TABULAR_REGISTRY`. :meth:`Tabular.for_holder` resolves a
holder's :class:`MediaType` to the right concrete leaf and
constructs ``Cls(holder=holder)``. Mirror of the ``scheme``-based
dispatch on :class:`Holder`.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterator, TypeVar

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MediaType, MimeType
from yggdrasil.lazy_imports import polars_module, pyarrow_dataset_module


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas
    import polars as pl
    import pyarrow.dataset as pds
    from pyspark.sql import DataFrame as SparkDataFrame
    from yggdrasil.io.holder import Holder


__all__ = ["Tabular", "TabularStaticValues", "is_tabular_source"]


def is_tabular_source(obj: Any) -> bool:
    """True iff *obj* is a shape :meth:`Tabular.from_` can coerce.

    Catches the three shapes the cast layer needs to recognise before
    dispatching to engine-native constructors:

    * an existing :class:`Tabular` (incl. :class:`Holder` / :class:`Path`
      / :class:`Memory` / :class:`IO` — anything with the read/write
      Arrow batch contract);
    * an :class:`os.PathLike` (incl. :class:`pathlib.PurePath`);
    * a path-shaped ``str`` (``"data.parquet"``, ``"s3://b/k"``,
      ``"./x.csv"``) — string heuristic deliberately tight so that a
      caller passing a plain content string (``"hello"``) still falls
      through to the existing engine-native dispatch.

    File-like objects (have ``read``) aren't probed here: dispatching
    them needs an explicit ``media_type=`` and the cast layer doesn't
    carry one — :meth:`Tabular.from_` raises :class:`TypeError` for
    those if the caller routes them through anyway.
    """
    if isinstance(obj, Tabular):
        return True
    if isinstance(obj, os.PathLike):
        return True
    if isinstance(obj, str):
        if not obj or len(obj) > 4096:
            return False
        if "://" in obj or "/" in obj or "\\" in obj:
            return True
        # Bare filename with extension — ``data.parquet``, ``q.csv``.
        return "." in obj and not obj.startswith(".")
    return False


O = TypeVar("O", bound=CastOptions)
_ChildT = TypeVar("_ChildT", bound="Tabular")


class TabularStaticValues(Mapping):
    """Lazy mapping of column → constant value the owning :class:`Tabular`
    knows holds across every row it yields.

    Per-key lookup so a query against one column doesn't pay for
    every column's discovery, cached on hit. Inherits from
    :attr:`Tabular.tabular_parent` automatically — a Hive partition
    leaf inside ``region=us/`` reports ``{"region": "us"}`` even
    when the leaf class doesn't know about partitioning, because
    :meth:`__getitem__` walks up the parent chain after the owner's
    own resolver returns ``...``.

    Subclasses extend :meth:`Tabular._resolve_static_value` /
    :meth:`Tabular._known_static_keys` to plug in their source
    (Hive partition KV from the directory path, Parquet column
    stats with min == max, Delta file-level partition values, …);
    every Tabular reuses this Mapping shape so callers don't branch
    on the concrete class to read constants.
    """

    __slots__ = ("_owner", "_cache")

    def __init__(self, owner: "Tabular") -> None:
        self._owner = owner
        self._cache: "dict[str, Any]" = {}

    def __getitem__(self, key: str) -> Any:
        cache = self._cache
        if key in cache:
            return cache[key]

        # Owner first — leaf-level invariants override anything an
        # outer scope might assert. Parent fallback only on the
        # owner's "I don't know" answer.
        value = self._owner._resolve_static_value(key)
        if value is ...:
            parent = self._owner.tabular_parent
            if parent is not None:
                try:
                    value = parent.static_values[key]
                except KeyError:
                    pass

        if value is ...:
            raise KeyError(key)
        cache[key] = value
        return value

    def __iter__(self) -> Iterator[str]:
        keys: "set[str]" = set(self._owner._known_static_keys())
        parent = self._owner.tabular_parent
        if parent is not None:
            keys.update(parent.static_values.keys())
        return iter(keys)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, key: Any) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __repr__(self) -> str:
        return f"TabularStaticValues({dict(self)!r})"

_TABULAR_REGISTRY: "dict[str, type[Tabular]]" = {}
_TABULAR_REGISTRY_BOOTSTRAPPED: bool = False


def _bootstrap_tabular_registry() -> None:
    """Force-load every concrete :class:`Tabular` leaf package once.

    Each leaf module registers its ``mime_type`` via
    :meth:`Tabular.__init_subclass__` on import, so importing the
    leaf packages is enough to populate :data:`_TABULAR_REGISTRY`.
    Idempotent — the module-level flag short-circuits repeat calls.
    """
    global _TABULAR_REGISTRY_BOOTSTRAPPED
    if _TABULAR_REGISTRY_BOOTSTRAPPED:
        return
    _TABULAR_REGISTRY_BOOTSTRAPPED = True
    import yggdrasil.io.primitive  # noqa: F401
    import yggdrasil.io.nested  # noqa: F401


class Tabular(ABC, Generic[O]):
    """Pure interface — Arrow record-batch source/sink + engine fan-out.

    No state, no lifecycle, with the single exception of a
    :attr:`parent` back-pointer for tree-shaped sources (folders,
    archives, partitioned tables) where a child Tabular wants to
    walk back up to whatever yielded it. :meth:`adopt_child` is the
    parent-stamp helper aggregators call when handing back a child.

    Concrete implementers add whatever substrate they need (a
    holder + cursor for byte-backed shapes, a session reference
    for catalog-backed shapes, etc.) and override the two batch
    hooks. Format-specific leaves (ParquetIO, CsvIO, ArrowIPCIO, …)
    additionally declare :attr:`mime_type` to register against the
    process-wide :data:`_TABULAR_REGISTRY` — :meth:`for_holder` uses
    that registry to dispatch from a holder's :class:`MediaType` to
    the right concrete leaf at runtime.
    """

    #: Format identity for the registry. Subclasses set to a concrete
    #: :class:`MimeType` (``MimeTypes.PARQUET``, ``MimeTypes.CSV``, …)
    #: to claim that mime in :data:`_TABULAR_REGISTRY`. ``None`` (the
    #: abstract default) opts out of registration — :class:`Tabular`
    #: itself and intermediate abstracts (:class:`BytesIO`,
    #: :class:`NestedIO`) leave it unset so they don't shadow the
    #: real format leaves. Mirrors :attr:`Holder.scheme`.
    mime_type: "ClassVar[MimeType | None]" = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses keyed on :attr:`mime_type`."""
        super().__init_subclass__(**kwargs)
        mt = cls.mime_type
        if mt is None:
            return
        key = mt.name
        existing = _TABULAR_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise RuntimeError(
                f"Duplicate Tabular mime_type {mt.value!r}: "
                f"{cls.__name__} clashes with {existing.__name__}. "
                "If the override is intentional, clear the slot first "
                "via _TABULAR_REGISTRY.pop(...) at module-load time."
            )
        _TABULAR_REGISTRY[key] = cls

    def __init__(
        self,
        *,
        tabular_parent: "Tabular | None" = None,
        static_values: "Mapping[str, Any] | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the tree-parent slot.

        Subclasses chain ``super().__init__(**kwargs)`` and don't
        otherwise touch :attr:`tabular_parent` — aggregators set it
        via :meth:`adopt_child` when they yield / mint the child.

        The slot is named ``tabular_parent`` (not ``parent``) so it
        doesn't collide with :attr:`yggdrasil.io.path.path.Path.parent`,
        which is the path-shaped parent directory and therefore
        property-backed and read-only.

        ``static_values`` seeds :attr:`static_values` with constants
        the constructor already knows hold across every row this
        Tabular yields — typically the partition KV when an
        aggregator (e.g. :class:`YGGFolderIO`) mints a per-partition
        leaf. The full lookup is :meth:`static_values`'s lazy
        Mapping; callers shouldn't read this slot directly.
        """
        super().__init__()
        self.tabular_parent: "Tabular | None" = tabular_parent
        self._schema_cache: "Schema | Any" = ...
        self._static_value_seed: "dict[str, Any]" = (
            dict(static_values) if static_values else {}
        )
        # Lazy :class:`TabularStaticValues` view — built on first
        # access so a Tabular that nobody asks for constants on
        # never allocates the Mapping wrapper.
        self._static_values_view: "TabularStaticValues | None" = None

    # ==================================================================
    # Parent / child linkage
    # ==================================================================

    def adopt_child(self, child: "_ChildT") -> "_ChildT":
        """Stamp ``child.tabular_parent = self`` and return *child*.

        Used by aggregator Tabulars (folders, archives, partitioned
        tables) inside their child-yielding hooks (``iter_children``,
        ``make_child``) so consumers can walk back up the tree
        without the aggregator scattering the same line of attribute
        assignment across every implementation::

            for entry in self.path.iterdir():
                yield self.adopt_child(LeafIO(holder=entry))

        Re-adopting an already-attached child is fine — the slot
        is just overwritten. No checks, no ceremony.
        """
        child.tabular_parent = self
        return child

    # ==================================================================
    # Static values — column → constant invariants across every row
    # ==================================================================

    @property
    def static_values(self) -> "TabularStaticValues":
        """Lazy Mapping of column → constant value across every row
        this Tabular yields.

        See :class:`TabularStaticValues` for the exact contract:
        per-key lookup with caching, parent-chain inheritance, and
        a per-Tabular resolver hook (:meth:`_resolve_static_value`
        + :meth:`_known_static_keys`) so each subclass plugs in its
        own metadata source without callers branching on the
        concrete class.
        """
        view = self._static_values_view
        if view is None:
            view = TabularStaticValues(self)
            self._static_values_view = view
        return view

    def _resolve_static_value(self, key: str) -> Any:
        """Hook: return the constant value of column *key* across
        every row this Tabular yields, or ``...`` (Ellipsis) when
        the value is unknown / not statically constant.

        Default reads from the constructor seed
        (``static_values=...``). Subclasses with cheap-to-read
        metadata (Hive partition KV from the directory path,
        Parquet column stats with min == max, Delta file-level
        partition values, …) override and may bypass the seed
        entirely.
        """
        return self._static_value_seed.get(key, ...)

    def _known_static_keys(self) -> "Iterable[str]":
        """Hook: column names this Tabular knows about statically.

        Default returns the constructor seed's keys. Subclasses with
        a cheap "list every constant column" path (a parsed
        ``col=val/...`` tail, a parquet footer's column-stat
        coverage, …) override for richer ``__iter__`` /
        ``len(static_values)`` semantics; per-key lookup still
        routes through :meth:`_resolve_static_value`.
        """
        return self._static_value_seed.keys()

    def matches_static(self, predicate: "Any") -> bool:
        """True iff *predicate* could match any row given
        :attr:`static_values`. Conservative on undecidables (column
        not in static values, predicate evaluator failure) so the
        caller still reads.

        Builds a one-row pyarrow Table from the predicate's free
        columns that we have static values for, then evaluates the
        predicate against it — same trick :meth:`YGGFolderIO._delete`
        already used inline for partition-only conjuncts, generalised
        so any aggregator (read prune, optimize prune, future
        warehouse-style file skip) reuses the one helper.
        """
        if predicate is None:
            return True
        try:
            from yggdrasil.io.tabular.execution.expr.nodes import free_columns
            free = free_columns(predicate)
        except Exception:
            return True
        if not free:
            return True
        sv = self.static_values
        relevant: "dict[str, Any]" = {}
        for name in free:
            try:
                relevant[name] = sv[name]
            except KeyError:
                # Predicate touches a column outside our static
                # surface — can't decide, must read.
                return True
        try:
            table = pa.Table.from_pydict({k: [v] for k, v in relevant.items()})
            return predicate.filter_arrow_table(table).num_rows == 1
        except Exception:
            return True

    def _should_prune_by_predicate(self, options: Any) -> bool:
        """Return ``True`` iff ``options.predicate`` is provably false
        against this Tabular's :attr:`static_values` — the caller may
        skip the read entirely.

        Thin wrapper around :meth:`matches_static`: pulls the predicate
        off *options* (any object that exposes a ``predicate``
        attribute; absence and ``None`` both mean "no prune") and
        inverts the sense so an aggregator's read loop reads as
        ``if self._should_prune_by_predicate(options): return``. The
        inherited :class:`TabularStaticValues` parent chain does the
        rest — a child folder under a Hive ``col=val/`` leaf inherits
        the partition KV without anyone re-stamping it, so the prune
        fires uniformly at every level the read recurses through.

        Conservative on undecidables: a predicate over a column we
        have no static value for returns ``False`` (no prune) and the
        caller still reads. Same contract as :meth:`matches_static`.
        """
        predicate = getattr(options, "predicate", None)
        if predicate is None:
            return False
        return not self.matches_static(predicate)

    # ==================================================================
    # Format registry — MediaType → Tabular subclass dispatch
    # ==================================================================

    @classmethod
    def class_for_media_type(
        cls,
        media_type: "MediaType | MimeType | str | Any",
        *,
        default: Any = ...,
    ) -> "type[Tabular]":
        """Resolve a :class:`MediaType` (or coercible) to its Tabular class.

        Looks up :attr:`MediaType.mime_type`'s name in
        :data:`_TABULAR_REGISTRY`. Codec is orthogonal — Parquet
        compressed with zstd or snappy still resolves to
        :class:`ParquetIO`; the codec layer is the holder's concern.

        Returns *default* on miss when supplied; otherwise raises
        :class:`KeyError` with a list of registered names.
        """
        mt = MediaType.from_(media_type, default=None)
        if mt is None:
            if default is ...:
                raise KeyError(
                    f"Cannot coerce {media_type!r} to a MediaType "
                    "for Tabular registry lookup."
                )
            return default

        hit = _TABULAR_REGISTRY.get(mt.mime_type.name)
        if hit is not None:
            return hit

        # Miss may just mean the leaf package hasn't been imported
        # yet — force the side-effect bootstrap once and retry. This
        # is what catches nested leaves (ZipIO / FolderIO / DeltaIO)
        # for callers that never touched ``yggdrasil.io.nested``.
        if not _TABULAR_REGISTRY_BOOTSTRAPPED:
            _bootstrap_tabular_registry()
            hit = _TABULAR_REGISTRY.get(mt.mime_type.name)
            if hit is not None:
                return hit

        if default is ...:
            raise KeyError(
                f"No Tabular registered for {mt.mime_type.value!r}. "
                f"Registered: {sorted(_TABULAR_REGISTRY)}."
            )
        return default

    @classmethod
    def for_holder(
        cls,
        holder: "Holder",
        *,
        media_type: "MediaType | MimeType | str | None" = None,
        default: Any = ...,
        **kwargs: Any,
    ) -> "Tabular":
        """Build the right :class:`Tabular` subclass for *holder*.

        Resolution order for the format discriminator:

        1. The explicit *media_type* kwarg, when supplied.
        2. ``holder.stat().media_type`` — set by the holder from its
           URL extension, magic-byte sniff, or content-type header.

        The resolved class is instantiated as ``Cls(holder=holder,
        **kwargs)`` — every registered Tabular leaf is expected to
        accept ``holder=`` (true for :class:`BytesIO` subclasses,
        which is what the registry actually contains).

        On lookup miss, falls back to *default* when supplied. With
        no default, raises :class:`KeyError`.
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
    def from_(
        cls,
        obj: Any,
        *,
        media_type: "MediaType | MimeType | str | None" = None,
        default: Any = ...,
        **kwargs: Any,
    ) -> "Tabular":
        """Coerce *obj* into a :class:`Tabular` leaf for read/write.

        Routes:

        * :class:`Tabular` (incl. :class:`Holder` / :class:`Path` /
          :class:`Memory` / :class:`IO`) — returned as-is. When
          *media_type* is supplied and *obj* is a :class:`Holder`,
          :meth:`for_holder` is invoked so the caller can override the
          format leaf the holder's stamped MediaType would otherwise
          dispatch to.
        * ``str`` / :class:`os.PathLike` — coerced via
          :meth:`Path.from_`, which scheme-dispatches to the right
          concrete subclass (:class:`LocalPath`, :class:`S3Path`,
          :class:`DatabricksPath`, …). Strings are accepted only when
          path-shaped (see :func:`is_tabular_source`); plain content
          strings raise.
        * File-like objects (anything with a callable ``read``
          returning ``bytes``) — drained into a :class:`Memory`
          holder; *media_type* is required since a raw byte stream
          has no URL extension to sniff.

        Falls back to *default* on unrecognised shapes when supplied;
        otherwise raises :class:`TypeError` with the offending type.
        """
        from yggdrasil.io.holder import Holder

        if isinstance(obj, Tabular):
            if media_type is not None and isinstance(obj, Holder):
                return cls.for_holder(obj, media_type=media_type, **kwargs)
            return obj

        if isinstance(obj, (str, os.PathLike)):
            if isinstance(obj, str) and not is_tabular_source(obj):
                if default is not ...:
                    return default
                raise TypeError(
                    f"Tabular.from_ string {obj!r} is not path-shaped; "
                    "pass a URL, a filesystem path, or a name with an "
                    "extension (e.g. 'data.parquet')."
                )
            from yggdrasil.io.path import Path as YggPath
            path = YggPath.from_(obj)
            if media_type is not None:
                return cls.for_holder(path, media_type=media_type, **kwargs)
            return path

        read = getattr(obj, "read", None)
        if callable(read):
            if media_type is None:
                if default is not ...:
                    return default
                raise TypeError(
                    f"Tabular.from_ requires media_type= for file-like "
                    f"inputs ({type(obj).__name__}); no URL extension "
                    "to sniff the format."
                )
            from yggdrasil.io.memory import Memory
            data = read()
            if isinstance(data, str):
                data = data.encode()
            return cls.for_holder(
                Memory(binary=data), media_type=media_type, **kwargs,
            )

        if default is not ...:
            return default
        raise TypeError(
            f"Cannot coerce {type(obj).__name__} to a Tabular. "
            "Pass a Tabular, a path / URL (str or os.PathLike), or "
            "a file-like object with media_type=."
        )

    @classmethod
    def registered_classes(cls) -> "dict[str, type[Tabular]]":
        """Snapshot of the registry — debugging / introspection only."""
        return dict(_TABULAR_REGISTRY)

    # ==================================================================
    # Options
    # ==================================================================

    @classmethod
    def options_class(cls) -> "type[O]":
        """The :class:`CastOptions` subclass this implementer consumes.

        Default :class:`CastOptions`. Format-specific leaves with
        their own knobs (Parquet compression, CSV delimiter, …)
        override.
        """
        return CastOptions  # type: ignore[return-value]

    @classmethod
    def check_options(
        cls,
        options: "O | None" = None,
        overrides: "dict | None" = None,
        **kwargs: Any,
    ) -> O:
        """Validate and merge caller kwargs into a resolved options.

        Canonical pattern: a public method passes ``overrides=locals()``
        and the ``...``-defaulted entries are stripped, the rest merged.
        """
        if overrides:
            overrides = {k: v for k, v in overrides.items() if v is not ...}
            overrides.pop("self", None)
            options = overrides.pop("options", options)
            kwargs.update(overrides)
            kwargs.update(kwargs.pop("kwargs", {}))
        return cls.options_class().check(options, **kwargs)

    # ==================================================================
    # Cleanup hook — backend-specific GC of stale state
    # ==================================================================

    def cleanup(self, wait: "Any" = False) -> int:
        """Garbage-collect stale state on this backend.

        Default no-op (returns ``0``) — single-file leaves and
        warehouse-backed tables don't have a sweep concept the
        client owns. :class:`yggdrasil.io.nested.ygg_folder_io.YGGFolderIO`
        overrides this to unlink stale ``part-*`` files, throttled by
        TTL.

        ``wait`` controls sync vs async dispatch on backends that
        support it: a truthy :class:`yggdrasil.dataclasses.waiting.WaitingConfig`
        (or ``True`` / a positive timeout) blocks until the sweep
        finishes; a falsy value (the default) hands the work off to a
        background thread. Backends without an async path treat both
        the same.

        Returns the number of files / rows removed when known; ``0``
        for fire-and-forget async dispatch or a no-op backend.
        """
        return 0

    # ==================================================================
    # Compaction hook
    # ==================================================================

    def optimize(
        self,
        byte_size: "int | None" = None,
        **kwargs: Any,
    ) -> int:
        """Repartition / compact this Tabular's storage.

        Default implementation is a no-op and returns ``0`` — single-file
        leaves (parquet, csv, arrow IPC, …) don't have a compaction
        concept. Aggregator subclasses (:class:`FolderIO`,
        :class:`YGGFolderIO`) override this to walk their child leaves
        and bin-pack small part files into bundles near *byte_size*.
        Files already close to the target size are left alone so a
        repeated call is cheap.

        ``byte_size=None`` keeps the legacy "collapse every leaf with
        more than one part into a single file" behavior, which is what
        the local-cache compaction loop in :class:`Session` expects.
        Any extra keyword arguments are accepted and ignored so
        upstream callers can pass forward-compatible knobs without the
        base raising.
        """
        return 0

    # ==================================================================
    # Row-level delete
    # ==================================================================

    def delete(
        self,
        predicate: "Any",
        *,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> int:
        """Delete every row matching *predicate*. Return rows removed.

        *predicate* is a :class:`Predicate` from
        :mod:`yggdrasil.io.tabular.execution.expr` or a SQL string
        that parses into one (``"id IN (1,2,3)"``,
        ``"price > 100 AND region = 'EU'"``).

        The default implementation reads every batch, drops rows the
        predicate accepts, and rewrites the leaf with the survivors.
        Aggregator subclasses (:class:`yggdrasil.io.nested.folder_io.FolderIO`,
        :class:`yggdrasil.io.nested.ygg_folder_io.YGGFolderIO`)
        override to walk children, prune subtrees whose partition
        bounds make the predicate trivially false, and only rewrite
        the leaves that actually hold matched rows — so a delete on a
        hive-partitioned tree never scans partitions it can prove
        don't match.
        """
        from yggdrasil.io.tabular.execution.expr import Expression, Predicate

        if isinstance(predicate, str):
            predicate = Expression.from_sql(predicate)
        if not isinstance(predicate, Predicate):
            raise TypeError(
                f"{type(self).__name__}.delete expected a Predicate "
                f"(or a SQL string parseable to one); got "
                f"{type(predicate).__name__}: {predicate!r}."
            )
        return self._delete(
            predicate, self.check_options(options, overrides=locals()),
        )

    def _delete(self, predicate: "Any", options: O) -> int:
        """Generic single-leaf delete: filter all batches, rewrite.

        Routes the row-level work through
        :meth:`Predicate.filter_arrow_batches`, so the streaming
        filter runs in pyarrow's C++ kernels — no Python row
        iteration. Counts rows by diffing input vs. surviving sizes
        as the stream goes by, keeping the streaming property: only
        one batch resides in memory at a time.
        """
        survivors: "list[pa.RecordBatch]" = []
        kept_rows = 0
        total_rows = 0
        not_pred = ~predicate

        def _counted() -> "Iterator[pa.RecordBatch]":
            nonlocal total_rows
            for b in self._read_arrow_batches(options):
                total_rows += b.num_rows
                yield b

        for kept in not_pred.filter_arrow_batches(_counted()):
            kept_rows += kept.num_rows
            survivors.append(kept)
        deleted = total_rows - kept_rows
        if deleted == 0:
            return 0
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s deleted %d / %d rows",
                type(self).__name__,
                deleted,
                total_rows,
            )
        self._write_arrow_batches(iter(survivors), options)
        return deleted

    # ==================================================================
    # Execution-plan entry point
    # ==================================================================

    def execute_plan(
        self,
        plan: "Any",
        *,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "Tabular":
        """Apply an :class:`ExecutionPlan` to this Tabular.

        The default returns a :class:`LazyTabular` wrapping ``self``
        with the plan attached — execution stays lazy and routes
        through whatever read hook the caller pulls (Arrow, polars
        LazyFrame, …). An empty plan returns ``self`` unchanged so
        callers can hand a possibly-empty plan in without an explicit
        guard.

        Subclasses with native plan execution (a SQL engine that can
        push the whole plan to a remote, an in-engine LazyFrame
        source, …) override to bypass the LazyTabular wrapper. The
        contract is just: return a :class:`Tabular` whose reads
        produce the same rows the wrapper would.
        """
        from yggdrasil.io.tabular.execution.plan import ExecutionPlan
        from yggdrasil.io.tabular.lazy import LazyTabular

        coerced = (
            plan if isinstance(plan, ExecutionPlan)
            else ExecutionPlan(tuple(plan)) if plan is not None
            else ExecutionPlan.empty()
        )
        if coerced.is_empty():
            return self
        # ``options`` / ``**kwargs`` are accepted for forward-compat
        # so subclasses can wire engine-specific knobs through;
        # the default LazyTabular doesn't need them — its reads
        # carry their own options.
        del options, kwargs
        return LazyTabular(self, plan=coerced)

    def lazy(self) -> "Tabular":
        """Return a :class:`LazyTabular` view with a ``SELECT *`` plan.

        Entry point for the builder API — chain :meth:`select`,
        :meth:`filter`, :meth:`group_by`, :meth:`apply` off the result
        without each call paying the wrapping cost. The seeded plan is
        a single :class:`Select` over ``"*"`` so the lazy frame still
        round-trips every column when collected with no further ops.
        """
        from yggdrasil.io.tabular.execution.plan import ExecutionPlan, Select
        from yggdrasil.io.tabular.lazy import LazyTabular

        return LazyTabular(self, plan=ExecutionPlan((Select(("*",)),)))

    # ==================================================================
    # Abstract batch hooks — the two things every implementer overrides
    # ==================================================================

    @abstractmethod
    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from the underlying source."""

    @abstractmethod
    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Consume Arrow record batches and persist them."""

    # ==================================================================
    # Schema inspection — per-instance cache
    # ==================================================================

    def collect_schema(self, options: "O | None" = None, **kwargs: Any) -> Schema:
        """Return this Tabular's :class:`Schema`, caching the first hit.
        """
        if self._schema_cache is not ...:
            return self._schema_cache

        options = self.check_options(options, overrides=locals())

        if options.target:
            return options.target

        schema = self._collect_schema(options)
        return schema

    def _collect_schema(self, options: O) -> Schema:
        """First batch's schema by default; merged across batches when
        ``options.safe`` is set, so heterogeneous sources surface a
        union schema instead of "whichever batch came first"."""
        batches = self._read_arrow_batches(options)
        first = next(iter(batches), None)
        if first is None:
            return Schema.empty()
        schema = Schema.from_arrow(first.schema)
        if not getattr(options, "safe", False):
            return schema
        for batch in batches:
            schema = schema.merge_with(
                Schema.from_arrow(batch.schema), inplace=True,
            )
        return schema

    def _persist_schema(self, schema: "Schema | None") -> None:
        """Stamp *schema* into the per-instance cache (writer hook).

        Writers call this after the bytes hit the sink so the next
        :meth:`collect_schema` returns the freshly-written shape
        instead of replaying the source. ``None`` and an empty schema
        are accepted but ignored — they would shadow whatever the
        cache held without telling readers anything new.
        """
        if schema is None:
            return
        self._schema_cache = schema

    def _unpersist_schema(self) -> None:
        """Drop the per-instance schema cache.

        Called from :meth:`close` (so reopening re-collects from the
        source) and from any path that mutates the underlying bytes
        in a way the writers can't describe (truncate, clear).
        """
        self._schema_cache = ...

    # ==================================================================
    # Lifecycle hook — clears the schema cache on close
    # ==================================================================

    def close(self, force: bool = False) -> None:
        """Drop the schema cache and forward to any cooperative ``close``.

        Tabular itself has no resources to release — the schema cache
        is the only state it owns. Subclasses that mix Tabular with a
        lifecycle (``Disposable``-derived ``IO``, holders, …) inherit
        this hook through cooperative ``super().close()``; pure
        Tabular subclasses without a lifecycle peer get a harmless
        no-op forward.
        """
        self._unpersist_schema()
        sup_close = getattr(super(), "close", None)
        if callable(sup_close):
            sup_close(force=force)

    def _commit_metadata(self) -> None:
        """Sync written metadata (size, mtime, media_type) to the backing.

        Hook for backends with persistent IO metadata. Default no-op —
        subclasses with a holder (``IO``) override to refresh
        :class:`IOStats` once at the end of a bulk write rather than
        per batch. Driven by :attr:`CastOptions.sync_metadata`.
        """

    # ==================================================================
    # Static helpers
    # ==================================================================

    @staticmethod
    def _normalize_records(data: Iterable[dict]) -> "list[dict]":
        """Backfill every row to the union of keys seen across all rows.

        :meth:`pa.Table.from_pylist` infers schema from the first row
        and silently drops keys that show up only later; callers that
        build rows incrementally (e.g. JSON parsers) route through
        here first.
        """
        rows = list(data) if not isinstance(data, list) else data
        if not rows:
            return []

        all_keys: "dict[str, None]" = {}
        needs_backfill = False
        reference: "tuple[str, ...] | None" = None
        for row in rows:
            if row is None:
                needs_backfill = True
                continue
            keys = tuple(row.keys())
            if reference is None:
                reference = keys
            elif keys != reference:
                needs_backfill = True
            for k in keys:
                if k not in all_keys:
                    all_keys[k] = None

        if not needs_backfill:
            return rows
        key_tuple = tuple(all_keys)
        return [
            {k: (row.get(k) if row is not None else None) for k in key_tuple}
            for row in rows
        ]

    # ==================================================================
    # Arrow surface
    # ==================================================================

    def read_arrow_batches(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> Iterator[pa.RecordBatch]:
        resolved = self.check_options(options, overrides=locals())
        if not logger.isEnabledFor(logging.DEBUG):
            yield from self._read_arrow_batches(resolved)
            return
        n_batches = 0
        n_rows = 0
        for batch in self._read_arrow_batches(resolved):
            n_batches += 1
            n_rows += batch.num_rows
            yield batch
        logger.debug(
            "%s read %d batches / %d rows",
            type(self).__name__,
            n_batches,
            n_rows,
        )

    def read_arrow_table(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> pa.Table:
        return self._read_arrow_table(self.check_options(options, overrides=locals()))

    def _read_arrow_table(self, options: O) -> pa.Table:
        batches = list(self._read_arrow_batches(options))
        if not batches:
            schema = (
                getattr(options, "target_schema", None)
                or getattr(options, "source_schema", None)
                or Schema.empty()
            )
            return schema.to_arrow_schema().empty_table()
        return pa.Table.from_batches(batches)

    def read_arrow_batch_reader(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pa.RecordBatchReader":
        return self._read_arrow_batch_reader(
            self.check_options(options, overrides=locals())
        )

    def _read_arrow_batch_reader(self, options: O) -> "pa.RecordBatchReader":
        schema = options.check_target(obj=self.collect_schema).merged
        return pa.RecordBatchReader.from_batches(
            schema.to_arrow_schema(), self._read_arrow_batches(options),
        )

    def read_arrow_dataset(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pds.Dataset":
        return self._read_arrow_dataset(
            self.check_options(options, overrides=locals())
        )

    def _read_arrow_dataset(self, options: O) -> "pds.Dataset":
        pds = pyarrow_dataset_module()
        reader = self._read_arrow_batch_reader(options)
        return pds.dataset(reader, schema=reader.schema)

    def read_table(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> pa.Table:
        """Read into a pyarrow :class:`pa.Table`.

        Pyarrow is the only hard runtime dependency, so it's the
        portable default. Callers that want a different engine can
        ask for :meth:`read_polars_frame` / :meth:`read_spark_frame`
        explicitly.
        """
        return self._read_arrow_table(self.check_options(options, overrides=locals()))

    def write_table(
        self,
        obj: Any,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Dispatch *obj* to the best ``_write_*`` hook based on its runtime type.

        Recognizes another :class:`Tabular` (drained as a pyarrow
        record-batch stream), ``pa.Table`` / ``pa.RecordBatch`` /
        ``pa.RecordBatchReader``, polars ``DataFrame`` / ``LazyFrame``,
        pandas ``DataFrame``, pyspark ``DataFrame``, ``list[dict]``,
        ``dict[str, list]``, and iterables of any of the above.
        Module-name sniffing keeps optional engine deps out of the
        import graph — we only touch a frame's API once we've
        confirmed it's an instance of one we know how to drain.
        """
        self._write_table(obj, self.check_options(options, overrides=locals()))

    def _write_table(self, obj: Any, options: O) -> None:
        if obj is None:
            return
        if isinstance(obj, Tabular):
            self._write_arrow_batches(obj.read_arrow_batches(), options)
            return
        if isinstance(obj, pa.Table):
            self._write_arrow_table(obj, options)
            return
        if isinstance(obj, pa.RecordBatch):
            self._write_arrow_batches([obj], options)
            return
        if isinstance(obj, pa.RecordBatchReader):
            self._write_arrow_batches(obj, options)
            return

        from yggdrasil.pickle.serde import ObjectSerde
        ns, _ = ObjectSerde.module_and_name(obj)
        if ns.startswith("polars"):
            self._write_polars_frame(obj, options)
            return
        if ns.startswith("pandas"):
            self._write_pandas_frame(obj, options)
            return
        if ns.startswith("pyspark"):
            self._write_spark_frame(obj, options)
            return

        if isinstance(obj, Mapping):
            if obj and all(isinstance(v, (list, tuple)) for v in obj.values()):
                self._write_pydict({k: list(v) for k, v in obj.items()}, options)
                return
            self._write_pylist([dict(obj)], options)
            return

        if isinstance(obj, (str, bytes, bytearray, memoryview)):
            raise TypeError(
                f"{type(self).__name__}.write_table can't infer a writer for "
                f"{type(obj).__name__}: {obj!r}. Accepted: pyarrow "
                "Table/RecordBatch/RecordBatchReader, polars DataFrame/LazyFrame, "
                "pandas DataFrame, pyspark DataFrame, list[dict], dict[str, list], "
                "or an iterable of any of those."
            )

        try:
            iterator = iter(obj)
        except TypeError as exc:
            raise TypeError(
                f"{type(self).__name__}.write_table can't infer a writer for "
                f"{ObjectSerde.full_namespace(obj)}: {obj!r}. "
                "Accepted: pyarrow Table/RecordBatch/RecordBatchReader, "
                "polars DataFrame/LazyFrame, pandas DataFrame, pyspark "
                "DataFrame, list[dict], dict[str, list], or an iterable of "
                "any of those."
            ) from exc

        try:
            first = next(iterator)
        except StopIteration:
            return
        import itertools as _it
        rest = _it.chain([first], iterator)
        if isinstance(first, pa.RecordBatch):
            self._write_arrow_batches(rest, options)
            return
        if isinstance(first, pa.Table):
            for inner in rest:
                self._write_arrow_table(inner, options)
            return
        if isinstance(first, Mapping):
            self._write_pylist(rest, options)
            return
        from yggdrasil.data.record import Record
        if isinstance(first, Record):
            self._write_records(rest, options)
            return
        raise TypeError(
            f"{type(self).__name__}.write_table can't infer a writer for "
            f"iterable of {type(first).__name__}: first item {first!r}. "
            "Accepted element types: pyarrow Table/RecordBatch, Mapping, "
            "or yggdrasil Record."
        )

    def write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        options = self.check_options(options, overrides=locals())
        if not logger.isEnabledFor(logging.DEBUG):
            self._write_arrow_batches(batches, options)
        else:
            n_batches = 0
            n_rows = 0
            def _counted() -> "Iterator[pa.RecordBatch]":
                nonlocal n_batches, n_rows
                for batch in batches:
                    n_batches += 1
                    n_rows += batch.num_rows
                    yield batch
            self._write_arrow_batches(_counted(), options)
            logger.debug(
                "%s wrote %d batches / %d rows (mode=%s)",
                type(self).__name__,
                n_batches,
                n_rows,
                options.mode,
            )
        if options.sync_metadata:
            self._commit_metadata()

    def write_arrow_table(
        self, table: pa.Table, options: "O | None" = None, **kwargs: Any,
    ) -> None:
        self._write_arrow_table(
            table, self.check_options(options, overrides=locals()),
        )

    def _write_arrow_table(self, table: pa.Table, options: O) -> None:
        casted = options.cast_arrow_tabular(table)

        # Keep ``target`` set: downstream writers (delta partition
        # inference, schema-driven SQL DDL, …) read tags off the
        # target field. Bind ``source`` to the merged target so the
        # per-batch ``cast_arrow_tabular`` collapses to its bypass
        # instead of re-running the cast on data we've already
        # shaped.
        merged = options.merged
        inner = options.copy(
            source=merged if merged is not None else ...,
            sync_metadata=False,
            row_size=None,
            byte_size=None,
        )
        self._write_arrow_batches(casted.to_batches(), inner)

        if options.sync_metadata:
            self._commit_metadata()

    # ==================================================================
    # Polars
    # ==================================================================

    def read_polars_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pl.DataFrame":
        return self._read_polars_frame(
            self.check_options(options, overrides=locals())
        )

    def _read_polars_frame(self, options: O) -> "pl.DataFrame":
        return polars_module().from_arrow(self._read_arrow_table(options))  # type: ignore[return-value]

    def scan_polars_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pl.LazyFrame":
        return self._scan_polars_frame(
            self.check_options(options, overrides=locals())
        )

    def _scan_polars_frame(self, options: O) -> "pl.LazyFrame":
        return polars_module().scan_pyarrow_dataset(self._read_arrow_dataset(options))

    def write_polars_frame(
        self,
        frame: "pl.DataFrame | pl.LazyFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_polars_frame(
            frame, self.check_options(options, overrides=locals()),
        )

    def _write_polars_frame(
        self, frame: "pl.DataFrame | pl.LazyFrame", options: O,
    ) -> None:
        pl = polars_module()

        casted = options.cast_polars_tabular(frame)

        if isinstance(casted, pl.LazyFrame):
            casted = casted.collect()

        # Empty-frame guard reads off the post-collect ``casted`` —
        # the input may have been a LazyFrame (no ``.height``), and
        # the cast can also produce an empty frame from a non-empty
        # source when projecting columns the source doesn't carry.
        if casted.height == 0:
            return

        # ``CompatLevel.newest()`` emits ``string_view`` / ``binary_view`` —
        # great for in-memory pyarrow.compute, but the pyarrow parquet
        # writer (and several downstream tabular paths) still raise
        # ``Slicing not implemented for StringView`` when fed view
        # buffers. ``oldest()`` keeps the legacy flat ``string`` /
        # ``binary`` shape that every reader/writer handles.
        self._write_arrow_table(casted.to_arrow(compat_level=pl.CompatLevel.oldest()), options)

        if options.sync_metadata:
            self._commit_metadata()

    # ==================================================================
    # Pandas
    # ==================================================================

    def read_pandas_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pandas.DataFrame":
        return self._read_pandas_frame(
            self.check_options(options, overrides=locals())
        )

    def _read_pandas_frame(self, options: O) -> "pandas.DataFrame":
        return self._read_arrow_table(options).to_pandas()

    def write_pandas_frame(
        self,
        frame: "pandas.DataFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_pandas_frame(frame, self.check_options(options, overrides=locals()))

    def _write_pandas_frame(self, frame: "pandas.DataFrame", options: O) -> None:
        # ``frame.index.names`` is always a list (e.g. ``[None]`` for a
        # default ``RangeIndex``), so ``bool(...)`` is True even on
        # an anonymous index — we'd round-trip a synthetic column that
        # the reader then complains about. Only preserve the index
        # when at least one level has a user-assigned name.
        include_index = any(n is not None for n in frame.index.names)

        # Fast path for object-dtype columns when a target schema is
        # bound. Object columns are the ones pyarrow's pandas bridge
        # has to infer from Python objects (list-of-dict for struct,
        # list-of-list for nested arrays, mixed scalars for strings) —
        # inference walks every cell and dominates wall time on nested
        # payloads. Handing the target type to
        # ``pa.array(col, type=..., from_pandas=True)`` drives the
        # conversion straight to the wanted shape and also emits
        # ``string`` instead of ``large_string`` so the downstream
        # parquet writer can dictionary-encode it (the post-cast path
        # used to preserve ``large_string``, which silently disabled
        # dictionary encoding on every staged Parquet file Databricks
        # ever saw from a pandas caller).
        #
        # We can't push ``schema=`` into ``pa.Table.from_pandas``: the
        # pandas bridge treats ``schema`` as a column projection and
        # silently drops every frame column not in the schema. Per-
        # column conversion keeps the column list intact while still
        # hinting the slow ones; typed columns stay un-hinted so the
        # downstream cast can still widen / narrow across dtype
        # mismatches the way it always has (that's the path the old
        # "string column → numeric target" case relied on, and it
        # stays intact here).
        #
        # Falls back to plain ``from_pandas`` whenever a hinted
        # conversion raises (incompatible cell contents, non-nullable
        # target with NaN, …) or whenever the fast path doesn't apply
        # (no target, index round-trip, duplicate column names).
        fast_casted: "pa.Table | None" = None
        target = getattr(options, "target", None)
        if (
            target is not None
            and not include_index
            and frame.columns.is_unique
        ):
            try:
                arrow_schema = target.to_arrow_schema()
            except Exception:
                arrow_schema = None
            if arrow_schema is not None:
                target_by_name = {f.name: f for f in arrow_schema}
                try:
                    arrays: "list[pa.Array]" = []
                    names: "list[str]" = []
                    for name in frame.columns:
                        col = frame[name]
                        tgt = target_by_name.get(name)
                        if tgt is not None and col.dtype.kind == "O":
                            arrays.append(
                                pa.array(col, type=tgt.type, from_pandas=True),
                            )
                        else:
                            arrays.append(pa.array(col, from_pandas=True))
                        names.append(str(name))
                    fast_casted = pa.table(arrays, names=names)
                except (pa.ArrowException, TypeError, ValueError):
                    fast_casted = None

        if fast_casted is not None:
            self._write_arrow_table(fast_casted, options)
            return

        try:
            casted = pa.Table.from_pandas(frame, preserve_index=include_index)
            self._write_arrow_table(casted, options)
        except (pa.ArrowException, TypeError, ValueError):
            # pyarrow's pandas bridge raises ``pa.ArrowException``
            # (e.g. for unsupported extension dtypes) and stdlib
            # ``TypeError`` / ``ValueError`` (e.g. for unhashable
            # cell contents); fall back to polars, which handles a
            # broader set of object-typed pandas frames at the cost
            # of an extra conversion hop.
            pl = polars_module()
            casted = pl.from_pandas(frame, include_index=include_index)
            self._write_polars_frame(casted, options)

    # ==================================================================
    # Spark — driver-side materialize, no streaming spill
    # ==================================================================

    def read_spark_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "SparkDataFrame":
        return self._read_spark_frame(
            self.check_options(options, overrides=locals())
        )

    def _read_spark_frame(self, options: O) -> "SparkDataFrame":
        from yggdrasil.environ import PyEnv
        spark = PyEnv.spark_session(create=True)
        return options.cast_spark(
            spark.createDataFrame(self._read_arrow_table(options))
        )

    def write_spark_frame(
        self,
        frame: "SparkDataFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_spark_frame(
            frame, self.check_options(options, overrides=locals()),
        )

    def _write_spark_frame(self, frame: "SparkDataFrame", options: O) -> None:
        to_arrow = getattr(frame, "toArrow", None)
        if to_arrow is not None:
            self._write_arrow_table(to_arrow(), options)
            return
        self._write_pandas_frame(frame.toPandas(), options)

    # ==================================================================
    # Python-native
    # ==================================================================

    def read_pylist(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "list[dict]":
        return self._read_pylist(self.check_options(options, overrides=locals()))

    def _read_pylist(self, options: O) -> "list[dict]":
        return self._read_arrow_table(options).to_pylist()

    def write_pylist(
        self,
        data: Iterable[dict],
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_pylist(data, self.check_options(options, overrides=locals()))

    def _write_pylist(self, data: Iterable[dict], options: O) -> None:
        rows = self._normalize_records(data)
        if not rows:
            return
        self._write_arrow_table(pa.Table.from_pylist(rows), options)

    def read_pydict(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "dict[str, list]":
        return self._read_pydict(self.check_options(options, overrides=locals()))

    def _read_pydict(self, options: O) -> "dict[str, list]":
        return self._read_arrow_table(options).to_pydict()

    def write_pydict(
        self,
        data: "dict[str, list]",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_pydict(data, self.check_options(options, overrides=locals()))

    def _write_pydict(self, data: "dict[str, list]", options: O) -> None:
        self._write_arrow_table(pa.Table.from_pydict(data), options)

    def read_record_iterator(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "Iterator[Mapping[str, Any]]":
        """Stream rows as plain ``dict``. True streaming — the full
        table never materializes; ``batch.to_pylist()`` does the
        column→row rotation in pyarrow C++ once per batch."""
        return self._read_record_iterator(
            self.check_options(options, overrides=locals())
        )

    def _read_record_iterator(self, options: O) -> "Iterator[Mapping[str, Any]]":
        for batch in self._read_arrow_batches(options):
            yield from batch.to_pylist()

    def read_records(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "Iterator[Any]":
        """Stream rows as :class:`yggdrasil.data.record.Record`. Lower
        per-row allocation than :meth:`read_pylist` for stable-schema
        sources — the underlying :class:`Schema` is materialized once
        and shared by reference across every record."""
        return self._read_records(
            self.check_options(options, overrides=locals())
        )

    def _read_records(self, options: O) -> "Iterator[Any]":
        from yggdrasil.data.record import Record
        yield from Record.from_arrow_batches(self._read_arrow_batches(options))

    def write_records(
        self,
        records: "Iterable[Any]",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_records(
            records, self.check_options(options, overrides=locals()),
        )

    def _write_records(self, records: "Iterable[Any]", options: O) -> None:
        """Bucket records by ``options.row_size`` and delegate to
        :meth:`_write_arrow_batches`. The first record's schema becomes
        the writer's target — heterogeneous-schema streams are silently
        re-aligned via dict-order iteration, so callers that need
        strictness should pre-validate. Subclasses with a row-native
        sink (SQL bulk-insert, Spark createDataFrame) should override
        to skip the row → Arrow round-trip.
        """
        from yggdrasil.data.record import Record

        chunk_size = max(1, getattr(options, "row_size", None) or 1024)
        chunk_rows: "list[dict]" = []
        chunk_schema: "pa.Schema | None" = None
        # Per-batch sub-call defers metadata commits; we commit once
        # below if the caller asked for it.
        inner = options.copy(sync_metadata=False) if options.sync_metadata else options

        def _flush() -> None:
            if not chunk_rows:
                return
            batch = pa.RecordBatch.from_pylist(chunk_rows, schema=chunk_schema)
            self._write_arrow_batches([batch], inner)
            chunk_rows.clear()

        for rec in records:
            if isinstance(rec, Record):
                if chunk_schema is None:
                    chunk_schema = rec.schema.to_arrow_schema()
                chunk_rows.append(rec.to_dict())
            elif isinstance(rec, Mapping):
                chunk_rows.append(dict(rec))
            else:
                raise TypeError(
                    f"_write_records expected Record or Mapping rows; "
                    f"got {type(rec).__name__}: {rec!r}"
                )
            if len(chunk_rows) >= chunk_size:
                _flush()
        _flush()
        if options.sync_metadata:
            self._commit_metadata()

    # ==================================================================
    # ``to_*`` aliases — pandas-style spelling for the ``read_*`` surface.
    # ==================================================================

    to_arrow = read_arrow_table
    to_arrow_batches = read_arrow_batches
    to_arrow_table = read_arrow_table
    to_arrow_batch_reader = read_arrow_batch_reader
    to_arrow_dataset = read_arrow_dataset

    to_table = read_table

    to_polars = read_polars_frame
    to_polars_frame = read_polars_frame

    to_pandas = read_pandas_frame
    to_pandas_frame = read_pandas_frame

    to_spark = read_spark_frame
    to_spark_frame = read_spark_frame

    to_pylist = read_pylist
    to_pydict = read_pydict
    to_record_iterator = read_record_iterator
    to_records = read_records