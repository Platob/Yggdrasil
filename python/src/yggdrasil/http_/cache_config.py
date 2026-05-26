from __future__ import annotations

import datetime as dt
import logging
import pathlib
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional

from yggdrasil.execution.expr import Predicate
from yggdrasil.data.cast.datetime import truncate_datetime
from yggdrasil.enums import Mode

if TYPE_CHECKING:
    from yggdrasil.io.tabular.base import Tabular

LOGGER = logging.getLogger(__name__)

_DEFAULT_CACHE_ROOT: pathlib.Path = pathlib.Path.home() / ".cache" / "http" / "response"


MATCH_COLUMN: str = "request_public_hash"
MATCH_KEY: str = "public_hash"

_CACHE_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "tabular",
        "mode",
        "anonymize",
        "received_from",
        "received_to",
        "cleanup_ttl",
    }
)

_SEND_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "wait",
        "raise_error",
        "remote_cache",
        "local_cache",
        "cache_only",
        "spark_session",
    }
)

# Default cap on how long the batcher will wait for an upstream
# request iterator to fill a chunk before flushing what it has
# (seconds). Bounds tail latency when requests are produced by a
# slow generator (paginated API, streaming SQL, etc.) without
# forcing the caller to think about a TTL up front.
DEFAULT_MAX_BATCH_TTL: float = 300.0



_TRUNCATE_INTERVAL = dt.timedelta(minutes=1)


def _truncate_from(value: Any) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    return truncate_datetime(value, _TRUNCATE_INTERVAL)


def _truncate_to(value: Any) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    return truncate_datetime(value, _TRUNCATE_INTERVAL, add_interval=True)


class CacheConfig:
    _FIELD_NAMES: ClassVar[frozenset[str]] = _CACHE_CONFIG_FIELDS

    __slots__ = (
        "tabular", "mode", "anonymize",
        "received_from", "received_to", "cleanup_ttl", "_derived",
    )


    @classmethod
    def default(cls):
        inst = cls.__dict__.get("_DEFAULT_INSTANCE")
        if inst is None:
            inst = cls()
            type.__setattr__(cls, "_DEFAULT_INSTANCE", inst)
        return inst

    @classmethod
    def parse_mapping(cls, options: Mapping[str, Any], **overrides: Any):
        values = {k: v for k, v in options.items() if k in cls._FIELD_NAMES}
        values.update(overrides)
        if cls._matches_default(values):
            return cls.default()
        return cls(**cls._check_mapping(values))

    def __init__(
        self,
        tabular: Optional[Holder] = None,
        mode: Mode = Mode.APPEND,
        anonymize: Literal["remove", "redact"] = "remove",
        received_from: Optional[dt.datetime] = None,
        received_to: Optional[dt.datetime] = None,
        cleanup_ttl: Optional[dt.timedelta] = dt.timedelta(days=1),
    ):
        self.mode = Mode.from_(mode, default=Mode.APPEND)
        self.anonymize = anonymize
        self.cleanup_ttl = cleanup_ttl
        self._derived = None

        self.received_from = _truncate_from(received_from)
        self.received_to = _truncate_to(received_to)

        if tabular is not None:
            from yggdrasil.io.tabular import Tabular
            self.tabular = Tabular.from_(tabular, as_folder=True)
        else:
            self.tabular = None

    def __repr__(self):
        parts = []
        if self.tabular is not None:
            parts.append(f"tabular={self.tabular!r}")
        if self.mode != Mode.APPEND:
            parts.append(f"mode={self.mode!r}")
        if self.received_from is not None:
            parts.append(f"received_from={self.received_from!r}")
        if self.received_to is not None:
            parts.append(f"received_to={self.received_to!r}")
        return f"CacheConfig({', '.join(parts)})"

    def __eq__(self, other):
        if not isinstance(other, CacheConfig):
            return NotImplemented
        return (
            self.tabular is other.tabular
            and self.mode == other.mode
            and self.anonymize == other.anonymize
            and self.received_from == other.received_from
            and self.received_to == other.received_to
            and self.cleanup_ttl == other.cleanup_ttl
        )

    def __hash__(self):
        return hash((
            id(self.tabular), self.mode, self.anonymize,
            self.received_from, self.received_to,
            self.cleanup_ttl,
        ))

    @staticmethod
    def _check_mapping(values: MutableMapping[str, Any]):
        cleanup_ttl = values.get("cleanup_ttl")
        if cleanup_ttl is not None:
            values["cleanup_ttl"] = any_to_timedelta(cleanup_ttl)

        received_from = values.get("received_from")
        if received_from is not None:
            values["received_from"] = _truncate_from(received_from)

        received_to = values.get("received_to")
        if received_to is not None:
            values["received_to"] = _truncate_to(received_to)

        # ``tabular`` accepts a live Tabular or a path-shaped sugar;
        # ``__post_init__`` resolves the sugar to a FolderPath. We
        # don't pre-coerce here so the canonical normalisation lives
        # in one place.

        return values

    @classmethod
    def _matches_default(cls, values: Mapping[str, Any]) -> bool:
        # CacheConfig treats ``None`` as a real value for some fields —
        # ``cleanup_ttl=None`` explicitly disables cache cleanup vs.
        # the default ``timedelta(days=1)``, ``wait=None`` flips wait
        # behavior off, etc. — and :meth:`_check_mapping` deliberately
        # keeps Nones rather than filtering them, so we can't reuse
        # the base implementation's "skip on None" shortcut.
        if not values:
            return True
        default = cls.default()
        for k, v in values.items():
            if getattr(default, k, ...) != v:
                return False
        return True

    def __getstate__(self):
        return {
            "mode": self.mode,
            "tabular": self.tabular,
            "received_from": self.received_from,
            "received_to": self.received_to,
            "anonymize": self.anonymize,
            "cleanup_ttl": self.cleanup_ttl,
        }

    def __setstate__(self, state):
        self.mode = state.get("mode", Mode.APPEND)
        self.received_from = state["received_from"]
        self.received_to = state["received_to"]
        tabular = state.get("tabular")
        if tabular is None:
            tabular_url = state.get("tabular_url", state.get("path"))
            if tabular_url is not None:
                from yggdrasil.io.nested.folder_path import FolderPath
                from yggdrasil.path import Path as _Path
                tabular = FolderPath(path=_Path.from_(tabular_url))
        self.tabular = tabular
        self.anonymize = state.get("anonymize", "remove")
        self.cleanup_ttl = state.get("cleanup_ttl", dt.timedelta(days=1))
        self._derived = None

    @classmethod
    def from_(
        cls,
        arg: "CacheConfig | Mapping[str, Any] | None",
        *,
        default: Any = ...,
        **overrides: Any,
    ) -> "CacheConfig":
        try:
            if arg is None:
                # Don't reuse :meth:`_matches_default` here:
                # :meth:`CacheConfig._check_mapping` intentionally does
                # *not* drop None — passing ``cleanup_ttl=None`` is the
                # documented way to disable cache cleanup — so collapsing
                # ``CacheConfig.from_(None, cleanup_ttl=None)`` back to
                # the default singleton would silently re-enable cleanup.
                return cls.parse_mapping(overrides) if overrides else cls.default()
            if isinstance(arg, cls):
                return arg.merge(**overrides) if overrides else arg
            if isinstance(arg, Mapping):
                return cls.parse_mapping(arg, **overrides)

            if isinstance(arg, dt.datetime):
                overrides["received_from"] = arg
            elif isinstance(arg, dt.date):
                overrides["received_from"] = dt.datetime.combine(arg, dt.time.min, tzinfo=dt.timezone.utc)

            elif isinstance(arg, dt.timedelta) or (
                isinstance(arg, (int, float)) and not isinstance(arg, bool)
            ):
                ttl = arg if isinstance(arg, dt.timedelta) else any_to_timedelta(arg)
                received_to = overrides.get("received_to")
                received_to = dt.datetime.now(dt.timezone.utc) if received_to is None else any_to_datetime(received_to)
                overrides["received_to"] = received_to
                if not overrides.get("received_from"):
                    overrides["received_from"] = received_to - ttl

            else:
                from yggdrasil.io.tabular import Tabular
                overrides["tabular"] = Tabular.from_(arg, as_folder=True)

            return cls.parse_mapping(overrides) if overrides else cls.default()
        except (TypeError, ValueError):
            if default is ...:
                raise
            return default

    def _derived_cache(self) -> dict:
        """Lazy-initialised dict for memoized derived properties.

        Every entry below is a pure function of the frozen fields — we
        compute on first access, stash it on ``_derived``, and serve
        the cached value on every subsequent read. Pickle round-trips
        leave ``_derived = None`` so a worker rebuilds its own cache.
        """
        cache = self._derived
        if cache is None:
            cache = {}
            self._derived = cache
        return cache

    @property
    def cache_enabled(self):
        return self.mode != Mode.IGNORE

    @property
    def local_cache_enabled(self):
        return self.cache_enabled

    @property
    def remote_cache_enabled(self):
        if not self.cache_enabled:
            return False
        return self.tabular is not None


    @property
    def defined_received_from(self) -> dt.datetime:
        if self.received_from:
            return self.received_from.timestamp()

        return dt.datetime.fromtimestamp(
            0,
            tz=dt.timezone.utc,
        )

    @property
    def defined_received_to(self) -> dt.datetime:
        if self.received_to:
            return self.received_to.timestamp()

        return dt.datetime.fromtimestamp(
            time.time() + 3600,
            tz=dt.timezone.utc,
        )

    def local_cache_folder(self, session: "Session | None" = None) -> Path:
        """Backend-agnostic root for the local cache.

        Returns the bound :class:`FolderPath`'s :attr:`path` when
        :attr:`tabular` is local (any :class:`yggdrasil.io.path.Path`
        subclass — LocalPath on disk, VolumePath on a Databricks
        Volume, S3Path on a bucket, …); otherwise builds the default
        LocalPath under ``~/.cache/http/response``, suffixed
        with the session's ``base_url`` host + path when one is
        available so different APIs sharing the same machine don't
        collide on disk:

        * ``base_url=https://api.example.com/v1/`` → ``…/response/api.example.com/v1``
        * ``base_url`` unset → ``…/response/default``

        Used as the per-config key for grouping cache hits in
        :class:`yggdrasil.http_.response_batch.HTTPResponseBatch`.
        """
        tab = self.tabular
        if tab is not None and hasattr(tab, "path"):
            return tab.path
        root = _DEFAULT_CACHE_ROOT
        base_url = getattr(session, "base_url", None) if session is not None else None
        host = getattr(base_url, "host", None) if base_url is not None else None
        if not host:
            folder = root / "default"
        else:
            url_path = (getattr(base_url, "path", "") or "").strip("/")
            folder = root / host / url_path if url_path else root / host
        from yggdrasil.path import Path as _Path
        return _Path.from_(folder)

    def cache_tabular(
        self, session: "Session | None" = None,
    ) -> "Any":
        """Return the active cache backend as a :class:`Tabular`.

        Single entry point both the local and remote pipelines call
        through:

        * :attr:`tabular` already set (constructor-supplied: live
          :class:`FolderPath` for local, Databricks Table or any
          third-party adapter for remote) — returned as-is.
        * Unset but the cache is otherwise enabled (``received_*``
          window) — the default
          ``~/.yggdrasil/cache/response/...`` :class:`FolderPath`
          is materialised via :meth:`local_cache_folder` and
          memoised back on :attr:`tabular` so the next call
          short-circuits.

        The on-disk layout is Hive-partitioned by whichever fields
        :meth:`partition_columns` reports (RESPONSE_SCHEMA's
        ``partition_by`` set — ``partition_key`` today), so the
        local Folder and the remote Table accept the same logical
        lookup primitive — the :class:`Predicate` built by
        :meth:`make_lookup_predicate` /
        :meth:`make_batch_lookup_predicate` — and the same
        :meth:`Tabular.write_arrow_batches` write call.
        """
        if self.tabular is not None:
            return self.tabular
        from yggdrasil.io.nested.folder_path import FolderPath

        tabular = FolderPath(path=self.local_cache_folder(session=session))
        # Stash the built folder back on the config so subsequent
        # cache scans reuse the same instance — the schema cache and
        # predicate ``free_columns`` memo only stay warm across calls
        # if the FolderPath is itself reused. ``tabular`` is
        # ``compare=False, hash=False`` and excluded from
        # ``__getstate__``, so the mutation doesn't affect equality
        # or pickling.
        self.tabular = tabular
        return tabular

    # ------------------------------------------------------------------
    # Cache read / write — unified surface for both session pipelines
    # ------------------------------------------------------------------

    def read_responses(
        self,
        requests: "Iterable[PreparedRequest]",
        *,
        spark_session: "Any" = None,
        session: "Session | None" = None,
    ) -> "tuple[list[Response], list[PreparedRequest]]":
        """Read cache hits as :class:`Response` objects.

        Returns ``(hits, misses)`` — matched by ``public_hash``
        and filtered by ``received_from`` / ``received_to``.
        """
        from yggdrasil.io.response import Response

        tab = self.read_responses_tabular(
            requests, spark_session=spark_session, session=session,
        )
        request_list = list(requests) if not isinstance(requests, list) else requests
        if tab is None:
            return [], list(request_list)

        result_map: dict[int, Response] = {}
        for response in Response.from_arrow_tabular(tab.read_arrow_batches()):
            req = response.request
            if req is None:
                continue
            key = req.match_value(MATCH_KEY)
            existing = result_map.get(key)
            if existing is None or response.received_at >= existing.received_at:
                result_map[key] = response

        hits: list[Response] = []
        misses: list[PreparedRequest] = []
        for req in request_list:
            key = req.match_value(MATCH_KEY)
            candidate = result_map.get(key)
            if candidate is not None and self.filter_response(candidate, request=req):
                hits.append(candidate)
            else:
                misses.append(req)

        return hits, misses

    def read_responses_tabular(
        self,
        requests: "Iterable[PreparedRequest]",
        *,
        spark_session: "Any" = None,
        session: "Session | None" = None,
    ) -> "Tabular | None":
        """Read matching cache rows as a :class:`Tabular`.

        Builds the batch lookup predicate from *requests* and reads
        from :attr:`tabular` (or the default local cache folder).
        """
        from yggdrasil.data.options import CastOptions
        from yggdrasil.io.response import RESPONSE_SCHEMA

        holder = self.tabular or self.cache_tabular(session=session)
        if holder is None:
            return None

        request_list = list(requests) if not isinstance(requests, list) else requests
        predicate = self.make_batch_lookup_predicate(request_list)
        opts = CastOptions(
            predicate=predicate,
            spark_session=spark_session,
            target=RESPONSE_SCHEMA,
        )
        return holder.read_table(options=opts)

    def write_responses(
        self,
        responses: "list[Response]",
        *,
        mode: "Mode | None" = None,
        spark_session: "Any" = None,
        session: "Session | None" = None,
    ) -> None:
        """Write :class:`Response` objects to the cache backend."""
        if not responses:
            return
        import pyarrow as pa
        from yggdrasil.io.response import Response

        table = pa.Table.from_batches(
            [Response.values_to_arrow_batch(responses)]
        )
        self.write_responses_tabular(
            table, mode=mode,
            spark_session=spark_session, session=session,
        )

    def write_responses_tabular(
        self,
        data: "Any",
        *,
        mode: "Mode | None" = None,
        spark_session: "Any" = None,
        session: "Session | None" = None,
    ) -> None:
        """Write Arrow / Spark response data to the cache backend."""
        import pyarrow as pa
        from yggdrasil.data.options import CastOptions

        if data is None:
            return

        holder = self.tabular or self.cache_tabular(session=session)
        if holder is None:
            return

        opts = CastOptions(
            mode=mode if mode is not None else self.mode,
            match_by=[MATCH_COLUMN],
            spark_session=spark_session,
        )
        try:
            if not isinstance(data, (pa.RecordBatch, pa.Table)) and (
                hasattr(data, "toArrow") or hasattr(data, "toPandas")
            ):
                holder.write_spark_frame(data, options=opts)
            elif isinstance(data, pa.RecordBatch):
                if data.num_rows > 0:
                    holder.write_arrow_batches((data,), options=opts)
            elif isinstance(data, pa.Table):
                if data.num_rows > 0:
                    holder.write_arrow_batches(data.to_batches(), options=opts)
            else:
                holder.write_arrow_batches(data, options=opts)
        except Exception:
            LOGGER.warning(
                "Cache write failed for %r", holder, exc_info=True,
            )

    def filter_response(
        self,
        response: "Response",
        request: PreparedRequest | None = None,
    ) -> bool:
        if request is not None:
            if response.match_value(MATCH_KEY) != request.match_value(MATCH_KEY):
                return False
        if self.received_from is not None and response.received_at < self.received_from:
            return False
        if self.received_to is not None and response.received_at >= self.received_to:
            return False
        return True

    # ------------------------------------------------------------------
    # Predicate builders — single source of truth for cache lookups
    # ------------------------------------------------------------------
    #
    # Both backends (local :class:`FolderPath` and remote
    # :class:`~yggdrasil.databricks.table.table.Table`) read through
    # :meth:`Tabular.read_arrow_batches` with the same ``options.predicate``
    # — :class:`Field`-aware backends translate the predicate to whatever
    # their engine speaks (Arrow ``RecordBatch.filter`` for the
    # folder, SQL ``WHERE`` for Databricks Tables). No SQL is built
    # here.

    @staticmethod
    def request_predicate(
        request: PreparedRequest | None,
    ) -> "Any | None":
        if request is None:
            return None
        from yggdrasil.execution.expr import col

        value = request.match_value(MATCH_KEY)
        return col(MATCH_COLUMN).is_null() if value is None else col(MATCH_COLUMN) == value

    def make_lookup_predicate(
        self,
        request: PreparedRequest | None = None,
    ) -> "Any | None":
        """Single-request :class:`Predicate` for the cache read.

        Shape: ``partition_key == <req.partition_key>`` AND the
        per-request match clause.
        """
        from yggdrasil.execution.expr import all_of, col

        clauses: list[Any] = []
        if request is not None:
            clauses.append(col("partition_key") == request.partition_key)
        req_pred = self.request_predicate(request)
        if req_pred is not None:
            clauses.append(req_pred)
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return all_of(*clauses)

    def make_batch_lookup_predicate(
        self,
        requests: Iterable[PreparedRequest],
    ) -> "Any | None":
        """Batch :class:`Predicate` for the cache read.

        Shape: ``partition_key IN (<distinct keys>)`` AND
        ``(req1_match) OR (req2_match) OR …``.
        """
        from yggdrasil.execution.expr import (
            all_of,
            any_of,
            col,
        )

        request_list = list(requests)
        if not request_list:
            return None

        clauses: list[Any] = []
        partition_keys = sorted({r.partition_key for r in request_list})
        clauses.append(col("partition_key").is_in(partition_keys))

        request_preds = [
            pred
            for pred in (self.request_predicate(r) for r in request_list)
            if pred is not None
        ]
        if len(request_preds) == 1:
            clauses.append(request_preds[0])
        elif request_preds:
            clauses.append(any_of(*request_preds))

        if len(clauses) == 1:
            return clauses[0]
        return all_of(*clauses)

    def merge(self, **overrides: Any):
        unknown = set(overrides) - self._FIELD_NAMES
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.merge got unexpected field(s): {sorted(unknown)!r}"
            )
        return self.copy(**self._check_mapping(overrides))

    def copy(self, **overrides):
        clean = {k: v for k, v in overrides.items() if v is not ...}
        if not clean:
            return self
        state = self.__getstate__()
        state.update(clean)
        return type(self)(**state)


DEFAULT_CACHE_CONFIG = CacheConfig()
