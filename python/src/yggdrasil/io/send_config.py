from __future__ import annotations

import dataclasses
import datetime as dt
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal, Mapping, MutableMapping, Optional, TYPE_CHECKING

from yggdrasil.data.cast import any_to_datetime, any_to_timedelta
from yggdrasil.dataclasses import DEFAULT_WAITING_CONFIG
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.data.enums import Mode
from yggdrasil.io.request import REQUEST_ARROW_SCHEMA, PreparedRequest
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, RESPONSE_SCHEMA

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from yggdrasil.io.tabular import Tabular
    from yggdrasil.io.nested.folder_io import FolderIO
    from yggdrasil.io.response import Response
    from yggdrasil.io.session import Session


__all__ = ["CacheConfig", "SendConfig", "SendManyConfig"]


# Identity-by-default — ``public_url_hash`` is the URL-based identity
# computed against ``url.anonymize('remove')``, so it stays stable
# across the cache's anonymize step (writes drop userinfo + sensitive
# query params; reads strip the same; both sides hash to the same
# int64). For dedup that should also respect body / headers, callers
# can pass ``request_by=["public_hash"]`` explicitly.
_DEFAULT_REQUEST_BY: tuple[str, ...] = (
    "public_url_hash",
)

_CACHE_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "tabular",
        "request_by",
        "response_by",
        "mode",
        "anonymize",
        "received_from",
        "received_to",
        "wait",
        "mirror_local_to_remote",
        "optimize_on_write",
    }
)

_SEND_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "wait",
        "raise_error",
        "stream",
        "remote_cache",
        "local_cache",
        "spark_session",
    }
)

_SEND_MANY_CONFIG_FIELDS: frozenset[str] = _SEND_CONFIG_FIELDS | frozenset(
    {
        "normalize",
        "batch_size",
        "ordered",
        "max_in_flight",
        "max_batch_size",
        "max_batch_ttl",
    }
)


# Default cap on how long the batcher will wait for an upstream
# request iterator to fill a chunk before flushing what it has
# (seconds). Bounds tail latency when requests are produced by a
# slow generator (paginated API, streaming SQL, etc.) without
# forcing the caller to think about a TTL up front.
DEFAULT_MAX_BATCH_TTL: float = 300.0


def _is_valid_request_key(key: str) -> bool:
    head, _, _ = key.partition(".")
    if head in REQUEST_ARROW_SCHEMA.names:
        return True
    # Accept the flattened ``request_<col>`` form too — same column,
    # different spelling — so callers can write the request_by keys
    # in either shape.
    if head.startswith("request_") and head[len("request_"):] in REQUEST_ARROW_SCHEMA.names:
        return True
    return False


def _is_valid_response_key(key: str) -> bool:
    head, _, _ = key.partition(".")
    return head in RESPONSE_ARROW_SCHEMA.names


def _request_column_sql_name(key: str) -> str:
    """SQL column name for a request-side ``request_by`` key.

    The response-cache table stores requests as flattened
    ``request_<col>`` columns (cf. :data:`RESPONSE_SCHEMA`), so a
    user-supplied ``request_by`` key that names a bare request column
    (``public_url_hash``, ``method`` …) needs the ``request_`` prefix
    when emitted into SQL. Already-prefixed keys pass through.
    """
    head, sep, tail = key.partition(".")
    if head in REQUEST_ARROW_SCHEMA.names:
        prefixed = f"request_{head}"
    else:
        prefixed = head
    return prefixed + (sep + tail if sep else "")


def _validate_request_by(arg: list[str] | tuple[str, ...] | None = None) -> list[str]:
    keys = list(_DEFAULT_REQUEST_BY if not arg else arg)
    invalid = [key for key in keys if not _is_valid_request_key(key)]
    if invalid:
        raise ValueError(
            f"Invalid request_by key(s): {invalid!r}. "
            f"Must be within: {REQUEST_ARROW_SCHEMA.names!r}"
        )
    return keys


def _validate_response_by(
    arg: list[str] | tuple[str, ...] | None = None,
) -> list[str] | None:
    if arg is None:
        return None

    keys = list(arg)
    invalid = [key for key in keys if not _is_valid_response_key(key)]
    if invalid:
        raise ValueError(
            f"Invalid response_by key(s): {invalid!r}. "
            f"Must be within: {RESPONSE_ARROW_SCHEMA.names!r}"
        )
    return keys


def _is_tabular_io(arg: Any) -> bool:
    """Duck-test ``arg`` for a :class:`Tabular`-shaped object.

    Used by :meth:`CacheConfig.check_arg` so the test doesn't pull
    in :class:`Tabular` (and its transitive ``yggdrasil.io.buffer``
    imports) at config-construction time. Anything that exposes
    both ``read_arrow_batches`` and ``write_arrow_batches`` qualifies
    — covers :class:`FolderIO`, :class:`Table`, and any third-party
    adapter following the same surface.
    """
    return (
        callable(getattr(arg, "read_arrow_batches", None))
        and callable(getattr(arg, "write_arrow_batches", None))
    )


def _folderio_for_local_cache(path: Path) -> "FolderIO":
    """Wrap a local filesystem :class:`Path` into a partitioned cache folder.

    Returns a :class:`YGGFolderIO` rooted at *path* and bound to
    :data:`RESPONSE_SCHEMA`. The schema's ``partition_by``-tagged
    fields drive the Hive layout automatically; partition pruning
    on read uses ``options.prune_values`` (which the cache flow
    in :mod:`yggdrasil.io.session` populates from the request
    batch's ``partition_key`` set).
    """
    from yggdrasil.io.nested.ygg_folder_io import YGGFolderIO
    from yggdrasil.io.path import LocalPath

    return YGGFolderIO(path=LocalPath(path), schema=RESPONSE_SCHEMA)


def _coerce_optional_datetime(value: Any) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return value
    return any_to_datetime(value)


# Hot dimensions for partition pruning of the local response cache —
# kept here for callers that introspect the cache layout. The same
# tag now lives on :data:`RESPONSE_SCHEMA` itself (set via the
# schema-level ``partition_by`` autotag), so :class:`FolderIO`
# derives the Hive partition layout straight from the schema without
# any per-cache rewriting.
LOCAL_CACHE_PARTITION_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in RESPONSE_SCHEMA.children_fields if f._tag_flag(b"partition_by")
)


@dataclass(frozen=True, slots=True)
class _ConfigBase:
    _FIELD_NAMES: ClassVar[frozenset[str]]

    @classmethod
    def default(cls):
        return cls()

    @classmethod
    def parse_mapping(cls, options: Mapping[str, Any], **overrides: Any):
        if not isinstance(options, Mapping):
            raise TypeError(
                f"{cls.__name__}.parse_mapping expects a Mapping, "
                f"got {type(options).__name__!r}"
            )
        values = {k: v for k, v in options.items() if k in cls._FIELD_NAMES}
        values.update(overrides)
        return cls(**cls._check_mapping(values))

    @staticmethod
    def _check_mapping(values: MutableMapping[str, Any]):
        spark_session = values.get("spark_session")
        if spark_session is not None and isinstance(spark_session, bool):
            if spark_session:
                from yggdrasil.environ import PyEnv

                values["spark_session"] = PyEnv.spark_session(
                    create=True,
                    install_spark=False,
                    import_error=True,
                )
            else:
                values["spark_session"] = None

        wait = values.get("wait")
        if wait is not None:
            values["wait"] = WaitingConfig.from_(wait)

        remote_cache = values.get("remote_cache")
        if remote_cache is not None:
            values["remote_cache"] = CacheConfig.check_arg(remote_cache)

        local_cache = values.get("local_cache")
        if local_cache is not None:
            values["local_cache"] = CacheConfig.check_arg(local_cache)

        return {
            k: v
            for k, v in values.items()
            if v is not None
        }

    def merge(self, **overrides: Any):
        unknown = set(overrides) - self._FIELD_NAMES
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.merge got unexpected field(s): {sorted(unknown)!r}"
            )
        return dataclasses.replace(self, **self._check_mapping(overrides))


@dataclass(frozen=True, slots=True)
class CacheConfig(_ConfigBase):
    _FIELD_NAMES: ClassVar[frozenset[str]] = _CACHE_CONFIG_FIELDS

    # Unified backend slot — accepts any :class:`Tabular` subclass:
    # :class:`FolderIO` for an on-disk cache, :class:`Table` (Databricks)
    # for a remote cache, or any other registered tabular adapter.
    # Both backends share the same partitioning / primary-key /
    # match-by rules driven from RESPONSE_SCHEMA, so the cache flow
    # in Session is the same regardless of where the rows actually
    # land.
    tabular: Optional["Tabular"] = field(default=None, hash=False, compare=False)
    request_by: Optional[list[str]] = field(default=None, hash=False, compare=False)
    response_by: Optional[list[str]] = field(default=None, hash=False, compare=False)
    mode: Mode = Mode.APPEND
    anonymize: Literal["remove", "redact"] = "remove"
    received_from: Optional[dt.datetime] = None
    received_to: Optional[dt.datetime] = None
    received_ttl: Optional[dt.timedelta] = None
    wait: WaitingConfig = False
    # When True, the session pushes local-cache hits up to the remote
    # cache as a bulk UPSERT before stage 3 (network fetch). This is
    # the "diff sync" path: anything the local cache has that remote
    # might not — historical responses persisted only on disk, a
    # warm-started session, etc. — gets mirrored upstream in one
    # write per group instead of waiting for a fresh network call to
    # repopulate remote. Default False keeps the legacy behavior.
    mirror_local_to_remote: bool = False
    # When True (default), each :meth:`Session.send_many` batch
    # invokes :meth:`YGGFolderIO.optimize` on the local cache once
    # the batch's writes have settled — but scoped to the partition
    # tuples that batch actually touched. A typical batch spreads its
    # writes across a handful of ``partition_key=…`` directories;
    # compacting only those leaves means we never pay for a full-tree
    # walk just because a few new small files landed. Remote caches
    # ignore this flag (compaction is the warehouse's job). Set False
    # to opt out — useful when an external process already runs
    # OPTIMIZE on a schedule, or for very write-heavy bursts where
    # the per-batch fsync is unwelcome.
    optimize_on_write: bool = True

    @staticmethod
    def _check_mapping(values: MutableMapping[str, Any]):
        wait = values.get("wait")
        if wait is not None:
            values["wait"] = WaitingConfig.from_(wait)

        received_ttl = values.get("received_ttl")
        if received_ttl is not None:
            values["received_ttl"] = any_to_timedelta(received_ttl)

        received_from = values.get("received_from")
        if received_from is not None:
            values["received_from"] = _coerce_optional_datetime(received_from)

        received_to = values.get("received_to")
        if received_to is not None:
            values["received_to"] = _coerce_optional_datetime(received_to)

        return values

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", Mode.from_(self.mode, default=Mode.APPEND))
        object.__setattr__(self, "wait", WaitingConfig.from_(self.wait))

        object.__setattr__(self, "request_by", _validate_request_by(self.request_by))
        object.__setattr__(self, "response_by", _validate_response_by(self.response_by))

        object.__setattr__(self, "received_from", _coerce_optional_datetime(self.received_from))
        object.__setattr__(self, "received_to", _coerce_optional_datetime(self.received_to))

        if self.received_ttl:
            if not self.received_to:
                object.__setattr__(self, "received_to", dt.datetime.now(dt.timezone.utc))

            if not self.received_from:
                object.__setattr__(self, "received_from", self.received_to - self.received_ttl)

    def __getstate__(self):
        return {
            "mode": self.mode,
            "wait": self.wait,
            "request_by": self.request_by,
            "response_by": self.response_by,
            "received_from": self.received_from,
            "received_to": self.received_to,
            "received_ttl": self.received_ttl,
            "mirror_local_to_remote": self.mirror_local_to_remote,
            "optimize_on_write": self.optimize_on_write,
        }

    def __setstate__(self, state):
        object.__setattr__(self, "mode", state["mode"])
        object.__setattr__(self, "wait", state["wait"])
        object.__setattr__(self, "request_by", state["request_by"])
        object.__setattr__(self, "response_by", state["response_by"])
        object.__setattr__(self, "received_from", state["received_from"])
        object.__setattr__(self, "received_to", state["received_to"])
        object.__setattr__(self, "received_ttl", state["received_ttl"])
        # ``tabular`` is intentionally excluded from __getstate__ —
        # local FolderIO paths don't survive process boundaries and
        # remote Table handles wrap a live Databricks client. Init
        # to None so attribute access on the deserialized side
        # doesn't AttributeError.
        object.__setattr__(self, "tabular", state.get("tabular"))
        object.__setattr__(self, "anonymize", state.get("anonymize", "remove"))
        object.__setattr__(
            self, "mirror_local_to_remote",
            state.get("mirror_local_to_remote", False),
        )
        object.__setattr__(
            self, "optimize_on_write",
            state.get("optimize_on_write", True),
        )

    @classmethod
    def check_arg(
        cls,
        arg: "CacheConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "CacheConfig":
        if arg is None:
            return cls.parse_mapping(overrides) if overrides else cls.default()
        if isinstance(arg, cls):
            return arg.merge(**overrides) if overrides else arg
        if isinstance(arg, Mapping):
            return cls.parse_mapping(arg, **overrides)

        if isinstance(arg, Path):
            overrides["tabular"] = _folderio_for_local_cache(arg)

        elif _is_tabular_io(arg):
            overrides["tabular"] = arg

        elif isinstance(arg, dt.datetime):
            overrides["received_from"] = arg
        elif isinstance(arg, dt.date):
            overrides["received_from"] = dt.datetime.combine(arg, dt.time.min, tzinfo=dt.timezone.utc)

        elif isinstance(arg, dt.timedelta):
            overrides["received_ttl"] = arg

            # fill received_from and received_to if not exists
            received_to = overrides.get("received_to")
            received_to = dt.datetime.now(dt.timezone.utc) if received_to is None else any_to_datetime(received_to)
            overrides["received_to"] = received_to

            received_from = overrides.get("received_from")
            if not received_from:
                overrides["received_from"] = received_to - arg

        return cls.parse_mapping(overrides) if overrides else cls.default()

    @property
    def cache_enabled(self):
        return self.mode in (Mode.APPEND, Mode.AUTO)

    @property
    def is_local_tabular(self) -> bool:
        """True when ``tabular`` is a :class:`FolderIO` (on-disk cache).

        Used to dispatch between FolderIO write semantics
        (``write_arrow_batches`` + ``FolderOptions``) and
        Databricks-Table write semantics (``insert(..., match_by=...,
        prune_by=...)``) inside :class:`Session`.
        """
        from yggdrasil.io.nested.folder_io import FolderIO
        return isinstance(self.tabular, FolderIO)

    @property
    def local_cache_enabled(self):
        # Two ways to opt into a local cache layer:
        #   1) ``tabular`` is set to a FolderIO (explicit local backend);
        #   2) a ``received_from`` / ``received_to`` window is set, in
        #      which case ``local_cache()`` lazy-builds a FolderIO at
        #      the default path on first read.
        if not self.cache_enabled:
            return False
        if self.is_local_tabular:
            return True
        return self.received_from is not None or self.received_to is not None

    @property
    def remote_cache_enabled(self):
        return self.cache_enabled and self.tabular is not None and not self.is_local_tabular

    @property
    def match_by(self) -> list[str]:
        return [
            *(self.request_by or ()),
            *(self.response_by or ()),
        ]

    @property
    def sql_match_by(self) -> list[str]:
        # Cache-table column names for the merge join. Request-side keys
        # are stored on the response table under the flattened
        # ``request_<col>`` form (cf. :data:`RESPONSE_SCHEMA`), so a
        # bare ``public_url_hash`` / ``method`` / etc. needs the
        # ``request_`` prefix before it can be referenced as a target
        # column in a Delta MERGE. Response-side keys map 1:1 already.
        return [
            *(_request_column_sql_name(k) for k in (self.request_by or ())),
            *(self.response_by or ()),
        ]

    @property
    def request_by_is_public(self) -> bool:
        """True when every ``request_by`` key is anonymization-invariant.

        ``public_hash`` / ``public_url_hash`` (and any future ``public_*``
        column) are computed against ``url.anonymize('remove')`` plus
        ``normalize_headers(anonymize=True)``, so they hash to the same
        int64 whether the caller looks them up on the original request or
        on the anonymized form stored in the cache. When this predicate
        holds the lookup paths can skip the per-request ``anonymize()``
        pass before computing ``request_tuple`` / interpolating the SQL
        clause — the saving is one URL parse + one header normalize per
        request per lookup, which adds up on send_many bursts.
        """
        keys = self.request_by or ()
        return bool(keys) and all(str(k).startswith("public_") for k in keys)

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

    @staticmethod
    def sql_literal(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dt.datetime):
            return f"timestamp '{value.isoformat(sep=' ', timespec='microseconds')}'"
        if isinstance(value, bytes):
            import base64
            value = base64.b64encode(value).decode("ascii")
        else:
            value = str(value)
        return f"'{value.replace(chr(39), chr(39) * 2)}'"

    def local_cache_folder(self, session: "Session | None" = None) -> Path:
        """Filesystem root for the local cache.

        Returns ``self.tabular.path`` when ``tabular`` is a
        :class:`FolderIO`. Otherwise builds the default path under
        ``~/.yggdrasil/cache/response``, suffixed with the session's
        ``base_url`` host + path when one is available so different
        APIs sharing the same machine don't collide on disk:

        * ``base_url=https://api.example.com/v1/`` → ``…/response/api.example.com/v1``
        * ``base_url`` unset → ``…/response/default``

        Used as the per-config key for grouping cache hits in
        :class:`yggdrasil.io.response_batch.ResponseBatch`.
        """
        if self.is_local_tabular:
            return Path(str(self.tabular.path))
        root = Path.home() / ".yggdrasil" / "cache" / "response"
        base_url = getattr(session, "base_url", None) if session is not None else None
        host = getattr(base_url, "host", None) if base_url is not None else None
        if not host:
            return root / "default"
        path = (getattr(base_url, "path", "") or "").strip("/")
        return root / host / path if path else root / host

    def local_cache(self, session: "Session | None" = None) -> "FolderIO":
        """Return the local-cache folder.

        Returns ``self.tabular`` when it's already a FolderIO,
        otherwise lazy-builds a :class:`YGGFolderIO` rooted at
        :meth:`local_cache_folder` (and caches it back into
        ``tabular`` so subsequent calls return the same instance).
        The schema is :data:`RESPONSE_SCHEMA` — its
        ``partition_by``-tagged ``partition_key`` column drives the
        Hive layout automatically; the ``.ygg/`` sidecar lets the
        cache attach stats / checkpoints without a separate
        wrapper.

        Pass *session* so the lazy default path picks up the
        session's ``base_url`` host/path (see :meth:`local_cache_folder`).
        """
        if self.is_local_tabular:
            folder = self.tabular  # type: ignore[assignment]
        else:
            folder = _folderio_for_local_cache(self.local_cache_folder(session=session))
            # Cache the lazy-built FolderIO so repeated send_many()
            # calls don't keep re-instantiating it. Frozen-dataclass
            # safe via ``object.__setattr__``.
            if self.tabular is None:
                object.__setattr__(self, "tabular", folder)

        # Sweep stale part files (older than 1 day) at most once per
        # day per cache root. The throttle (sentinel + in-process
        # done-set) lives on :class:`YGGFolderIO` so any consumer of
        # the protocol — not just the response cache — benefits.
        # ``cleanup_stale_once`` is best-effort and never raises;
        # non-YGG backends just don't expose it.
        cleanup_once = getattr(folder, "cleanup_stale_once", None)
        if callable(cleanup_once):
            cleanup_once()
        return folder  # type: ignore[return-value]

    def request_values(
        self,
        request: PreparedRequest,
    ) -> dict[str, Any]:
        return {
            key: request.match_value(key)
            for key in (self.request_by or [])
        }

    def response_values(
        self,
        response: "Response",
    ) -> dict[str, Any]:
        return {key: response.match_value(key) for key in (self.response_by or [])}

    def filter_request(
        self,
        request: PreparedRequest,
    ) -> bool:
        for key in self.request_by or []:
            request.match_value(key)
        return True

    def filter_response(
        self,
        response: "Response",
        request: PreparedRequest | None = None,
    ) -> bool:
        if request is not None:
            for key, expected in self.request_values(request).items():
                actual = response.match_value(key)
                if actual != expected:
                    return False

        for key in self.response_by or []:
            response.match_value(key)

        if self.received_from is not None:
            if response.received_at < self.received_from:
                return False

        if self.received_to is not None:
            if response.received_at >= self.received_to:
                return False

        return True

    def request_tuple(
        self,
        request: PreparedRequest,
    ) -> tuple[Any, ...]:
        values = self.request_values(request)
        return tuple(values[key] for key in (self.request_by or []))

    def response_tuple(
        self,
        response: "Response",
    ) -> tuple[Any, ...]:
        values = self.response_values(response)
        return tuple(values[key] for key in (self.response_by or []))

    def identity_tuple(
        self,
        response: "Response",
        request: PreparedRequest | None = None,
    ) -> tuple[Any, ...]:
        out: list[Any] = []
        if request is not None:
            out.extend(self.request_tuple(request))
        out.extend(self.response_tuple(response))
        return tuple(out)

    def sql_request_clause(
        self,
        request: PreparedRequest | None,
    ) -> str:
        clauses: list[str] = []

        if request is not None:
            for key, value in self.request_values(request).items():
                column = _request_column_sql_name(key)
                if value is None:
                    clauses.append(f"{column} IS NULL")
                else:
                    clauses.append(f"{column} = {self.sql_literal(value)}")

        return " AND ".join(clauses) if clauses else "1=1"

    def sql_response_clause(
        self,
        response: "Response | None" = None,
    ) -> str:
        clauses: list[str] = []

        if response is not None:
            for key, value in self.response_values(response).items():
                if value is None:
                    clauses.append(f"{key} IS NULL")
                else:
                    clauses.append(f"{key} = {self.sql_literal(value)}")

        if self.received_from is not None:
            clauses.append(f"received_at >= {self.sql_literal(self.received_from)}")

        if self.received_to is not None:
            clauses.append(f"received_at < {self.sql_literal(self.received_to)}")

        return " AND ".join(clauses) if clauses else "1=1"

    def sql_clause(
        self,
        request: PreparedRequest | None = None,
        response: "Response | None" = None,
    ) -> str:
        clauses: list[str] = []

        request_clause = self.sql_request_clause(request)
        if request_clause != "1=1":
            clauses.append(f"({request_clause})")

        response_clause = self.sql_response_clause(response)
        if response_clause != "1=1":
            clauses.append(f"({response_clause})")

        return " AND ".join(clauses) if clauses else "1=1"

    def make_lookup_sql(
        self,
        table_name: str,
        request: PreparedRequest | None = None,
        response: "Response | None" = None,
        *,
        identity_by: Optional[Iterable[str]] = None,
    ) -> str:
        where_clause = self.sql_clause(request=request, response=response)
        # Single-request partition prune — narrows the SQL engine's
        # data-file scan to one partition before any per-row predicate.
        if request is not None:
            partition_clause = (
                f"partition_key = {self.sql_literal(request.partition_key)}"
            )
            where_clause = (
                f"({partition_clause}) AND ({where_clause})"
                if where_clause != "1=1"
                else partition_clause
            )
        base_query = f"SELECT * FROM {table_name}"
        if where_clause != "1=1":
            base_query += f" WHERE {where_clause}"

        identity_cols = list(identity_by) if identity_by is not None else self.match_by
        if identity_cols:
            partition_by = ", ".join(
                _request_column_sql_name(col)
                if col.partition(".")[0] in REQUEST_ARROW_SCHEMA.names
                else col
                for col in identity_cols
            )
            return (
                "SELECT * FROM ("
                "  SELECT t.*, row_number() OVER ("
                f"    PARTITION BY {partition_by} "
                "    ORDER BY received_at DESC"
                "  ) AS __rn "
                f"  FROM ({base_query}) t"
                ") ranked WHERE __rn = 1"
            )

        return base_query

    def make_batch_lookup_sql(
        self,
        table_name: str,
        requests: Iterable[PreparedRequest],
        *,
        identity_by: Optional[Iterable[str]] = None,
    ) -> str:
        request_list = list(requests)
        request_clauses = " OR ".join(
            f"({self.sql_request_clause(req)})"
            for req in request_list
        )
        response_clause = self.sql_response_clause(None)

        # Partition prune: ``partition_key`` is the table's partition
        # column (declared on RESPONSE_SCHEMA). An ``IN (…)`` clause
        # over the request batch's distinct partition_keys lets the
        # SQL engine skip every other partition before evaluating the
        # per-request OR — turns a full-table scan into an N-partition
        # read on a partition-pruned engine (Delta / Iceberg / etc.).
        partition_clause = ""
        if request_list:
            partition_keys = sorted({r.partition_key for r in request_list})
            if partition_keys:
                literals = ", ".join(self.sql_literal(v) for v in partition_keys)
                partition_clause = f"partition_key IN ({literals})"

        where_parts: list[str] = []
        if partition_clause:
            where_parts.append(f"({partition_clause})")
        if request_clauses:
            where_parts.append(f"({request_clauses})")
        if response_clause != "1=1":
            where_parts.append(f"({response_clause})")

        base_query = f"SELECT * FROM {table_name}"
        if where_parts:
            base_query += " WHERE " + " AND ".join(where_parts)

        identity_cols = list(identity_by) if identity_by is not None else self.match_by
        if identity_cols:
            partition_by = ", ".join(
                _request_column_sql_name(col)
                if col.partition(".")[0] in REQUEST_ARROW_SCHEMA.names
                else col
                for col in identity_cols
            )
            return (
                "SELECT * FROM ("
                "  SELECT t.*, row_number() OVER ("
                f"    PARTITION BY {partition_by} "
                "    ORDER BY received_at DESC"
                "  ) AS __rn "
                f"  FROM ({base_query}) t"
                ") ranked WHERE __rn = 1"
            )

        return base_query

    def copy(
        self,
        **overrides
    ):
        clean = {
            k: v
            for k, v in overrides.items()
            if v is not ...
        }

        if not clean:
            return self

        return dataclasses.replace(self, **overrides)


DEFAULT_CACHE_CONFIG = CacheConfig()


@dataclass(frozen=True, slots=True)
class SendConfig(_ConfigBase):
    _FIELD_NAMES: ClassVar[frozenset[str]] = _SEND_CONFIG_FIELDS

    raise_error: bool = True
    stream: bool = True
    wait: WaitingConfig = field(default=DEFAULT_WAITING_CONFIG)
    remote_cache: CacheConfig = field(default=DEFAULT_CACHE_CONFIG)
    local_cache: CacheConfig = field(default=DEFAULT_CACHE_CONFIG)
    spark_session: Optional["SparkSession"] = field(
        default=None,
        hash=False,
        compare=False,
        repr=False,
    )

    def __post_init__(self):
        object.__setattr__(self, "wait", WaitingConfig.from_(self.wait))
        object.__setattr__(self, "remote_cache", CacheConfig.check_arg(self.remote_cache))
        object.__setattr__(self, "local_cache", CacheConfig.check_arg(self.local_cache))

    def __getstate__(self):
        return {
            "raise_error": self.raise_error,
            "stream": self.stream,
            "wait": self.wait,
            "remote_cache": self.remote_cache,
            "local_cache": self.local_cache,
            "spark_session": None,
        }

    def __setstate__(self, state):
        object.__setattr__(self, "raise_error", state["raise_error"])
        object.__setattr__(self, "stream", state["stream"])
        object.__setattr__(self, "wait", state["wait"])
        object.__setattr__(self, "remote_cache", state["remote_cache"])
        object.__setattr__(self, "local_cache", state["local_cache"])
        object.__setattr__(self, "spark_session", None)

    @classmethod
    def check_arg(
        cls,
        arg: "SendConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "SendConfig":
        if arg is None:
            return cls.parse_mapping(overrides) if overrides else cls.default()
        if isinstance(arg, cls):
            return arg.merge(**overrides) if overrides else arg
        if isinstance(arg, Mapping):
            return cls.parse_mapping(arg, **overrides)
        raise TypeError(
            f"{cls.__name__}.check_arg expects a {cls.__name__}, Mapping, or None; "
            f"got {type(arg).__name__!r}"
        )


@dataclass(frozen=True, slots=True)
class SendManyConfig(_ConfigBase):
    _FIELD_NAMES: ClassVar[frozenset[str]] = _SEND_MANY_CONFIG_FIELDS

    wait: WaitingConfigArg = None
    raise_error: bool = True
    stream: bool = True
    remote_cache: CacheConfig = field(default_factory=CacheConfig)
    local_cache: CacheConfig = field(default_factory=CacheConfig)
    spark_session: Optional["SparkSession"] = field(
        default=None,
        hash=False,
        compare=False,
        repr=False,
    )

    normalize: bool | None = None
    batch_size: int | None = None
    ordered: bool = False
    max_in_flight: int | None = None
    max_batch_size: int | None = None
    max_batch_ttl: float | None = DEFAULT_MAX_BATCH_TTL

    def __post_init__(self):
        object.__setattr__(self, "wait", WaitingConfig.from_(self.wait))
        object.__setattr__(self, "remote_cache", CacheConfig.check_arg(self.remote_cache))
        object.__setattr__(self, "local_cache", CacheConfig.check_arg(self.local_cache))

    def __getstate__(self):
        return {
            "wait": self.wait,
            "raise_error": self.raise_error,
            "stream": self.stream,
            "remote_cache": self.remote_cache,
            "local_cache": self.local_cache,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "ordered": self.ordered,
            "max_in_flight": self.max_in_flight,
            "max_batch_size": self.max_batch_size,
            "max_batch_ttl": self.max_batch_ttl,
            "spark_session": None,
        }

    def __setstate__(self, state):
        object.__setattr__(self, "wait", state["wait"])
        object.__setattr__(self, "raise_error", state["raise_error"])
        object.__setattr__(self, "stream", state["stream"])
        object.__setattr__(self, "remote_cache", state["remote_cache"])
        object.__setattr__(self, "local_cache", state["local_cache"])
        object.__setattr__(self, "normalize", state["normalize"])
        object.__setattr__(self, "batch_size", state["batch_size"])
        object.__setattr__(self, "ordered", state["ordered"])
        object.__setattr__(self, "max_in_flight", state["max_in_flight"])
        object.__setattr__(self, "max_batch_size", state.get("max_batch_size"))
        object.__setattr__(
            self, "max_batch_ttl", state.get("max_batch_ttl", DEFAULT_MAX_BATCH_TTL),
        )
        object.__setattr__(self, "spark_session", state["spark_session"])

    @classmethod
    def check_arg(
        cls,
        arg: "SendManyConfig | SendConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "SendManyConfig":
        if arg is None:
            return cls(**overrides) if overrides else cls.default()
        if isinstance(arg, cls):
            return arg.merge(**overrides) if overrides else arg
        if isinstance(arg, SendConfig):
            base = {
                "wait": arg.wait,
                "raise_error": arg.raise_error,
                "stream": arg.stream,
                "remote_cache": arg.remote_cache,
                "local_cache": arg.local_cache,
                "spark_session": arg.spark_session,
            }
            # Overrides win, but a None override means "no opinion" — fall back
            # to the base value so we don't silently clobber the parent config.
            for key, value in overrides.items():
                if value is not None:
                    base[key] = value
            return cls.parse_mapping(base)
        if isinstance(arg, Mapping):
            return cls.parse_mapping(arg, **overrides)
        raise TypeError(
            f"{cls.__name__}.check_arg expects a {cls.__name__}, SendConfig, "
            f"Mapping, or None; got {type(arg).__name__!r}"
        )

    def to_send_config(
        self,
        with_remote_cache: bool = True,
        with_local_cache: bool = True,
        with_spark: bool = False,
        raise_error: bool | None = None
    ) -> SendConfig:
        return SendConfig(
            wait=self.wait,
            raise_error=self.raise_error if raise_error is None else raise_error,
            stream=self.stream,
            remote_cache=self.remote_cache if with_remote_cache else CacheConfig(),
            local_cache=self.local_cache if with_local_cache else CacheConfig(),
            spark_session=self.spark_session if with_spark else None,
        )
