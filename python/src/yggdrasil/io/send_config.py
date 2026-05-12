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
from yggdrasil.environ import PyEnv
from yggdrasil.io.request import REQUEST_ARROW_SCHEMA, PreparedRequest
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, RESPONSE_SCHEMA

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from yggdrasil.io.tabular import Tabular
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
        "path",
        "request_by",
        "response_by",
        "mode",
        "anonymize",
        "received_from",
        "received_to",
        "wait",
        "mirror_local_to_remote",
        "cleanup_ttl",
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


def _coerce_optional_datetime(value: Any) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return value
    return any_to_datetime(value)


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
        if spark_session is not None:
            try:
                values["spark_session"] = PyEnv.spark_session(obj=spark_session)
            except Exception:
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

    # Local cache root directory. When set, the session writes each
    # successful response as one Arrow IPC file at
    # ``<path>/<METHOD>/<host>/<seg>/.../<public_hash>.arrow`` (see
    # :mod:`yggdrasil.io.session._local_fast_path_relative`). Stored
    # as a string so the frozen-dataclass instance pickles cleanly
    # across worker boundaries (Spark, multiprocessing, Power Query).
    path: Optional[str] = field(default=None, hash=False, compare=False)
    # Remote cache backend — a :class:`Tabular` subclass (Databricks
    # Table, …). Mutually exclusive with :attr:`path`: a single
    # config is either local *or* remote, not both.
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
    # TTL after which orphaned fast-path ``.arrow`` files in the
    # local cache tree are unlinked by the writer-side cleanup pass.
    # Default 1 day mirrors the previous YGGFolderIO behaviour. Set
    # to ``None`` to disable cleanup entirely (cache grows unbounded).
    cleanup_ttl: Optional[dt.timedelta] = dt.timedelta(days=1)
    # Lazy memo for derived predicates and SQL column lists
    # (``match_by``, ``sql_match_by``, ``request_by_is_public``, the
    # ``request_sql_column_names`` array, the partition_by SQL clause).
    # Each is purely a function of the frozen fields above — caching
    # turns a per-send (or per-batch-request) recomputation into a
    # dict lookup. Excluded from pickle / equality / repr; rebuilt
    # transparently after ``__setstate__``.
    _derived: Optional[dict] = field(
        default=None, init=False, hash=False, compare=False, repr=False,
    )

    @staticmethod
    def _check_mapping(values: MutableMapping[str, Any]):
        wait = values.get("wait")
        if wait is not None:
            values["wait"] = WaitingConfig.from_(wait)

        received_ttl = values.get("received_ttl")
        if received_ttl is not None:
            values["received_ttl"] = any_to_timedelta(received_ttl)

        cleanup_ttl = values.get("cleanup_ttl")
        if cleanup_ttl is not None:
            values["cleanup_ttl"] = any_to_timedelta(cleanup_ttl)

        received_from = values.get("received_from")
        if received_from is not None:
            values["received_from"] = _coerce_optional_datetime(received_from)

        received_to = values.get("received_to")
        if received_to is not None:
            values["received_to"] = _coerce_optional_datetime(received_to)

        path = values.get("path")
        if path is not None and not isinstance(path, str):
            values["path"] = str(path)

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

        if self.path is not None and not isinstance(self.path, str):
            object.__setattr__(self, "path", str(self.path))

        if self.path is not None and self.tabular is not None:
            raise ValueError(
                "CacheConfig accepts either ``path`` (local cache) or "
                "``tabular`` (remote cache), not both."
            )

    def __getstate__(self):
        return {
            "mode": self.mode,
            "wait": self.wait,
            "path": self.path,
            "request_by": self.request_by,
            "response_by": self.response_by,
            "received_from": self.received_from,
            "received_to": self.received_to,
            "received_ttl": self.received_ttl,
            "mirror_local_to_remote": self.mirror_local_to_remote,
            "cleanup_ttl": self.cleanup_ttl,
        }

    def __setstate__(self, state):
        object.__setattr__(self, "mode", state["mode"])
        object.__setattr__(self, "wait", state["wait"])
        object.__setattr__(self, "request_by", state["request_by"])
        object.__setattr__(self, "response_by", state["response_by"])
        object.__setattr__(self, "received_from", state["received_from"])
        object.__setattr__(self, "received_to", state["received_to"])
        object.__setattr__(self, "received_ttl", state["received_ttl"])
        object.__setattr__(self, "path", state.get("path"))
        # ``tabular`` is intentionally excluded from __getstate__ —
        # remote :class:`Tabular` handles wrap a live Databricks
        # client and don't survive process boundaries. Init to None
        # so attribute access on the deserialized side doesn't
        # AttributeError.
        object.__setattr__(self, "tabular", state.get("tabular"))
        object.__setattr__(self, "anonymize", state.get("anonymize", "remove"))
        object.__setattr__(
            self, "mirror_local_to_remote",
            state.get("mirror_local_to_remote", False),
        )
        object.__setattr__(
            self, "cleanup_ttl",
            state.get("cleanup_ttl", dt.timedelta(days=1)),
        )
        object.__setattr__(self, "_derived", None)

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
            overrides["path"] = str(arg)

        elif isinstance(arg, str):
            overrides["path"] = arg

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
            object.__setattr__(self, "_derived", cache)
        return cache

    @property
    def cache_enabled(self):
        cache = self._derived_cache()
        out = cache.get("cache_enabled", ...)
        if out is ...:
            out = self.mode in (Mode.APPEND, Mode.AUTO)
            cache["cache_enabled"] = out
        return out

    @property
    def is_local(self) -> bool:
        """True when this config drives the on-disk fast-path cache."""
        # ``path`` may be lazy-filled by :meth:`local_cache_path`, so
        # don't cache this one — read straight from the field.
        return self.path is not None

    @property
    def is_remote(self) -> bool:
        """True when this config drives a remote :class:`Tabular`-backed cache."""
        return self.tabular is not None

    @property
    def local_cache_enabled(self):
        # Two ways to opt into a local cache layer:
        #   1) ``path`` is set to a directory (explicit local backend);
        #   2) a ``received_from`` / ``received_to`` window is set,
        #      in which case ``local_cache_path()`` lazy-fills the
        #      default ``~/.yggdrasil/cache/response/...`` root.
        if not self.cache_enabled:
            return False
        if self.is_local:
            return True
        return self.received_from is not None or self.received_to is not None

    @property
    def remote_cache_enabled(self):
        return self.cache_enabled and self.is_remote

    @property
    def match_by(self) -> list[str]:
        cache = self._derived_cache()
        out = cache.get("match_by", ...)
        if out is ...:
            out = [
                *(self.request_by or ()),
                *(self.response_by or ()),
            ]
            cache["match_by"] = out
        return out

    @property
    def sql_match_by(self) -> list[str]:
        # Cache-table column names for the merge join. Request-side keys
        # are stored on the response table under the flattened
        # ``request_<col>`` form (cf. :data:`RESPONSE_SCHEMA`), so a
        # bare ``public_url_hash`` / ``method`` / etc. needs the
        # ``request_`` prefix before it can be referenced as a target
        # column in a Delta MERGE. Response-side keys map 1:1 already.
        cache = self._derived_cache()
        out = cache.get("sql_match_by", ...)
        if out is ...:
            out = [
                *(_request_column_sql_name(k) for k in (self.request_by or ())),
                *(self.response_by or ()),
            ]
            cache["sql_match_by"] = out
        return out

    @property
    def request_sql_columns(self) -> list[str]:
        """Cached ``_request_column_sql_name`` mapping for every ``request_by`` key.

        ``make_batch_lookup_sql`` walks N requests and emits the same
        column name for each (it depends only on the config), so
        precomputing the list cuts the per-request loop's cost — at
        ``BATCH_SIZE=64`` requests that is 64 ``_request_column_sql_name``
        calls replaced by one shared list of strings.
        """
        cache = self._derived_cache()
        out = cache.get("request_sql_columns", ...)
        if out is ...:
            out = [_request_column_sql_name(k) for k in (self.request_by or ())]
            cache["request_sql_columns"] = out
        return out

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
        cache = self._derived_cache()
        out = cache.get("request_by_is_public", ...)
        if out is ...:
            keys = self.request_by or ()
            out = bool(keys) and all(str(k).startswith("public_") for k in keys)
            cache["request_by_is_public"] = out
        return out

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

        Returns :attr:`path` when explicitly set, otherwise builds
        the default path under ``~/.yggdrasil/cache/response``,
        suffixed with the session's ``base_url`` host + path when
        one is available so different APIs sharing the same machine
        don't collide on disk:

        * ``base_url=https://api.example.com/v1/`` → ``…/response/api.example.com/v1``
        * ``base_url`` unset → ``…/response/default``

        Used as the per-config key for grouping cache hits in
        :class:`yggdrasil.io.response_batch.ResponseBatch`.
        """
        if self.path is not None:
            return Path(self.path)
        root = Path.home() / ".yggdrasil" / "cache" / "response"
        base_url = getattr(session, "base_url", None) if session is not None else None
        host = getattr(base_url, "host", None) if base_url is not None else None
        if not host:
            return root / "default"
        path = (getattr(base_url, "path", "") or "").strip("/")
        return root / host / path if path else root / host

    def local_cache_path(self, session: "Session | None" = None) -> str:
        """Return the local-cache root as a string and memoize it on :attr:`path`.

        Wraps :meth:`local_cache_folder` and stores the resolved
        directory back onto the frozen dataclass via
        ``object.__setattr__`` so repeat calls (per-batch lookups,
        per-response writes) don't keep recomputing the default-path
        suffix from the session's ``base_url``. The string form is
        what every downstream caller needs (Job pickle payloads,
        ``str(...)`` group keys, ``os.path.join`` callers).
        """
        if self.path is not None:
            return self.path
        resolved = str(self.local_cache_folder(session=session))
        object.__setattr__(self, "path", resolved)
        return resolved

    def prebuild(self, session: "Session | None" = None) -> "CacheConfig":
        """Materialise :attr:`path` for local-cache configs.

        After this call the session-level cache flow can reach for
        ``cfg.path`` directly without a ``local_cache_path(session)``
        dance. Symmetric to remote configs which always ship with a
        prebuilt :attr:`tabular`.

        No-op when:

        - :attr:`path` is already set;
        - the cache is remote-only (:attr:`tabular` is set);
        - :attr:`local_cache_enabled` is False (mode disables the
          cache, no ``received_*`` window, etc).

        Returns ``self`` so callers can chain
        ``cfg.prebuild(session)`` at the entry of
        :meth:`Session._send_many_batches`.
        """
        if self.path is not None:
            return self
        if self.is_remote:
            return self
        if not self.local_cache_enabled:
            return self
        self.local_cache_path(session=session)
        return self

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
        if request is None:
            return "1=1"

        request_by = self.request_by
        if not request_by:
            return "1=1"

        # Walk the precomputed columns in parallel with the keys —
        # ``make_batch_lookup_sql`` calls this once per request in a
        # 100-1000-wide batch, and ``_request_column_sql_name`` is
        # config-level (not request-level), so the per-request loop
        # shouldn't repeat the prefix lookup.
        columns = self.request_sql_columns
        match_value = request.match_value
        sql_literal = self.sql_literal
        clauses: list[str] = []
        for column, key in zip(columns, request_by):
            value = match_value(key)
            if value is None:
                clauses.append(f"{column} IS NULL")
            else:
                clauses.append(f"{column} = {sql_literal(value)}")

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

        partition_by = self._ranked_window_partition_by(identity_by)
        if partition_by is not None:
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

    def _ranked_window_partition_by(
        self,
        identity_by: Optional[Iterable[str]],
    ) -> Optional[str]:
        """Build the ``PARTITION BY`` clause for the ranked-row window.

        When *identity_by* is ``None`` the result depends only on the
        config — memoize it on ``_derived`` so a 100-wide
        ``send_many`` batch doesn't rebuild the same string per
        request. With an explicit *identity_by* the caller is opting
        out of the default; recompute fresh each time.
        """
        if identity_by is None:
            cache = self._derived_cache()
            cached = cache.get("ranked_window_partition_by", ...)
            if cached is not ...:
                return cached
            identity_cols = self.match_by
            if not identity_cols:
                cache["ranked_window_partition_by"] = None
                return None
            out = ", ".join(
                _request_column_sql_name(col)
                if col.partition(".")[0] in REQUEST_ARROW_SCHEMA.names
                else col
                for col in identity_cols
            )
            cache["ranked_window_partition_by"] = out
            return out

        identity_cols = list(identity_by)
        if not identity_cols:
            return None
        return ", ".join(
            _request_column_sql_name(col)
            if col.partition(".")[0] in REQUEST_ARROW_SCHEMA.names
            else col
            for col in identity_cols
        )

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

        partition_by = self._ranked_window_partition_by(identity_by)
        if partition_by is not None:
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
