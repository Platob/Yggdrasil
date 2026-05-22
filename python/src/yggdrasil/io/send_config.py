from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Literal, Mapping, MutableMapping, Optional, TYPE_CHECKING

from yggdrasil.data.cast import any_to_datetime, any_to_timedelta
from yggdrasil.dataclasses import DEFAULT_WAITING_CONFIG
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.data.enums import Mode
from yggdrasil.environ import PyEnv
from yggdrasil.io.path import Path
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
        "cache_only",
        "spark_session",
        "as_tabular",
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
    """Cache-table column name for a request-side ``request_by`` key.

    The response cache stores requests as flattened ``request_<col>``
    columns (cf. :data:`RESPONSE_SCHEMA`), so a user-supplied
    ``request_by`` key that names a bare request column
    (``public_url_hash``, ``method`` …) needs the ``request_``
    prefix when used as a column reference (predicate match clause,
    :meth:`Tabular.write_arrow_batches` match-by). Already-prefixed
    keys pass through.
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
        # Per-class singleton — ``Session.send`` constructs one of these
        # per call as a final fallback (every argument either None or
        # equal to the field default), so we'd rather hand back the
        # cached instance than rebuild + ``__post_init__``-normalize a
        # fresh frozen dataclass on every send. Stamp on ``cls.__dict__``
        # directly (not ``setattr``) so subclasses get their own slot
        # instead of inheriting the parent's singleton.
        inst = cls.__dict__.get("_DEFAULT_INSTANCE")
        if inst is None:
            inst = cls()
            type.__setattr__(cls, "_DEFAULT_INSTANCE", inst)
        return inst

    @classmethod
    def parse_mapping(cls, options: Mapping[str, Any], **overrides: Any):
        if not isinstance(options, Mapping):
            raise TypeError(
                f"{cls.__name__}.parse_mapping expects a Mapping, "
                f"got {type(options).__name__!r}"
            )
        values = {k: v for k, v in options.items() if k in cls._FIELD_NAMES}
        values.update(overrides)
        if cls._matches_default(values):
            return cls.default()
        return cls(**cls._check_mapping(values))

    @classmethod
    def _matches_default(cls, values: Mapping[str, Any]) -> bool:
        """``True`` when every value is None or already equal to the
        field default — i.e. the resulting instance would be
        value-equal to :meth:`default`.

        Lets ``from_`` / ``parse_mapping`` skip the constructor +
        ``__post_init__`` round trip on the steady-state shape that
        ``Session.send`` produces (every kwarg either ``None`` or the
        field default). Any non-None override that diverges from
        defaults — or an unknown key — falls through to the full
        constructor.

        Identity match (``is``) is required for non-primitive types
        because :class:`CacheConfig` (and friends) declare
        ``path`` / ``tabular`` / ``request_by`` with
        ``compare=False`` — so two CacheConfigs that differ only on
        an excluded field compare equal under ``==`` and would
        otherwise be silently collapsed back to the default.
        """
        if not values:
            return True
        default = cls.default()
        for k, v in values.items():
            if v is None:
                # ``None`` means "not supplied" — ``_check_mapping``
                # drops it before the constructor sees it, so it
                # can't shift the result off-default.
                continue
            default_v = getattr(default, k, ...)
            if default_v is v:
                continue
            # Value-equality only for primitive built-ins where
            # ``==`` and identity-of-interest agree. Custom dataclasses
            # with ``compare=False`` fields are excluded above.
            if type(v) in (bool, int, float, str, bytes) and default_v == v:
                continue
            return False
        return True

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
            values["remote_cache"] = CacheConfig.from_(remote_cache)

        local_cache = values.get("local_cache")
        if local_cache is not None:
            values["local_cache"] = CacheConfig.from_(local_cache)

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

    # Cache backend — any :class:`Tabular` subclass:
    #
    #   - :class:`~yggdrasil.io.nested.folder_path.FolderPath` for the
    #     on-disk fast-path local cache (string / pathlib /
    #     :class:`~yggdrasil.io.path.Path` constructor arguments are
    #     auto-coerced to a :class:`FolderPath` by :meth:`from_`);
    #   - :class:`~yggdrasil.databricks.table.table.Table` (or any
    #     third-party adapter exposing ``read_arrow_batches`` /
    #     ``write_arrow_batches``) for the remote / network-backed
    #     cache.
    #
    # The Session-level read / write helpers
    # (:meth:`Session._load_cached_response`,
    # :meth:`Session._lookup_cached`,
    # :meth:`Session._store_cached_response`,
    # :func:`Session._insert_cache`) dispatch entirely through this
    # one slot — local vs. remote is determined by the backend type,
    # not by a parallel ``path`` field. :meth:`__getstate__` projects
    # a local FolderPath down to its URL string so the frozen-
    # dataclass instance survives Spark / multiprocessing /
    # Power Query worker boundaries without dragging bound backend
    # handles; remote backends are dropped on pickle and the
    # receiver rebuilds them from session-level config.
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
    # Default 1 day. Set to ``None`` to disable cleanup entirely
    # (cache grows unbounded).
    cleanup_ttl: Optional[dt.timedelta] = dt.timedelta(days=1)
    # Lazy memo for derived predicate-pipeline values
    # (``cache_enabled``, ``match_by``, ``match_by_columns``,
    # ``request_match_columns``, ``request_by_is_public``). Each is
    # purely a function of the frozen fields above — caching turns
    # a per-send (or per-batch-request) recomputation into a dict
    # lookup. Excluded from pickle / equality / repr; rebuilt
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

        # Path-shaped inputs to ``tabular`` are local-cache sugar:
        # ``CacheConfig(tabular="/cache")`` builds a FolderPath under
        # the hood so callers don't have to import the class. Live
        # Tabular instances (FolderPath, Databricks Table, third-
        # party adapters) pass through unchanged — recognised by the
        # ``read_arrow_batches`` / ``write_arrow_batches`` duck-test
        # so test doubles that only stub the bits a particular path
        # exercises (``_StubTabular.full_name``, …) still flow
        # through without being mis-coerced to a FolderPath.
        tab = self.tabular
        if isinstance(tab, (Path, pathlib.PurePath, str)):
            from yggdrasil.io.nested.folder_path import FolderPath
            object.__setattr__(
                self, "tabular", FolderPath(path=Path.from_(tab)),
            )

    def __getstate__(self):
        # Project local FolderPath caches down to their URL string for
        # transport — keeps the payload free of bound backend handles
        # (Databricks client on a :class:`VolumePath`, boto3 client on
        # an :class:`S3Path`, …) so the dataclass survives Spark /
        # multiprocessing / Power Query worker boundaries without
        # dragging unpicklable live state. :meth:`__setstate__`
        # rehydrates by rebuilding the FolderPath, which is
        # Singleton-cached so the receiving side coalesces onto the
        # same live instance as any other config pointing there.
        #
        # Remote :class:`Tabular` backends (Databricks Table, …) are
        # dropped on pickle — they wrap live SDK clients that don't
        # cross process boundaries; the receiver rebuilds them from
        # session-level config.
        local_url = self._local_cache_url()
        return {
            "mode": self.mode,
            "wait": self.wait,
            "tabular_url": local_url,
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
        # Rehydrate local FolderPath from its URL string; remote
        # backends are reattached out-of-band by the receiver.
        # Accept the legacy ``path`` key from snapshots taken before
        # the unification so old pickled payloads still load.
        tabular_url = state.get("tabular_url", state.get("path"))
        if tabular_url is not None:
            from yggdrasil.io.nested.folder_path import FolderPath
            object.__setattr__(
                self, "tabular", FolderPath(path=Path.from_(tabular_url)),
            )
        else:
            object.__setattr__(self, "tabular", None)
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

    def _local_cache_url(self) -> "str | None":
        """URL string of the local cache root, or ``None`` for non-local.

        Local cache (FolderPath) carries an addressable path; remote
        backends are dropped from the pickle wire format.
        """
        tab = self.tabular
        if tab is None:
            return None
        try:
            from yggdrasil.io.nested.folder_path import FolderPath
        except ImportError:
            return None
        if isinstance(tab, FolderPath):
            return str(tab.path.url)
        return None

    @classmethod
    def from_(
        cls,
        arg: "CacheConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "CacheConfig":
        if arg is None:
            # Don't reuse :meth:`_ConfigBase._matches_default` here:
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
            # Bare number → TTL in seconds. Same shape as the timedelta
            # branch; the ``bool`` exclusion mirrors
            # :func:`any_to_timedelta`, which refuses to treat a bool as
            # a seconds value.
            isinstance(arg, (int, float)) and not isinstance(arg, bool)
        ):
            ttl = arg if isinstance(arg, dt.timedelta) else any_to_timedelta(arg)
            overrides["received_ttl"] = ttl

            # fill received_from and received_to if not exists
            received_to = overrides.get("received_to")
            received_to = dt.datetime.now(dt.timezone.utc) if received_to is None else any_to_datetime(received_to)
            overrides["received_to"] = received_to

            received_from = overrides.get("received_from")
            if not received_from:
                overrides["received_from"] = received_to - ttl

        else:
            # Non-temporal arg → cache backend. ``Holder.from_`` is the
            # canonical IO dispatch parser: bytes → :class:`Memory`,
            # path-like str / ``pathlib`` / :class:`URL` → the
            # scheme-matched storage class, an already-built IO /
            # :class:`Tabular` (FolderPath, Databricks Table, …) →
            # identity passthrough. ``__post_init__`` still wraps a
            # bare :class:`Path` into a :class:`FolderPath` so the
            # stored ``tabular`` is always a live :class:`Tabular`.
            from yggdrasil.io.holder import Holder
            overrides["tabular"] = Holder.from_(arg)

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
        """True when the bound :attr:`tabular` is a :class:`FolderPath`.

        The local-cache fast path keys on this — it's the on-disk
        backend that supports the partitioned write directly. Lazy
        import keeps the property cheap for the no-cache default.
        """
        tab = self.tabular
        if tab is None:
            return False
        from yggdrasil.io.nested.folder_path import FolderPath
        return isinstance(tab, FolderPath)

    @property
    def is_remote(self) -> bool:
        """True when the bound :attr:`tabular` is a remote backend.

        Any :class:`Tabular` that isn't a :class:`FolderPath` —
        Databricks Table, third-party adapters — drives the remote
        cache pipeline (predicate→SQL translation, MERGE writes,
        spark-frame dispatch).
        """
        return self.tabular is not None and not self.is_local

    @property
    def local_cache_enabled(self):
        # Two ways to opt into a local cache layer:
        #   1) ``tabular`` is a :class:`FolderPath` (explicit local
        #      backend, either constructor-supplied or sugar-coerced
        #      from a path-shaped argument);
        #   2) a ``received_from`` / ``received_to`` window is set,
        #      in which case :meth:`cache_tabular` lazy-fills the
        #      default ``~/.yggdrasil/cache/response/...`` FolderPath.
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
    def match_by_columns(self) -> list[str]:
        """Flattened column names for keyed cache operations.

        Request-side keys are stored on the response table under the
        flattened ``request_<col>`` form (cf. :data:`RESPONSE_SCHEMA`),
        so a bare ``public_url_hash`` / ``method`` / etc. needs the
        ``request_`` prefix before it can be referenced as a target
        column in :meth:`Tabular.insert(match_by=...)` /
        :meth:`Tabular.write_arrow_batches`. Response-side keys map
        1:1 already.
        """
        cache = self._derived_cache()
        out = cache.get("match_by_columns", ...)
        if out is ...:
            out = [
                *(_request_column_sql_name(k) for k in (self.request_by or ())),
                *(self.response_by or ()),
            ]
            cache["match_by_columns"] = out
        return out

    @property
    def request_match_columns(self) -> list[str]:
        """Flattened column names for every ``request_by`` key.

        :meth:`make_batch_lookup_predicate` walks N requests and
        emits the same column name for each (it depends only on
        the config), so precomputing the list cuts the per-request
        loop's cost — at ``BATCH_SIZE=64`` requests that is 64
        ``_request_column_sql_name`` calls replaced by one shared
        list of strings.
        """
        cache = self._derived_cache()
        out = cache.get("request_match_columns", ...)
        if out is ...:
            out = [_request_column_sql_name(k) for k in (self.request_by or ())]
            cache["request_match_columns"] = out
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
        pass before computing ``request_tuple`` / building the lookup
        :class:`Predicate` — the saving is one URL parse + one header
        normalize per request per lookup, which adds up on send_many
        bursts.
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

    def local_cache_folder(self, session: "Session | None" = None) -> Path:
        """Backend-agnostic root for the local cache.

        Returns the bound :class:`FolderPath`'s :attr:`path` when
        :attr:`tabular` is local (any :class:`yggdrasil.io.path.Path`
        subclass — LocalPath on disk, VolumePath on a Databricks
        Volume, S3Path on a bucket, …); otherwise builds the default
        LocalPath under ``~/.yggdrasil/cache/response``, suffixed
        with the session's ``base_url`` host + path when one is
        available so different APIs sharing the same machine don't
        collide on disk:

        * ``base_url=https://api.example.com/v1/`` → ``…/response/api.example.com/v1``
        * ``base_url`` unset → ``…/response/default``

        Used as the per-config key for grouping cache hits in
        :class:`yggdrasil.io.response_batch.ResponseBatch`.
        """
        if self.is_local:
            return self.tabular.path
        root = pathlib.Path.home() / ".yggdrasil" / "cache" / "response"
        base_url = getattr(session, "base_url", None) if session is not None else None
        host = getattr(base_url, "host", None) if base_url is not None else None
        if not host:
            folder = root / "default"
        else:
            url_path = (getattr(base_url, "path", "") or "").strip("/")
            folder = root / host / url_path if url_path else root / host
        return Path.from_(folder)

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
        object.__setattr__(self, "tabular", tabular)
        return tabular

    def prebuild(self, session: "Session | None" = None) -> "CacheConfig":
        """Materialise :attr:`tabular` for local-cache configs.

        After this call the session-level cache flow can reach for
        ``cfg.tabular`` directly without a ``cache_tabular(session)``
        dance. Symmetric to remote configs which always ship with a
        prebuilt :attr:`tabular`.

        No-op when :attr:`tabular` is already set or
        :attr:`local_cache_enabled` is False (mode disables the
        cache, no ``received_*`` window, etc).

        Returns ``self`` so callers can chain
        ``cfg.prebuild(session)`` at the entry of
        :meth:`Session._send_many_batches`.
        """
        if self.tabular is not None:
            return self
        if not self.local_cache_enabled:
            return self
        self.cache_tabular(session=session)
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

    def request_predicate(
        self,
        request: PreparedRequest | None,
    ) -> "Any | None":
        """Build the per-request match predicate as an :class:`Expression`.

        Walks ``request_by`` keys, builds ``col(request_<key>) ==
        request.match_value(key)`` per entry, and ANDs them together.
        ``None`` request → ``None`` (no per-request constraint).
        """
        if request is None or not self.request_by:
            return None
        from yggdrasil.io.tabular.execution.expr import all_of, col

        clauses: list[Any] = []
        match_value = request.match_value
        for column, key in zip(self.request_match_columns, self.request_by):
            value = match_value(key)
            clauses.append(
                col(column).is_null() if value is None else col(column) == value
            )
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return all_of(*clauses)

    def response_predicate(
        self,
        response: "Response | None" = None,
    ) -> "Any | None":
        """Build the response-side / time-window predicate as an :class:`Expression`.

        Carries the response-side match keys (when *response* is
        supplied) plus the configured ``received_*`` window. Returns
        ``None`` when no clauses apply so callers can compose with
        :func:`all_of` cleanly.
        """
        from yggdrasil.io.tabular.execution.expr import all_of, col

        clauses: list[Any] = []
        if response is not None:
            for key, value in self.response_values(response).items():
                clauses.append(
                    col(key).is_null() if value is None else col(key) == value
                )
        if self.received_from is not None:
            clauses.append(col("received_at") >= self.received_from)
        if self.received_to is not None:
            clauses.append(col("received_at") < self.received_to)
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return all_of(*clauses)

    def make_lookup_predicate(
        self,
        request: PreparedRequest | None = None,
        response: "Response | None" = None,
    ) -> "Any | None":
        """Single-request :class:`Predicate` for the cache read.

        Shape: ``partition_key == <req.partition_key>`` AND the
        per-request match clause AND the response/time-window
        clause. Returns ``None`` when no clauses apply (an
        unconstrained :class:`Tabular` read).

        The same predicate drives both backends: a :class:`FolderPath`
        consumes it via :meth:`Predicate.filter_arrow_batches` (with
        ``extract_partition_filters`` short-circuiting the
        ``<col>=<val>/`` listing), and a remote
        :class:`Tabular` (Databricks Table, …) translates it to its
        engine's native filter (SQL ``WHERE``) inside
        :meth:`Tabular.read_arrow_batches`.
        """
        from yggdrasil.io.tabular.execution.expr import all_of, col

        clauses: list[Any] = []
        if request is not None:
            clauses.append(col("partition_key") == request.partition_key)
        req_pred = self.request_predicate(request)
        if req_pred is not None:
            clauses.append(req_pred)
        resp_pred = self.response_predicate(response)
        if resp_pred is not None:
            clauses.append(resp_pred)
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
        ``(req1_match) OR (req2_match) OR …`` AND the
        response/time-window clause. Returns ``None`` when the
        batch is empty and no time window applies.

        Drives both backends through :meth:`Tabular.read_arrow_batches`:
        :class:`FolderPath` lets :meth:`iter_children` probe candidate
        ``partition_key=<v>/`` sub-folders directly (one ``stat``
        per accepted value, no ``iterdir`` over the full tree) and
        :meth:`Predicate.filter_arrow_batches` keeps the matching
        rows on the read side; remote Tabular backends translate the
        same predicate into their engine's native filter.
        """
        from yggdrasil.io.tabular.execution.expr import all_of, any_of, col

        request_list = list(requests)
        clauses: list[Any] = []

        if request_list:
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

        resp_pred = self.response_predicate(None)
        if resp_pred is not None:
            clauses.append(resp_pred)

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return all_of(*clauses)

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
    # When True, ``Session._send`` consults the local + remote caches as
    # usual but skips the network fallback — a full miss raises
    # :class:`LookupError` instead of crossing the wire. ``send_many``
    # silently drops misses from the stream. Lets callers replay a known
    # warm cache offline (or after an outage) without an unintended
    # upstream fetch.
    cache_only: bool = False
    # ``True`` flips the public ``Session.send_many`` return type from
    # ``Iterator[Response]`` to a single concatenated :class:`Tabular`
    # (an :class:`ArrowTabular` in Python mode, a :class:`Dataset`
    # wrapping a Spark frame when a session is bound). Single-request
    # ``Session.send`` ignores this flag — it always returns
    # :class:`Response`.
    as_tabular: bool = False
    spark_session: Optional["SparkSession"] = field(
        default=None,
        hash=False,
        compare=False,
        repr=False,
    )

    def __post_init__(self):
        object.__setattr__(self, "wait", WaitingConfig.from_(self.wait))
        object.__setattr__(self, "remote_cache", CacheConfig.from_(self.remote_cache))
        object.__setattr__(self, "local_cache", CacheConfig.from_(self.local_cache))
        # ``True`` / ``...`` → resolve the live SparkSession via
        # :meth:`PyEnv.spark_session` so callers don't have to thread
        # one through every layer. ``None`` stays ``None`` (Python mode).
        spark = self.spark_session
        if spark is True or spark is ...:
            spark = PyEnv.spark_session()
        object.__setattr__(self, "spark_session", spark)

    def __getstate__(self):
        return {
            "raise_error": self.raise_error,
            "stream": self.stream,
            "wait": self.wait,
            "remote_cache": self.remote_cache,
            "local_cache": self.local_cache,
            "cache_only": self.cache_only,
            "as_tabular": self.as_tabular,
            "spark_session": None,
        }

    def __setstate__(self, state):
        object.__setattr__(self, "raise_error", state["raise_error"])
        object.__setattr__(self, "stream", state["stream"])
        object.__setattr__(self, "wait", state["wait"])
        object.__setattr__(self, "remote_cache", state["remote_cache"])
        object.__setattr__(self, "local_cache", state["local_cache"])
        object.__setattr__(self, "cache_only", state.get("cache_only", False))
        object.__setattr__(self, "as_tabular", state.get("as_tabular", False))
        object.__setattr__(self, "spark_session", None)

    @classmethod
    def from_(
        cls,
        arg: "SendConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "SendConfig":
        if arg is None:
            if not overrides or cls._matches_default(overrides):
                return cls.default()
            return cls.parse_mapping(overrides)
        if isinstance(arg, cls):
            return arg.merge(**overrides) if overrides else arg
        if isinstance(arg, Mapping):
            return cls.parse_mapping(arg, **overrides)
        raise TypeError(
            f"{cls.__name__}.from_ expects a {cls.__name__}, Mapping, or None; "
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
    cache_only: bool = False
    # See :class:`SendConfig.as_tabular` — controls whether
    # ``Session.send_many`` returns an ``Iterator[Response]`` (False)
    # or one concatenated :class:`Tabular` (True).
    as_tabular: bool = False
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
        object.__setattr__(self, "remote_cache", CacheConfig.from_(self.remote_cache))
        object.__setattr__(self, "local_cache", CacheConfig.from_(self.local_cache))
        spark = self.spark_session
        if spark is True or spark is ...:
            spark = PyEnv.spark_session()
        object.__setattr__(self, "spark_session", spark)

    def __getstate__(self):
        return {
            "wait": self.wait,
            "raise_error": self.raise_error,
            "stream": self.stream,
            "remote_cache": self.remote_cache,
            "local_cache": self.local_cache,
            "cache_only": self.cache_only,
            "as_tabular": self.as_tabular,
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
        object.__setattr__(self, "cache_only", state.get("cache_only", False))
        object.__setattr__(self, "as_tabular", state.get("as_tabular", False))
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
    def from_(
        cls,
        arg: "SendManyConfig | SendConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "SendManyConfig":
        if arg is None:
            if not overrides or cls._matches_default(overrides):
                return cls.default()
            return cls(**overrides)
        if isinstance(arg, cls):
            return arg.merge(**overrides) if overrides else arg
        if isinstance(arg, SendConfig):
            base = {
                "wait": arg.wait,
                "raise_error": arg.raise_error,
                "stream": arg.stream,
                "remote_cache": arg.remote_cache,
                "local_cache": arg.local_cache,
                "cache_only": arg.cache_only,
                "as_tabular": arg.as_tabular,
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
            f"{cls.__name__}.from_ expects a {cls.__name__}, SendConfig, "
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
            cache_only=self.cache_only,
            as_tabular=self.as_tabular,
            spark_session=self.spark_session if with_spark else None,
        )
