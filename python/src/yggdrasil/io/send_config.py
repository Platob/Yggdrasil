from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Literal, Mapping, MutableMapping, Optional, TYPE_CHECKING

from yggdrasil.data.cast import any_to_datetime, any_to_timedelta
from yggdrasil.data.enums import Mode
from yggdrasil.dataclasses.waiting import WaitingConfig
from yggdrasil.environ import PyEnv
from yggdrasil.io.holder import Holder
from yggdrasil.io.path import Path
from yggdrasil.io.request import REQUEST_ARROW_SCHEMA, PreparedRequest
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from yggdrasil.io.response import Response
    from yggdrasil.io.session import Session


__all__ = ["CacheConfig", "SendConfig"]


# Module-level cached paths — avoids repeated syscalls in hot paths
# (``local_cache_folder`` is called per-request in the batch pipeline).
_DEFAULT_CACHE_ROOT: pathlib.Path = pathlib.Path.home() / ".cache" / "http" / "response"


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


class CacheConfig(_ConfigBase):
    _FIELD_NAMES: ClassVar[frozenset[str]] = _CACHE_CONFIG_FIELDS

    __slots__ = (
        "tabular", "request_by", "response_by", "mode", "anonymize",
        "received_from", "received_to", "wait",
        "mirror_local_to_remote", "cleanup_ttl", "_derived",
    )

    __setattr__ = object.__setattr__
    __delattr__ = object.__delattr__

    def __init__(
        self,
        tabular: Optional[Holder] = None,
        request_by: Optional[list[str]] = None,
        response_by: Optional[list[str]] = None,
        mode: Mode = Mode.APPEND,
        anonymize: Literal["remove", "redact"] = "remove",
        received_from: Optional[dt.datetime] = None,
        received_to: Optional[dt.datetime] = None,
        wait: WaitingConfig = False,
        mirror_local_to_remote: bool = False,
        cleanup_ttl: Optional[dt.timedelta] = dt.timedelta(days=1),
    ):
        self.mode = Mode.from_(mode, default=Mode.APPEND)
        self.wait = WaitingConfig.from_(wait)
        self.anonymize = anonymize
        self.mirror_local_to_remote = mirror_local_to_remote
        self.cleanup_ttl = cleanup_ttl
        self._derived = None

        self.request_by = _validate_request_by(request_by)
        self.response_by = _validate_response_by(response_by)

        self.received_from = _coerce_optional_datetime(received_from)
        self.received_to = _coerce_optional_datetime(received_to)

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
        if self.request_by and self.request_by != list(_DEFAULT_REQUEST_BY):
            parts.append(f"request_by={self.request_by!r}")
        if self.response_by:
            parts.append(f"response_by={self.response_by!r}")
        if self.received_from is not None:
            parts.append(f"received_from={self.received_from!r}")
        if self.received_to is not None:
            parts.append(f"received_to={self.received_to!r}")
        return f"CacheConfig({', '.join(parts)})"

    def __eq__(self, other):
        if not isinstance(other, CacheConfig):
            return NotImplemented
        return (
            self.mode == other.mode
            and self.anonymize == other.anonymize
            and self.received_from == other.received_from
            and self.received_to == other.received_to
            and self.wait == other.wait
            and self.mirror_local_to_remote == other.mirror_local_to_remote
            and self.cleanup_ttl == other.cleanup_ttl
        )

    def __hash__(self):
        return hash((
            self.mode, self.anonymize,
            self.received_from, self.received_to,
            self.wait, self.mirror_local_to_remote, self.cleanup_ttl,
        ))

    @staticmethod
    def _check_mapping(values: MutableMapping[str, Any]):
        wait = values.get("wait")
        if wait is not None:
            values["wait"] = WaitingConfig.from_(wait)

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

    def __getstate__(self):
        return {
            "mode": self.mode,
            "wait": self.wait,
            "tabular": self.tabular,
            "request_by": self.request_by,
            "response_by": self.response_by,
            "received_from": self.received_from,
            "received_to": self.received_to,
            "anonymize": self.anonymize,
            "mirror_local_to_remote": self.mirror_local_to_remote,
            "cleanup_ttl": self.cleanup_ttl,
        }

    def __setstate__(self, state):
        self.mode = state["mode"]
        self.wait = state["wait"]
        self.request_by = state["request_by"]
        self.response_by = state["response_by"]
        self.received_from = state["received_from"]
        self.received_to = state["received_to"]
        tabular = state.get("tabular")
        if tabular is None:
            tabular_url = state.get("tabular_url", state.get("path"))
            if tabular_url is not None:
                from yggdrasil.io.nested.folder_path import FolderPath
                tabular = FolderPath(path=Path.from_(tabular_url))
        self.tabular = tabular
        self.anonymize = state.get("anonymize", "remove")
        self.mirror_local_to_remote = state.get("mirror_local_to_remote", False)
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
        self.tabular = tabular
        return tabular

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
        from yggdrasil.execution.expr import all_of, col

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
        """Build the response-side match predicate as an :class:`Expression`.

        Carries the response-side match keys (when *response* is
        supplied). Returns ``None`` when no clauses apply so callers
        can compose with :func:`all_of` cleanly.

        ``received_from`` / ``received_to`` are NOT included — they
        are staleness checks applied post-read by
        :meth:`filter_response`, not identity filters. Baking them
        into the predicate would reject backfilled rows whose
        original ``received_at`` falls outside the window.
        """
        if response is None:
            return None
        from yggdrasil.execution.expr import all_of, col

        clauses: list[Any] = []
        for key, value in self.response_values(response).items():
            clauses.append(
                col(key).is_null() if value is None else col(key) == value
            )
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
        from yggdrasil.execution.expr import all_of, col

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
        ``(req1_match) OR (req2_match) OR …``. Returns ``None``
        when the batch is empty.

        Drives both backends through :meth:`Tabular.read_arrow_batches`:
        :class:`FolderPath` lets :meth:`iter_children` probe candidate
        ``partition_key=<v>/`` sub-folders directly (one ``stat``
        per accepted value, no ``iterdir`` over the full tree) and
        :meth:`Predicate.filter_arrow_batches` keeps the matching
        rows on the read side; remote Tabular backends translate the
        same predicate into their engine's native filter.
        """
        from yggdrasil.execution.expr import (
            all_of,
            any_of,
            col,
        )

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


@dataclass(frozen=True, slots=True)
class SendConfig(_ConfigBase):
    _FIELD_NAMES: ClassVar[frozenset[str]] = _SEND_CONFIG_FIELDS

    raise_error: bool = True
    wait: WaitingConfig | None = None
    remote_cache: CacheConfig | None = None
    local_cache: CacheConfig | None = None
    cache_only: bool = False
    spark_session: Optional["SparkSession"] = field(
        default=None,
        hash=False,
        compare=False,
        repr=False,
    )

    def __post_init__(self):
        w = self.wait
        if w is not None:
            object.__setattr__(self, "wait", WaitingConfig.from_(w))
        rc = self.remote_cache
        if rc is not None:
            object.__setattr__(self, "remote_cache", CacheConfig.from_(rc))
        lc = self.local_cache
        if lc is not None:
            object.__setattr__(self, "local_cache", CacheConfig.from_(lc))
        spark = self.spark_session
        if spark is True or spark is ...:
            spark = PyEnv.spark_session()
        object.__setattr__(self, "spark_session", spark)

    def __getstate__(self):
        return {
            "raise_error": self.raise_error,
            "wait": self.wait,
            "remote_cache": self.remote_cache,
            "local_cache": self.local_cache,
            "cache_only": self.cache_only,
            "spark_session": None,
        }

    def __setstate__(self, state):
        object.__setattr__(self, "raise_error", state.get("raise_error", True))
        object.__setattr__(self, "wait", state.get("wait"))
        object.__setattr__(self, "remote_cache", state.get("remote_cache"))
        object.__setattr__(self, "local_cache", state.get("local_cache"))
        object.__setattr__(self, "cache_only", state.get("cache_only", False))
        object.__setattr__(self, "spark_session", None)

    @classmethod
    def from_(
        cls,
        arg: "SendConfig | Mapping[str, Any] | None",
        *,
        default: Any = ...,
        **overrides: Any,
    ) -> "SendConfig":
        try:
            if arg is None:
                if not overrides:
                    return default if default is not ... else cls.default()
                if cls._matches_default(overrides):
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
        except (TypeError, ValueError):
            if default is ...:
                raise
            return default


