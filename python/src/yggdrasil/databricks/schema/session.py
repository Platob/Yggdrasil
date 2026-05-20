"""HTTP session that uses a Unity Catalog schema as its remote response cache.

:class:`SchemaSession` is a thin :class:`HTTPSession` subclass that maps
every outbound request's URL path to one Delta table inside a bound
:class:`Schema`. The parent's ``_send`` pipeline already owns the local
cache → remote cache → network → writeback flow; this subclass's only
job is to stamp the per-path :class:`CacheConfig` onto the request so
that pipeline knows *which* table backs the remote tier.

Two cache modes (``mode`` constructor arg, default :attr:`Mode.APPEND`):

* :attr:`Mode.APPEND` — read-through, write-once. First call against a
  given URL fetches from the wire and appends the response to the
  per-path table; subsequent calls return the cached row without
  hitting the network.
* :attr:`Mode.UPSERT` — bypass the read. Every call goes to the wire
  and replaces the row on the per-path table. Useful when the upstream
  state may have changed and the cache needs to be repaired.

Local on-disk cache (``local_cache`` constructor arg, default ``True``)
runs in front of the remote tier — a single Arrow IPC file per response
under ``~/.yggdrasil/cache/response/<host>/<path>/<hash>.arrow``. Pass
``False`` to disable it (remote-only) or a path / :class:`CacheConfig`
to override the location.

Time-window normalisation (``time_window`` constructor arg, default
``None``) snaps start / end query parameters onto a canonical grid
*before* the cache key is computed. Without it, a caller asking for
``[10:23, 10:38]`` then ``[10:25, 10:40]`` misses the cache every
time even though the data is the same; with :class:`TimeWindowPolicy`
both calls collapse to the same snapped URL and the second one hits
the Delta cache. See :class:`TimeWindowPolicy` for the knobs.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Union

from yggdrasil.data.cast.datetime import truncate_datetime
from yggdrasil.dataclasses import ExpiringDict
from yggdrasil.dataclasses.waiting import DEFAULT_WAITING_CONFIG, WaitingConfig
from yggdrasil.data.enums import Mode
from yggdrasil.data.enums.mode import ModeLike
from yggdrasil.databricks.table.table import Table
from yggdrasil.io.http_.session import HTTPSession
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.send_config import CacheConfig
from yggdrasil.io.url import URL

if TYPE_CHECKING:
    from yggdrasil.databricks.schema.schema import Schema
    from yggdrasil.io.authorization.base import Authorization
    from yggdrasil.io.headers import Headers


__all__ = ["SchemaSession", "TimeWindowPolicy"]


LOGGER = logging.getLogger(__name__)

# Per-path Table handle cache TTL. One hour amortises the schema lookup
# across a typical job while still surfacing upstream renames / drops
# on the next refresh.
_DEFAULT_TABLE_CACHE_TTL: dt.timedelta = dt.timedelta(hours=1)


# Sentinel — used in :meth:`TimeWindowPolicy.__post_init__` to validate
# the configured ``granularity`` at construction time by running it
# through :func:`truncate_datetime` with a known-good anchor. The anchor
# itself is irrelevant; only the parse + truncate steps need to succeed.
_GRANULARITY_PROBE: dt.datetime = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)


@dataclass(frozen=True)
class TimeWindowPolicy:
    """Snap time-range query parameters onto a canonical grid.

    External time-windowed APIs (price history, telemetry, ledger
    entries) accept arbitrary start / end timestamps but the per-URL
    cache key is only useful when repeated calls produce the *same*
    URL. A caller asking for ``[10:23, 10:38]`` one minute and
    ``[10:25, 10:40]`` the next misses the cache every time even
    though the data they need is identical.

    The policy walks the URL's query string, snaps every value whose
    key appears in :attr:`start_keys` / :attr:`end_keys` onto
    :attr:`granularity` — floor the start, ceil the end when
    :attr:`expand` is ``True`` — and rebuilds the query in canonical
    sorted order. Two calls whose requested windows fall inside the
    same snapped grid produce identical URLs, identical
    :attr:`PreparedRequest.public_url_hash` cache keys, and therefore
    the second call hits the Delta-table cache without crossing the
    wire.

    Args:
        granularity: Grid the window snaps onto. Same input as
            :func:`yggdrasil.data.cast.datetime.truncate_datetime`:
            ISO 8601 duration strings (``"PT1H"``, ``"PT15M"``,
            ``"P1D"``) or :class:`datetime.timedelta`. Calendar
            intervals (``"P1M"``, ``"P1Y"``) snap to month / year
            boundaries.
        start_keys: Query parameter names treated as window starts.
            Matched case-sensitively. Defaults cover the common snake
            and camel spellings (``start``, ``from``, ``start_time``,
            ``startTime``, ``period_start``, ``start_date``,
            ``startDate``).
        end_keys: Query parameter names treated as window ends.
            Defaults mirror :attr:`start_keys`.
        expand: When ``True`` (default), end values ceil up to the
            next boundary so the snapped window fully covers the
            requested one. When ``False``, both ends floor — tighter
            cache keys but the returned data may miss the tail of the
            original request.
        output_format: ``strftime`` format applied to snapped
            datetimes before writing them back to the URL. Defaults
            to RFC 3339 UTC (``"%Y-%m-%dT%H:%M:%SZ"``). Override when
            the upstream API needs epoch seconds, date-only, etc.

    Example::

        from yggdrasil.databricks.schema import SchemaSession, TimeWindowPolicy

        session = SchemaSession(
            schema=schema,
            base_url="https://api.example.com",
            time_window=TimeWindowPolicy(granularity="PT1H"),
        )
        # Both calls become GET /prices?end=2026-05-20T11:00:00Z&start=2026-05-20T10:00:00Z
        session.send(make_request("/prices?start=2026-05-20T10:23:14Z&end=2026-05-20T10:38:02Z"))
        session.send(make_request("/prices?start=2026-05-20T10:11:05Z&end=2026-05-20T10:55:39Z"))
        # Second call hits the cache.
    """

    granularity: Union[str, dt.timedelta]
    start_keys: tuple[str, ...] = (
        "start", "from", "start_time", "startTime",
        "period_start", "start_date", "startDate",
    )
    end_keys: tuple[str, ...] = (
        "end", "to", "end_time", "endTime",
        "period_end", "end_date", "endDate",
    )
    expand: bool = True
    output_format: str = "%Y-%m-%dT%H:%M:%SZ"

    def __post_init__(self) -> None:
        if not (self.start_keys or self.end_keys):
            raise ValueError(
                "TimeWindowPolicy: at least one of start_keys / end_keys "
                "must be non-empty; otherwise the policy is a no-op. "
                "Pass the query parameter names your upstream API uses "
                "for time-window boundaries."
            )
        try:
            truncate_datetime(_GRANULARITY_PROBE, self.granularity)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"TimeWindowPolicy: granularity={self.granularity!r} is not a "
                "parseable interval. Use an ISO 8601 duration "
                "('PT1H', 'PT15M', 'P1D', 'P1M') or a datetime.timedelta. "
                f"Underlying error: {exc}"
            ) from exc

    @classmethod
    def from_(
        cls,
        value: "TimeWindowPolicy | str | dt.timedelta | None",
    ) -> "TimeWindowPolicy | None":
        """Coerce a constructor argument into a :class:`TimeWindowPolicy`.

        - ``None`` returns ``None`` (policy disabled).
        - A :class:`TimeWindowPolicy` is returned unchanged.
        - A ``str`` / :class:`datetime.timedelta` is treated as the
          ``granularity`` and a policy with default key sets is built.
        """
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, (str, dt.timedelta)):
            return cls(granularity=value)
        raise TypeError(
            f"TimeWindowPolicy.from_: cannot coerce {type(value).__name__} "
            "into a TimeWindowPolicy. Pass a TimeWindowPolicy, an ISO duration "
            "string ('PT1H', 'P1D'), a datetime.timedelta, or None."
        )

    def apply(self, url: URL) -> URL:
        """Return *url* with its time-window query params snapped to grid.

        Returns the input URL unchanged when nothing matched (no query
        string, or none of the configured keys appear in it). Otherwise
        returns a new :class:`URL` whose query is rebuilt with the
        snapped values in canonical sorted order.

        Raises:
            ValueError: A value bound to a matching key is not a
                parseable datetime. The fix is to drop the key from
                ``start_keys`` / ``end_keys`` or supply an
                ISO-8601-compatible value.
        """
        items = url.query_items(keep_blank_values=True)
        if not items:
            return url

        start_set = frozenset(self.start_keys)
        end_set = frozenset(self.end_keys)

        new_items: list[tuple[str, str]] = []
        changed = False
        for key, value in items:
            if key in start_set:
                snapped = self._snap(key, value, ceil=False)
            elif key in end_set:
                snapped = self._snap(key, value, ceil=self.expand)
            else:
                new_items.append((key, value))
                continue
            if snapped != value:
                changed = True
            new_items.append((key, snapped))

        if not changed:
            return url
        return url.with_query_items(new_items, sort_keys=True)

    def _snap(self, key: str, value: str, *, ceil: bool) -> str:
        if not value:
            return value
        try:
            snapped = truncate_datetime(
                value,
                self.granularity,
                tz=dt.timezone.utc,
                add_interval=ceil,
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"TimeWindowPolicy: query parameter {key!r}={value!r} is not "
                f"a parseable datetime (granularity={self.granularity!r}). "
                f"Drop {key!r} from start_keys / end_keys, or send a value "
                "the datetime cast registry understands (ISO 8601, epoch "
                f"seconds). Underlying error: {exc}"
            ) from exc
        # Always emit in UTC so vendors that flip between ``+00:00`` /
        # ``Z`` and callers that mix UTC with local-time strings still
        # collapse onto one canonical cache key.
        if snapped.tzinfo is None:
            snapped = snapped.replace(tzinfo=dt.timezone.utc)
        else:
            snapped = snapped.astimezone(dt.timezone.utc)
        return snapped.strftime(self.output_format)


class SchemaSession(HTTPSession):
    """HTTP session that caches responses in per-path Delta tables.

    Args:
        schema: The :class:`Schema` whose tables back the cache.
        base_url: Forwarded to :class:`HTTPSession`; participates in
            the parent's ``(cls, base_url, key)`` singleton key.
        mode: Cache write disposition. :attr:`Mode.APPEND` (default)
            reads the cache first; :attr:`Mode.UPSERT` always fetches
            fresh and replaces the cached row.
        local_cache: ``True`` (default) enables the on-disk fast-path
            cache at the session's default folder; ``str`` / :class:`Path`
            picks an explicit directory; a :class:`CacheConfig` is used
            as-is; ``False`` / ``None`` disables the local tier.
        time_window: Snap time-range query parameters onto a canonical
            grid before the cache key is computed. Pass an ISO duration
            string (``"PT1H"``, ``"P1D"``), a :class:`datetime.timedelta`,
            or an explicit :class:`TimeWindowPolicy`. ``None`` (default)
            disables the rewrite. See :class:`TimeWindowPolicy`.
        table_cache_ttl: TTL on the in-process per-path :class:`Table`
            handle cache (1 hour by default).

    Remaining keyword arguments forward to :class:`HTTPSession`.
    """

    def __init__(
        self,
        schema: "Schema",
        base_url: Optional[URL | str] = None,
        *,
        mode: ModeLike = Mode.APPEND,
        local_cache: Union[bool, str, Path, CacheConfig, Mapping[str, Any], None] = True,
        time_window: Union[TimeWindowPolicy, str, dt.timedelta, None] = None,
        table_cache_ttl: "float | int | dt.timedelta | None" = _DEFAULT_TABLE_CACHE_TTL,
        verify: bool = True,
        pool_maxsize: int = 10,
        headers: "Headers | Mapping[str, str] | None" = None,
        waiting: WaitingConfig = DEFAULT_WAITING_CONFIG,
        auth: Optional["Authorization"] = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        if schema is None:
            raise TypeError(
                "SchemaSession requires a bound Schema; got None. Pass "
                "the Schema whose tables should back the response cache."
            )
        super().__init__(
            base_url=base_url,
            verify=verify,
            pool_maxsize=pool_maxsize,
            headers=headers,
            waiting=waiting,
            auth=auth,
        )
        self.schema = schema
        self.mode = Mode.from_(mode, default=Mode.APPEND)
        self.local_cache = local_cache
        self.time_window = TimeWindowPolicy.from_(time_window)
        self.table_cache_ttl = table_cache_ttl
        # Derived from ``local_cache`` / ``table_cache_ttl``; not part
        # of the identity key. Listed in ``_TRANSIENT_STATE_ATTRS`` so
        # the singleton-key probe and ``__getstate__`` both skip them,
        # then rebuilt lazily on access via the cached_property pair.
        self._table_cache = ExpiringDict(
            default_ttl=table_cache_ttl, max_size=1024,
        )
        self._local_cache_template = self._build_local_template(local_cache)

    _TRANSIENT_STATE_ATTRS = HTTPSession._TRANSIENT_STATE_ATTRS | {
        "_table_cache", "_local_cache_template",
    }

    def __setstate__(self, state):
        # Inherit the live-singleton short-circuit and the
        # transient-attr defaulting from :class:`HTTPSession` /
        # :class:`Session`, then rebuild the derived caches from the
        # already-restored ``local_cache`` / ``table_cache_ttl`` so a
        # cross-process clone has working caches without re-pickling
        # the live :class:`ExpiringDict` state.
        if getattr(self, "_initialized", False):
            return
        super().__setstate__(state)
        self._table_cache = ExpiringDict(
            default_ttl=self.table_cache_ttl, max_size=1024,
        )
        self._local_cache_template = self._build_local_template(self.local_cache)

    def __getnewargs_ex__(self):
        # SchemaSession's ``__init__`` takes ``schema`` + ``base_url``
        # positionally; every other argument is keyword-only. Reuse
        # the parent's ``__dict__`` walk for the HTTPSession-side
        # kwargs and just restate the positional pair.
        _, parent_kwargs = super().__getnewargs_ex__()
        parent_kwargs.pop("schema", None)
        return (self.schema, self.base_url), parent_kwargs

    # ── identity / introspection ───────────────────────────────────────────

    def __repr__(self) -> str:
        base = self.base_url.to_string() if self.base_url else None
        return (
            f"SchemaSession(schema={self.schema.full_name()!r}, "
            f"base_url={base!r}, mode={self.mode.value!r})"
        )

    # ── path → table mapping ───────────────────────────────────────────────

    def table_for(self, request: PreparedRequest) -> "Table":
        """Return (caching) the :class:`Table` that backs *request*'s URL path.

        The URL path is fed through :meth:`Table.safe_name` — the
        centralized "raw string → Unity-Catalog-safe identifier"
        builder — so the cache table name is derived the same way
        for every caller that touches the codebase, and the warning
        for non-trivial sanitization fires once per fresh path.
        """
        name = Table.safe_name(request.url.path)
        return self._table_cache.get_or_set(name, lambda: self.schema.table(name))

    # ── cache config attachment ────────────────────────────────────────────

    def _build_local_template(
        self,
        local_cache: Union[bool, str, Path, CacheConfig, Mapping[str, Any], None],
    ) -> Optional[CacheConfig]:
        """Resolve the constructor's ``local_cache`` arg into a reusable template."""
        if local_cache is False or local_cache is None:
            return None
        if local_cache is True:
            base = CacheConfig.default()
            return base.merge(path=base.local_cache_path(session=self), mode=self.mode)
        cfg = CacheConfig.check_arg(local_cache)
        if not cfg.is_local:
            cfg = cfg.merge(path=cfg.local_cache_path(session=self))
        return cfg if cfg.mode == self.mode else cfg.merge(mode=self.mode)

    def _attach_cache(self, request: PreparedRequest) -> PreparedRequest:
        """Stamp the per-path remote :class:`CacheConfig` (and local template,
        when enabled) onto *request* unless the caller already supplied one.

        Per-request ``mode`` overrides ride into the remote config; the
        local template carries the session-level mode and merges only if
        the request asks for something different.

        When a :class:`TimeWindowPolicy` is configured, the request's
        URL is normalised first so the cache key (and the writeback
        URL on a miss) reflects the snapped window, not the raw
        caller-supplied bounds.
        """
        if self.time_window is not None:
            normalized = self.time_window.apply(request.url)
            if normalized is not request.url:
                request.url = normalized
        if request.remote_cache_config is None:
            mode = request.mode if request.mode is not None else self.mode
            request.remote_cache_config = CacheConfig(
                tabular=self.table_for(request), mode=mode,
            )
        if request.local_cache_config is None and self._local_cache_template is not None:
            tmpl = self._local_cache_template
            mode = request.mode if request.mode is not None else self.mode
            request.local_cache_config = tmpl if tmpl.mode == mode else tmpl.merge(mode=mode)
        return request

    # ── transport hooks ────────────────────────────────────────────────────

    def _send(self, request: PreparedRequest, config):  # type: ignore[override]
        return super()._send(self._attach_cache(request), config)

    def _send_many_batches(
        self,
        requests: Iterator[PreparedRequest],
        config: Any,
    ):  # type: ignore[override]
        return super()._send_many_batches(
            (self._attach_cache(r) for r in requests),
            config,
        )
