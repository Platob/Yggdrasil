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
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Mapping, Optional, Union

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


__all__ = ["SchemaSession"]


LOGGER = logging.getLogger(__name__)

# Per-path Table handle cache TTL. One hour amortises the schema lookup
# across a typical job while still surfacing upstream renames / drops
# on the next refresh.
_DEFAULT_TABLE_CACHE_TTL: dt.timedelta = dt.timedelta(hours=1)


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
        table_cache_ttl: TTL on the in-process per-path :class:`Table`
            handle cache (1 hour by default).

    Remaining keyword arguments forward to :class:`HTTPSession`.
    """

    #: Namespace prefix on the parent's singleton key so a
    #: ``SchemaSession`` never collides with a bare ``HTTPSession``
    #: sharing the same ``(base_url, key)`` pair.
    KEY_PREFIX: ClassVar[str] = "yggdrasil.schema_session:"

    @classmethod
    def _prefixed_key(cls, key: str) -> str:
        return key if key.startswith(cls.KEY_PREFIX) else f"{cls.KEY_PREFIX}{key}"

    def __new__(  # type: ignore[override]
        cls,
        schema: "Schema",
        base_url: Optional[URL | str] = None,
        *args: Any,
        key: str = "",
        **kwargs: Any,
    ) -> "SchemaSession":
        return super().__new__(cls, base_url, key=cls._prefixed_key(key))

    def __init__(
        self,
        schema: "Schema",
        base_url: Optional[URL | str] = None,
        *,
        mode: ModeLike = Mode.APPEND,
        local_cache: Union[bool, str, Path, CacheConfig, Mapping[str, Any], None] = True,
        table_cache_ttl: "float | int | dt.timedelta | None" = _DEFAULT_TABLE_CACHE_TTL,
        verify: bool = True,
        pool_maxsize: int = 10,
        headers: "Headers | Mapping[str, str] | None" = None,
        waiting: WaitingConfig = DEFAULT_WAITING_CONFIG,
        key: str = "",
        auth: Optional["Authorization"] = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        if schema is None:
            raise TypeError(
                "SchemaSession requires a bound Schema; got None. Pass "
                "the Schema whose tables should back the response cache."
            )
        key = self._prefixed_key(key)
        super().__init__(
            base_url=base_url,
            verify=verify,
            pool_maxsize=pool_maxsize,
            headers=headers,
            waiting=waiting,
            key=key,
            auth=auth,
        )
        self._schema = schema
        self._mode = Mode.from_(mode, default=Mode.APPEND)
        self._table_cache: ExpiringDict[str, "Table"] = ExpiringDict(
            default_ttl=table_cache_ttl,
            max_size=1024,
        )
        self._local_cache_template: Optional[CacheConfig] = self._build_local_template(
            local_cache,
        )

    # ``_table_cache`` pickles live entries via ``ExpiringDict.__getstate__``;
    # ``_local_cache_template`` is a frozen dataclass — both round-trip cleanly.
    _TRANSIENT_STATE_ATTRS = HTTPSession._TRANSIENT_STATE_ATTRS

    def __getnewargs_ex__(self):
        return (
            (self._schema, self.base_url),
            {"key": self.key, "mode": self._mode.value},
        )

    # ── identity / introspection ───────────────────────────────────────────

    @property
    def schema(self) -> "Schema":
        """The bound :class:`Schema` whose tables back the cache."""
        return self._schema

    @property
    def mode(self) -> Mode:
        """Cache write disposition (:attr:`Mode.APPEND` by default)."""
        return self._mode

    def __repr__(self) -> str:
        base = self.base_url.to_string() if self.base_url else None
        return (
            f"SchemaSession(schema={self._schema.full_name()!r}, "
            f"base_url={base!r}, mode={self._mode.value!r})"
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
        return self._table_cache.get_or_set(name, lambda: self._schema.table(name))

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
            return base.merge(path=base.local_cache_path(session=self), mode=self._mode)
        cfg = CacheConfig.check_arg(local_cache)
        if not cfg.is_local:
            cfg = cfg.merge(path=cfg.local_cache_path(session=self))
        return cfg if cfg.mode == self._mode else cfg.merge(mode=self._mode)

    def _attach_cache(self, request: PreparedRequest) -> PreparedRequest:
        """Stamp the per-path remote :class:`CacheConfig` (and local template,
        when enabled) onto *request* unless the caller already supplied one.

        Per-request ``mode`` overrides ride into the remote config; the
        local template carries the session-level mode and merges only if
        the request asks for something different.
        """
        if request.remote_cache_config is None:
            mode = request.mode if request.mode is not None else self._mode
            request.remote_cache_config = CacheConfig(
                tabular=self.table_for(request), mode=mode,
            )
        if request.local_cache_config is None and self._local_cache_template is not None:
            tmpl = self._local_cache_template
            mode = request.mode if request.mode is not None else self._mode
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
