"""HTTP session backed by a Unity Catalog schema as remote response cache.

:class:`SchemaSession` subclasses :class:`~yggdrasil.io.http_.HTTPSession`
and routes every outbound request through a per-path Delta table that
acts as the remote cache. Identical URLs collapse to the same table on
the schema, so a warmed cache repays subsequent calls with a SQL lookup
instead of a network round-trip.

Cache semantics, controlled by ``mode`` (default :attr:`Mode.APPEND`):

* :attr:`Mode.APPEND` — check the cache table for a matching row first;
  on miss, fetch from the API and append the new response. This is the
  pure "read-through, write-once" path.
* Any other mode (:attr:`Mode.UPSERT`, :attr:`Mode.OVERWRITE`, …) —
  skip the cache lookup, fetch fresh from the API, then UPSERT / replace
  the row in the table. Useful when the upstream API state may have
  changed and the cache is being repaired.

The session is otherwise fully transparent: callers use ``get`` / ``post``
/ ``send`` exactly as on the parent :class:`HTTPSession`; the cache is
plumbed via the existing :class:`~yggdrasil.io.send_config.CacheConfig`
remote-cache pipeline.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Union

from yggdrasil.dataclasses import ExpiringDict
from yggdrasil.dataclasses.waiting import DEFAULT_WAITING_CONFIG, WaitingConfig
from yggdrasil.data.enums import Mode
from yggdrasil.data.enums.mode import ModeLike
from yggdrasil.databricks.sql.sql_utils import MAX_TABLE_NAME_LEN, safe_table_name
from yggdrasil.io.http_.session import HTTPSession
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.send_config import CacheConfig
from yggdrasil.io.url import URL

if TYPE_CHECKING:
    from yggdrasil.databricks.schema.schema import Schema
    from yggdrasil.databricks.table.table import Table
    from yggdrasil.io.authorization.base import Authorization
    from yggdrasil.io.headers import Headers


__all__ = ["SchemaSession"]


LOGGER = logging.getLogger(__name__)


# URL path → identifier: collapse anything outside ``[0-9A-Za-z]`` to ``_``.
# Run once at module load so every per-request derivation is one ``re.sub``
# call instead of allocating a fresh compiled pattern.
_PATH_TO_IDENT_RE: re.Pattern[str] = re.compile(r"[^0-9A-Za-z]+")

# Default per-path :class:`Table` cache TTL. One hour is long enough to
# amortise the schema lookup across a typical batch / job, short enough
# that a rename / drop upstream surfaces on the next refresh rather than
# pinning a stale handle for the process lifetime.
_DEFAULT_TABLE_CACHE_TTL: dt.timedelta = dt.timedelta(hours=1)


class SchemaSession(HTTPSession):
    """HTTP session that uses a Databricks schema as its remote cache.

    Every distinct URL path resolves to one Delta table on the bound
    :class:`Schema`. The first call against a path lazily provisions the
    table (the parent ``_load_remote_cached_response`` already creates
    on ``TABLE_OR_VIEW_NOT_FOUND``); subsequent calls reuse it.

    Args:
        schema:           The :class:`Schema` whose tables will back the
                          per-path cache.
        base_url:         Forwarded to :class:`HTTPSession`. When set,
                          the session is singleton-cached per
                          ``(class, base_url, key)`` like any other.
        mode:             Cache write disposition. :attr:`Mode.APPEND`
                          (default) reads the cache first; anything
                          else (:attr:`Mode.UPSERT`, …) always fetches
                          from the API and writes back.
        table_cache_ttl:  TTL on the in-process per-path :class:`Table`
                          handle cache. Defaults to 1 hour.
        local_cache:      On-disk fast-path cache control. ``True``
                          (default) enables the local cache at the
                          session's default folder
                          (``~/.yggdrasil/cache/response/<host>/<path>``);
                          a ``str`` / :class:`Path` sets an explicit
                          directory; a :class:`CacheConfig` is used
                          as-is; ``False`` / ``None`` disables the
                          local layer (remote cache only).
    """

    def __new__(  # type: ignore[override]
        cls,
        schema: "Schema | None" = None,
        base_url: Optional[URL | str] = None,
        *args: Any,
        key: str = "",
        **kwargs: Any,
    ) -> "SchemaSession":
        # The parent's singleton cache keys off ``(cls, base_url, key)``;
        # forward ``base_url`` explicitly so SchemaSession's signature
        # (``schema`` first) doesn't clash with the positional ``base_url``
        # in :meth:`Session.__new__`. The ``schema`` arg is allowed to
        # be ``None`` here so :meth:`__getnewargs_ex__` round-trips
        # don't trip the type checker — the real ``__init__`` rejects
        # a missing schema.
        return super().__new__(cls, base_url, key=key)

    def __init__(
        self,
        schema: "Schema",
        base_url: Optional[URL | str] = None,
        *,
        mode: ModeLike = Mode.APPEND,
        table_cache_ttl: "float | int | dt.timedelta | None" = _DEFAULT_TABLE_CACHE_TTL,
        local_cache: Union[bool, str, Path, CacheConfig, Mapping[str, Any], None] = True,
        verify: bool = True,
        pool_maxsize: int = 10,
        headers: "Headers | Mapping[str, str] | None" = None,
        waiting: WaitingConfig = DEFAULT_WAITING_CONFIG,
        key: str = "",
        auth: Optional["Authorization"] = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
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
        # Memoized ``CacheConfig`` per (table_name, mode). The remote
        # cache config is purely a function of those two — caching the
        # built dataclass turns a per-cold-request ``CacheConfig(...)``
        # (≈9 µs: ``__post_init__`` runs request_by validation,
        # WaitingConfig.from_, mode coercion, …) into a dict lookup.
        # Bounded by ``max_size=1024`` so a runaway distinct-path
        # session doesn't grow the dict unbounded; same TTL as the
        # table cache.
        self._remote_config_cache: ExpiringDict[tuple[str, Mode], CacheConfig] = ExpiringDict(
            default_ttl=table_cache_ttl,
            max_size=1024,
        )
        self._local_cache_template: Optional[CacheConfig] = self._build_local_template(
            local_cache,
        )

    # ``_table_cache`` is picklable (live entries only — see
    # ``ExpiringDict.__getstate__``), so it stays out of the transient
    # set. The parent's ``_lock`` / ``_job_pool`` / ``_http_pool`` are
    # already covered.
    _TRANSIENT_STATE_ATTRS = HTTPSession._TRANSIENT_STATE_ATTRS

    def __getnewargs_ex__(self):
        # Route unpickling through ``__new__`` with the schema as the
        # first positional arg so the singleton cache key matches the
        # original construction.
        return (
            (self._schema, self.base_url),
            {"key": self.key, "mode": self._mode.value},
        )

    # ── factory ────────────────────────────────────────────────────────────

    @staticmethod
    def host_to_schema_name(base_url: URL | str) -> str:
        """Derive a Unity-Catalog-safe schema name from a URL host.

        Same sanitization as :meth:`path_to_table_name` (lowercase,
        collapse non-alphanumeric runs to ``_``, strip surrounding
        underscores, length-cap with :func:`safe_table_name`'s
        split-and-hash strategy — the 255-char UC ceiling applies to
        every identifier, schema names included). Non-default ports
        are folded in so ``https://api.example.com:8443`` and
        ``https://api.example.com`` route to different schemas; an
        empty host falls back to ``"default"`` so the result is always
        a legal identifier.
        """
        url = base_url if isinstance(base_url, URL) else URL.from_(base_url)
        host = (url.host or "").lower()
        port = getattr(url, "port", None)
        token = f"{host}_{port}" if port else host
        cleaned = _PATH_TO_IDENT_RE.sub("_", token).strip("_") or "default"
        name = safe_table_name(cleaned)
        assert name is not None and len(name) <= MAX_TABLE_NAME_LEN
        return name

    @classmethod
    def from_(
        cls,
        obj: Any = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        ensure_created: bool = True,
        **kwargs: Any,
    ) -> "SchemaSession":
        """Build a :class:`SchemaSession` from one of the usual schema specs.

        Polymorphic on *obj*:

        * :class:`SchemaSession` — returned as-is (idempotent).
        * :class:`Schema` — wrapped directly.
        * :class:`Schemas` service — combined with *catalog_name* /
          *schema_name* (or its own defaults) to resolve a Schema.
        * :class:`DatabricksClient` — its ``schemas`` service is used.
        * ``str`` — parsed as a one- or two-part ``"catalog.schema"``
          name; missing parts come from the *catalog_name* /
          *schema_name* kwargs.
        * ``None`` — *catalog_name* / *schema_name* must be supplied;
          the active :class:`Schemas` service is used (``Schemas.current()``
          via :meth:`Schema` construction).

        When *schema_name* is not supplied (and the caller did not hand
        in a fully-resolved :class:`Schema`), the session's ``base_url``
        host is used: ``https://api.example.com`` →
        ``api_example_com``. This makes the default
        "one cache schema per upstream host" — call sites pointed at
        different APIs land in their own namespaces without ceremony.
        See :meth:`host_to_schema_name` for the sanitization rules.

        When *ensure_created* (default ``True``) is set, the resolved
        :class:`Schema` is created if it does not already exist via
        :meth:`Schema.ensure_created`. Set to ``False`` to skip the
        existence probe (e.g. when the caller knows the schema is
        present and wants to avoid the round-trip).

        Extra keyword arguments are forwarded to :class:`SchemaSession`
        (``base_url``, ``mode``, ``local_cache``, ``key``, …).
        """
        if isinstance(obj, cls):
            return obj

        from yggdrasil.databricks.client import DatabricksClient
        from yggdrasil.databricks.schema.schema import Schema as _Schema
        from yggdrasil.databricks.schema.schemas import Schemas as _Schemas

        if isinstance(obj, _Schema):
            schema = obj
        else:
            schemas: _Schemas
            location: str | None = None
            if obj is None:
                schemas = _Schemas.current()
            elif isinstance(obj, _Schemas):
                schemas = obj
            elif isinstance(obj, DatabricksClient):
                schemas = obj.schemas
            elif isinstance(obj, str):
                schemas = _Schemas.current()
                location = obj
            else:
                raise TypeError(
                    f"SchemaSession.from_: cannot resolve a Schema from "
                    f"{type(obj).__name__}: {obj!r}. Pass a Schema, "
                    "SchemaSession, Schemas service, DatabricksClient, "
                    "two-part 'catalog.schema' string, or None plus "
                    "explicit catalog_name / schema_name."
                )

            # Default schema_name = ``base_url`` host. Skipped when the
            # caller passed a ``location`` ('catalog.schema') string —
            # the dotted form already carries the schema name.
            if not schema_name and location is None:
                base_url = kwargs.get("base_url")
                if base_url is not None:
                    schema_name = cls.host_to_schema_name(base_url)

            schema = schemas.schema(
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
            )

        if ensure_created:
            schema.ensure_created()

        return cls(schema, **kwargs)

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

    # ── path → table mapping ────────────────────────────────────────────────

    def path_to_table_name(self, path: str | None) -> str:
        """Derive a Unity-Catalog-safe table name from a URL path.

        Pipeline:

        1. Lowercase the path, collapse every run of non-alphanumeric
           chars to a single ``_`` (so ``/``, ``.``, query-string
           punctuation, and non-ASCII all fold to the same separator).
        2. Strip surrounding underscores; substitute ``"root"`` for the
           empty result so ``"/"`` still yields a legal identifier.
        3. Hand off to :func:`safe_table_name`, which enforces Unity
           Catalog's 255-char identifier ceiling by splitting on ``_``,
           keeping as many leading tokens as fit, and BLAKE2b-hashing
           the overflow tail into a 32-char suffix. Distinct overflows
           still produce distinct digests, so two long paths that
           agree only on a common prefix get separate cache tables.
        4. Defensive assert that the result fits the limit — if a
           future refactor breaks the contract, fail loudly at the
           boundary rather than at the SQL layer with a generic
           identifier-too-long error.

        The result is deterministic, identifier-safe ASCII, and bounded
        at :data:`MAX_TABLE_NAME_LEN` (255).
        """
        cleaned = _PATH_TO_IDENT_RE.sub("_", (path or "").lower()).strip("_")
        if not cleaned:
            cleaned = "root"
        # ``safe_table_name`` never returns ``None`` for a non-empty input.
        name = safe_table_name(cleaned)
        assert name is not None and len(name) <= MAX_TABLE_NAME_LEN, (
            f"SchemaSession.path_to_table_name: derived name {name!r} "
            f"({len(name) if name else 0} chars) exceeds Unity Catalog's "
            f"{MAX_TABLE_NAME_LEN}-char limit — safe_table_name contract broken."
        )
        return name

    def table_for(self, request: PreparedRequest) -> "Table":
        """Return (caching) the :class:`Table` that backs *request*'s URL path.

        Idempotent and thread-safe — multiple concurrent calls for the
        same path collapse to one :meth:`Schema.table` lookup via the
        :class:`ExpiringDict`'s ``get_or_set`` fast path.
        """
        name = self.path_to_table_name(request.url.path)
        return self._table_cache.get_or_set(name, lambda: self._schema.table(name))

    def cache_config_for(self, request: PreparedRequest) -> CacheConfig:
        """Build the :class:`CacheConfig` that drives the remote cache for *request*.

        Honors a per-request mode override (``request.mode``); falls back
        to the session-level :attr:`mode` when the request didn't set one.

        The result is memoized per ``(table_name, mode)`` on the
        session — :class:`CacheConfig` construction runs a handful of
        coercion + validation passes through ``__post_init__`` (≈9 µs
        cold), and bursts against the same endpoint always rebuild the
        same config. Cache TTL matches the per-path :class:`Table`
        cache so a stale handle and its associated config age out
        together.
        """
        table = self.table_for(request)
        mode = request.mode if request.mode is not None else self._mode
        # ``id(table)`` is stable for the lifetime of the cached Table
        # handle (held by ``self._table_cache``) and avoids the
        # ``full_name()`` join on every hot lookup. When the table
        # entry expires the live config expires alongside it.
        key = (id(table), mode)
        return self._remote_config_cache.get_or_set(
            key, lambda: CacheConfig(tabular=table, mode=mode),
        )

    def local_cache_config_for(self, request: PreparedRequest) -> Optional[CacheConfig]:
        """Return the local cache :class:`CacheConfig` for *request*, or ``None``.

        ``None`` means local caching is disabled on this session. When
        enabled, the returned config inherits the session-level template
        (path + cleanup TTL) and picks up any per-request
        :attr:`PreparedRequest.mode` override.
        """
        tmpl = self._local_cache_template
        if tmpl is None:
            return None
        mode = request.mode if request.mode is not None else self._mode
        if tmpl.mode == mode:
            return tmpl
        return tmpl.merge(mode=mode)

    def _build_local_template(
        self,
        local_cache: Union[bool, str, Path, CacheConfig, Mapping[str, Any], None],
    ) -> Optional[CacheConfig]:
        """Resolve the constructor's ``local_cache`` arg into a reusable template.

        Done once at init so per-request attachment is a single
        attribute read instead of a fresh :class:`CacheConfig` build.
        Returns ``None`` when local caching is disabled.
        """
        if local_cache is False or local_cache is None:
            return None
        if local_cache is True:
            base = CacheConfig.default()
            return base.merge(path=base.local_cache_path(session=self), mode=self._mode)
        cfg = CacheConfig.check_arg(local_cache)
        if not cfg.is_local:
            # Ensure ``local_cache_enabled`` actually fires for a caller
            # who passed a bare mapping or CacheConfig with no path —
            # resolve to the session-default folder so the on-disk
            # fast-path is reachable.
            cfg = cfg.merge(path=cfg.local_cache_path(session=self))
        return cfg if cfg.mode == self._mode else cfg.merge(mode=self._mode)

    def _attach_cache(self, request: PreparedRequest) -> PreparedRequest:
        """Stamp the per-path :class:`CacheConfig`(s) onto *request* if missing.

        Attaches both the remote (per-path Delta table) and — when
        enabled — the local on-disk fast-path config. Per-request
        ``*_cache_config`` overrides take precedence over the
        session-level configs inside :meth:`Session._send`, so a
        caller that explicitly passes ``remote_cache=...`` /
        ``local_cache=...`` to ``send`` still wins. A per-request
        :attr:`PreparedRequest.mode` override flows into both cache
        configs automatically.
        """
        if request.remote_cache_config is None:
            request.remote_cache_config = self.cache_config_for(request)
        if request.local_cache_config is None and self._local_cache_template is not None:
            request.local_cache_config = self.local_cache_config_for(request)
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
