"""Configuration dataclasses for :class:`~yggdrasil.io.session.Session` send methods.

Two dataclasses are provided:

* :class:`SendConfig` – controls a **single** request sent via
  :meth:`~yggdrasil.io.session.Session.send`.
* :class:`SendManyConfig` – extends :class:`SendConfig` with additional
  options for concurrent, batched, and cached execution via
  :meth:`~yggdrasil.io.session.Session.send_many`.

Both are *frozen* so they can be safely shared across threads and expose a
consistent factory API:

``cls.default()``
    Return the instance with all fields at their defaults.

``cls.parse_mapping(options)``
    Build an instance from a ``Mapping[str, Any]``.  Unknown keys are
    silently ignored so callers can pass broad option dicts without filtering.

``cls.check_arg(arg)``
    Coerce *arg* into a config instance.  Accepts ``None`` (returns default),
    an existing instance (returned unchanged), or any mapping.

``instance.merge(**overrides)``
    Return a new instance with the given fields replaced, leaving all other
    fields unchanged.

Typical usage::

    from yggdrasil.io.send_config import SendConfig, SendManyConfig

    cfg = SendConfig(stream=True, raise_error=True, wait=30)

    # override a single field without rebuilding the whole object
    no_stream = cfg.merge(stream=False)

    # build from an options dict (e.g. from a config file or API response)
    cfg2 = SendConfig.parse_mapping({"stream": False, "raise_error": False})

    # many requests
    many_cfg = SendManyConfig(stream=True, batch_size=50, ordered=True)
    for resp in session.send_many(requests, config=many_cfg):
        ...
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, TYPE_CHECKING

from yggdrasil.dataclasses.waiting import WaitingConfigArg

if TYPE_CHECKING:
    from yggdrasil.databricks.sql.table import Table

__all__ = ["SendConfig", "SendManyConfig"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Field names understood by each config class – used by parse_mapping to
# filter out unknown keys before calling the dataclass constructor.
_SEND_CONFIG_FIELDS: frozenset[str] = frozenset({
    "wait", "raise_error", "stream", "cache", "cache_by",
    "anonymize", "received_from", "received_to", "wait_cache",
})

_SEND_MANY_CONFIG_FIELDS: frozenset[str] = _SEND_CONFIG_FIELDS | frozenset({
    "cache_anonymize", "normalize", "batch_size", "ordered", "max_in_flight",
})


# ---------------------------------------------------------------------------
# SendConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SendConfig:
    """Configuration for a single HTTP request.

    Pass an instance as ``config=`` to
    :meth:`~yggdrasil.io.session.Session.send` or any of the convenience
    HTTP methods (``get``, ``post``, …).  Individual keyword arguments on the
    call site override the corresponding field.

    Parameters
    ----------
    wait:
        Waiting / retry configuration forwarded to the transport layer.
        Accepts anything that :class:`~yggdrasil.dataclasses.waiting.WaitingConfig`
        understands: a :class:`WaitingConfig` instance, a dict, a timeout in
        seconds (``int`` or ``float``), a deadline :class:`datetime.datetime`,
        or ``True`` / ``False``.  ``None`` defers to the session default.
    raise_error:
        When ``True`` (default) a non-2xx response raises immediately.
        Set to ``False`` to inspect error responses manually.
    stream:
        When ``True`` (default) the response body is streamed lazily.
        Set to ``False`` to buffer the entire body before returning.
    cache:
        Optional Databricks Delta table used to cache responses.  When
        provided the session checks the table before hitting the network and
        writes new responses back on success.
    cache_by:
        List of response-schema column names that form the cache key.
        Defaults to ``["request_method", "request_url_host",
        "request_url_path", "request_url_query",
        "request_content_length", "request_body_hash"]`` when ``cache``
        is set.  Must be a subset of
        :data:`~yggdrasil.io.response.RESPONSE_ARROW_SCHEMA` field names.
    anonymize:
        Controls how sensitive fields are removed before caching or
        comparing cached entries.

        * ``"remove"`` – strip values completely (default).
        * ``"redact"`` – replace values with a fixed placeholder.
    received_from:
        Earliest ``response_received_at`` timestamp to accept from cache.
        Any timezone-aware datetime, date, or ISO-8601 string is accepted.
    received_to:
        Latest ``response_received_at`` timestamp to accept from cache.
    wait_cache:
        Waiting config for the asynchronous cache-write operation.
        ``False`` (default) means fire-and-forget.
    """

    wait: WaitingConfigArg = None
    raise_error: bool = True
    stream: bool = True
    cache: Optional["Table"] = field(default=None, hash=False, compare=False)
    cache_by: Optional[list[str]] = field(default=None, hash=False, compare=False)
    anonymize: Literal["remove", "redact"] = "remove"
    received_from: Optional[dt.datetime | dt.date | str] = None
    received_to: Optional[dt.datetime | dt.date | str] = None
    wait_cache: WaitingConfigArg = False

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "SendConfig":
        """Return a :class:`SendConfig` with all fields at their defaults.

        Examples
        --------
        >>> SendConfig.default()
        SendConfig(wait=None, raise_error=True, stream=True, ...)
        """
        return cls()

    @classmethod
    def parse_mapping(
        cls,
        options: Mapping[str, Any],
        **overrides: Any,
    ) -> "SendConfig":
        """Build a :class:`SendConfig` from a mapping (e.g. a dict or config file row).

        Unknown keys are silently dropped so callers can pass broad option
        dicts without filtering them first.  *overrides* are applied on top
        after the mapping is parsed.

        Parameters
        ----------
        options:
            Any ``Mapping[str, Any]``.  Recognised keys match the field names
            of :class:`SendConfig`; all other keys are ignored.
        **overrides:
            Keyword arguments that take precedence over values in *options*.
            Useful for pinning one field while letting the rest come from the
            mapping.

        Returns
        -------
        SendConfig

        Raises
        ------
        TypeError
            If *options* is not a ``Mapping``.

        Examples
        --------
        >>> SendConfig.parse_mapping({"stream": False, "raise_error": False})
        SendConfig(wait=None, raise_error=False, stream=False, ...)

        >>> SendConfig.parse_mapping({"stream": False}, raise_error=False)
        SendConfig(wait=None, raise_error=False, stream=False, ...)

        >>> # extra keys are silently ignored
        >>> SendConfig.parse_mapping({"stream": False, "unknown_key": 99})
        SendConfig(wait=None, raise_error=True, stream=False, ...)
        """
        if not isinstance(options, Mapping):
            raise TypeError(
                f"SendConfig.parse_mapping expects a Mapping, got {type(options).__name__!r}"
            )
        known = {k: v for k, v in options.items() if k in _SEND_CONFIG_FIELDS}
        known.update(overrides)
        return cls(**known)

    @classmethod
    def check_arg(
        cls,
        arg: "SendConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "SendConfig":
        """Coerce *arg* into a :class:`SendConfig`.

        Accepted forms:

        * ``None`` – returns :meth:`default` (optionally with *overrides*).
        * :class:`SendConfig` instance – returned as-is when no *overrides*
          are given; otherwise merged with the overrides.
        * ``Mapping`` – forwarded to :meth:`parse_mapping`.

        Parameters
        ----------
        arg:
            Value to coerce.
        **overrides:
            Field overrides applied after coercion.  Allows callers to pin
            individual fields regardless of the input type.

        Returns
        -------
        SendConfig

        Raises
        ------
        TypeError
            If *arg* is not one of the accepted types.

        Examples
        --------
        >>> SendConfig.check_arg(None)
        SendConfig(wait=None, raise_error=True, stream=True, ...)

        >>> SendConfig.check_arg({"stream": False})
        SendConfig(wait=None, raise_error=True, stream=False, ...)

        >>> cfg = SendConfig(stream=False)
        >>> SendConfig.check_arg(cfg)
        SendConfig(wait=None, raise_error=True, stream=False, ...)

        >>> SendConfig.check_arg(cfg, raise_error=False)
        SendConfig(wait=None, raise_error=False, stream=False, ...)
        """
        if arg is None:
            return cls(**overrides) if overrides else cls.default()
        if isinstance(arg, cls):
            return arg.merge(**overrides) if overrides else arg
        if isinstance(arg, Mapping):
            return cls.parse_mapping(arg, **overrides)
        raise TypeError(
            f"SendConfig.check_arg expects a SendConfig, Mapping, or None; "
            f"got {type(arg).__name__!r}"
        )

    # ------------------------------------------------------------------
    # Instance helpers
    # ------------------------------------------------------------------

    def merge(self, **overrides: Any) -> "SendConfig":
        """Return a new :class:`SendConfig` with *overrides* applied.

        All fields not listed in *overrides* retain their current values.
        Because :class:`SendConfig` is frozen, this always creates a new
        instance.

        Parameters
        ----------
        **overrides:
            Fields to replace.  Unknown field names raise :exc:`TypeError`.

        Returns
        -------
        SendConfig

        Raises
        ------
        TypeError
            If any key in *overrides* is not a recognised field name.

        Examples
        --------
        >>> cfg = SendConfig(stream=True, raise_error=True)
        >>> cfg.merge(stream=False)
        SendConfig(wait=None, raise_error=True, stream=False, ...)
        """
        unknown = set(overrides) - _SEND_CONFIG_FIELDS
        if unknown:
            raise TypeError(
                f"SendConfig.merge got unexpected field(s): {sorted(unknown)!r}"
            )
        return dataclasses.replace(self, **overrides)


# ---------------------------------------------------------------------------
# SendManyConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SendManyConfig:
    """Configuration for concurrent batched HTTP requests.

    Pass an instance as ``config=`` to
    :meth:`~yggdrasil.io.session.Session.send_many`.  Individual keyword
    arguments on the call site override the corresponding field.

    All :class:`SendConfig` fields are duplicated here (rather than
    inheriting) to preserve ``frozen=True`` and ``slots=True`` semantics,
    and to allow independent defaults for the many-request path.

    Parameters
    ----------
    wait:
        See :attr:`SendConfig.wait`.
    raise_error:
        See :attr:`SendConfig.raise_error`.
    stream:
        See :attr:`SendConfig.stream`.
    cache:
        See :attr:`SendConfig.cache`.
    cache_by:
        See :attr:`SendConfig.cache_by`.
    cache_anonymize:
        Controls how sensitive fields are stripped before the cache lookup
        and before writing new entries.  Uses the same
        ``"remove"`` / ``"redact"`` semantics as :attr:`SendConfig.anonymize`.
    received_from:
        See :attr:`SendConfig.received_from`.
    received_to:
        See :attr:`SendConfig.received_to`.
    wait_cache:
        See :attr:`SendConfig.wait_cache`.
    normalize:
        When ``True`` URLs are normalised before the request is dispatched.
        ``None`` (default) normalises automatically when ``cache`` is set.
    batch_size:
        Number of requests per cache-lookup batch.  ``None`` defaults to
        ``pool_maxsize × 100``.  Has no effect when ``cache`` is ``None``.
    ordered:
        When ``True`` responses are yielded in the same order as the input
        requests.  ``False`` (default) yields in completion order for higher
        throughput.
    max_in_flight:
        Maximum number of concurrent in-flight requests in the thread pool.
        ``None`` uses the pool's natural concurrency limit.
    """

    # --- shared with SendConfig ---
    wait: WaitingConfigArg = None
    raise_error: bool = True
    stream: bool = True
    cache: Optional["Table"] = field(default=None, hash=False, compare=False)
    cache_by: Optional[list[str]] = field(default=None, hash=False, compare=False)
    cache_anonymize: Literal["remove", "redact"] = "remove"
    received_from: Optional[dt.datetime | dt.date | str] = None
    received_to: Optional[dt.datetime | dt.date | str] = None
    wait_cache: WaitingConfigArg = False

    # --- send_many-specific ---
    normalize: Optional[bool] = None
    batch_size: Optional[int] = None
    ordered: bool = False
    max_in_flight: Optional[int] = None

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "SendManyConfig":
        """Return a :class:`SendManyConfig` with all fields at their defaults.

        Examples
        --------
        >>> SendManyConfig.default()
        SendManyConfig(wait=None, raise_error=True, stream=True, ...)
        """
        return cls()

    @classmethod
    def parse_mapping(
        cls,
        options: Mapping[str, Any],
        **overrides: Any,
    ) -> "SendManyConfig":
        """Build a :class:`SendManyConfig` from a mapping.

        Unknown keys are silently dropped.  *overrides* are applied on top
        after the mapping is parsed.

        Parameters
        ----------
        options:
            Any ``Mapping[str, Any]``.  Recognised keys match the field names
            of :class:`SendManyConfig` (a superset of :class:`SendConfig`
            fields); all other keys are ignored.
        **overrides:
            Keyword arguments that take precedence over values in *options*.

        Returns
        -------
        SendManyConfig

        Raises
        ------
        TypeError
            If *options* is not a ``Mapping``.

        Examples
        --------
        >>> SendManyConfig.parse_mapping({"batch_size": 50, "ordered": True})
        SendManyConfig(..., batch_size=50, ordered=True, ...)

        >>> SendManyConfig.parse_mapping({"stream": False}, batch_size=100)
        SendManyConfig(..., stream=False, batch_size=100, ...)

        >>> # SendConfig fields are also accepted
        >>> SendManyConfig.parse_mapping({"raise_error": False, "stream": False})
        SendManyConfig(..., raise_error=False, stream=False, ...)
        """
        if not isinstance(options, Mapping):
            raise TypeError(
                f"SendManyConfig.parse_mapping expects a Mapping, "
                f"got {type(options).__name__!r}"
            )
        known = {k: v for k, v in options.items() if k in _SEND_MANY_CONFIG_FIELDS}
        known.update(overrides)
        return cls(**known)

    @classmethod
    def check_arg(
        cls,
        arg: "SendManyConfig | SendConfig | Mapping[str, Any] | None",
        **overrides: Any,
    ) -> "SendManyConfig":
        """Coerce *arg* into a :class:`SendManyConfig`.

        Accepted forms:

        * ``None`` – returns :meth:`default` (optionally with *overrides*).
        * :class:`SendManyConfig` instance – returned as-is (or merged with
          *overrides*).
        * :class:`SendConfig` instance – promoted to a :class:`SendManyConfig`
          using the shared fields; send-many–specific fields take their
          defaults (or are set via *overrides*).
        * ``Mapping`` – forwarded to :meth:`parse_mapping`.

        Parameters
        ----------
        arg:
            Value to coerce.
        **overrides:
            Field overrides applied after coercion.

        Returns
        -------
        SendManyConfig

        Raises
        ------
        TypeError
            If *arg* is not one of the accepted types.

        Examples
        --------
        >>> SendManyConfig.check_arg(None)
        SendManyConfig(wait=None, raise_error=True, ...)

        >>> SendManyConfig.check_arg({"batch_size": 20, "ordered": True})
        SendManyConfig(..., batch_size=20, ordered=True, ...)

        >>> base = SendConfig(stream=False, raise_error=False)
        >>> SendManyConfig.check_arg(base, batch_size=50)
        SendManyConfig(..., stream=False, raise_error=False, batch_size=50, ...)
        """
        if arg is None:
            return cls(**overrides) if overrides else cls.default()
        if isinstance(arg, cls):
            return arg.merge(**overrides) if overrides else arg
        if isinstance(arg, SendConfig):
            # Promote a SendConfig to SendManyConfig, mapping the shared fields.
            return cls(
                wait=arg.wait,
                raise_error=arg.raise_error,
                stream=arg.stream,
                cache=arg.cache,
                cache_by=arg.cache_by,
                cache_anonymize=arg.anonymize,
                received_from=arg.received_from,
                received_to=arg.received_to,
                wait_cache=arg.wait_cache,
                **overrides,
            )
        if isinstance(arg, Mapping):
            return cls.parse_mapping(arg, **overrides)
        raise TypeError(
            f"SendManyConfig.check_arg expects a SendManyConfig, SendConfig, "
            f"Mapping, or None; got {type(arg).__name__!r}"
        )

    # ------------------------------------------------------------------
    # Instance helpers
    # ------------------------------------------------------------------

    def merge(self, **overrides: Any) -> "SendManyConfig":
        """Return a new :class:`SendManyConfig` with *overrides* applied.

        All fields not listed in *overrides* retain their current values.

        Parameters
        ----------
        **overrides:
            Fields to replace.  Unknown field names raise :exc:`TypeError`.

        Returns
        -------
        SendManyConfig

        Raises
        ------
        TypeError
            If any key in *overrides* is not a recognised field name.

        Examples
        --------
        >>> cfg = SendManyConfig(batch_size=50)
        >>> cfg.merge(batch_size=100, ordered=True)
        SendManyConfig(..., batch_size=100, ordered=True, ...)
        """
        unknown = set(overrides) - _SEND_MANY_CONFIG_FIELDS
        if unknown:
            raise TypeError(
                f"SendManyConfig.merge got unexpected field(s): {sorted(unknown)!r}"
            )
        return dataclasses.replace(self, **overrides)

    def to_send_config(self) -> SendConfig:
        """Return a :class:`SendConfig` built from the shared fields.

        Useful when delegating from :meth:`send_many` to :meth:`send` for
        each cache miss.  The ``cache`` field is intentionally set to ``None``
        because individual miss dispatches never write to the cache directly.

        Returns
        -------
        SendConfig

        Examples
        --------
        >>> many = SendManyConfig(stream=False, raise_error=False, batch_size=10)
        >>> many.to_send_config()
        SendConfig(wait=None, raise_error=False, stream=False, ...)
        """
        return SendConfig(
            wait=self.wait,
            raise_error=self.raise_error,
            stream=self.stream,
            cache=None,
            cache_by=self.cache_by,
            anonymize=self.cache_anonymize,
            received_from=self.received_from,
            received_to=self.received_to,
            wait_cache=self.wait_cache,
        )



# ---------------------------------------------------------------------------
# SendConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SendConfig:
    """Configuration for a single HTTP request.

    Pass an instance as ``config=`` to
    :meth:`~yggdrasil.io.session.Session.send` or any of the convenience
    HTTP methods (``get``, ``post``, …).  Individual keyword arguments on the
    call site override the corresponding field.

    Parameters
    ----------
    wait:
        Waiting / retry configuration forwarded to the transport layer.
        Accepts anything that :class:`~yggdrasil.dataclasses.waiting.WaitingConfig`
        understands: a :class:`WaitingConfig` instance, a dict, a timeout in
        seconds (``int`` or ``float``), a deadline :class:`datetime.datetime`,
        or ``True`` / ``False``.  ``None`` defers to the session default.
    raise_error:
        When ``True`` (default) a non-2xx response raises immediately.
        Set to ``False`` to inspect error responses manually.
    stream:
        When ``True`` (default) the response body is streamed lazily.
        Set to ``False`` to buffer the entire body before returning.
    cache:
        Optional Databricks Delta table used to cache responses.  When
        provided the session checks the table before hitting the network and
        writes new responses back on success.
    cache_by:
        List of response-schema column names that form the cache key.
        Defaults to ``["request_method", "request_url_host",
        "request_url_path", "request_url_query",
        "request_content_length", "request_body_hash"]`` when ``cache``
        is set.  Must be a subset of
        :data:`~yggdrasil.io.response.RESPONSE_ARROW_SCHEMA` field names.
    anonymize:
        Controls how sensitive fields are removed before caching or
        comparing cached entries.

        * ``"remove"`` – strip values completely (default).
        * ``"redact"`` – replace values with a fixed placeholder.
    received_from:
        Earliest ``response_received_at`` timestamp to accept from cache.
        Any timezone-aware datetime, date, or ISO-8601 string is accepted.
    received_to:
        Latest ``response_received_at`` timestamp to accept from cache.
    wait_cache:
        Waiting config for the asynchronous cache-write operation.
        ``False`` (default) means fire-and-forget.
    """

    wait: WaitingConfigArg = None
    raise_error: bool = True
    stream: bool = True
    cache: Optional["Table"] = field(default=None, hash=False, compare=False)
    cache_by: Optional[list[str]] = field(default=None, hash=False, compare=False)
    anonymize: Literal["remove", "redact"] = "remove"
    received_from: Optional[dt.datetime | dt.date | str] = None
    received_to: Optional[dt.datetime | dt.date | str] = None
    wait_cache: WaitingConfigArg = False

    @classmethod
    def default(cls) -> "SendConfig":
        """Return the default :class:`SendConfig` (all fields at their defaults)."""
        return cls()


# ---------------------------------------------------------------------------
# SendManyConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SendManyConfig:
    """Configuration for concurrent batched HTTP requests.

    Pass an instance as ``config=`` to
    :meth:`~yggdrasil.io.session.Session.send_many`.  Individual keyword
    arguments on the call site override the corresponding field.

    All :class:`SendConfig` fields are duplicated here (rather than
    inheriting) to preserve ``frozen=True`` and ``slots=True`` semantics,
    and to allow independent defaults for the many-request path.

    Parameters
    ----------
    wait:
        See :attr:`SendConfig.wait`.
    raise_error:
        See :attr:`SendConfig.raise_error`.
    stream:
        See :attr:`SendConfig.stream`.
    cache:
        See :attr:`SendConfig.cache`.
    cache_by:
        See :attr:`SendConfig.cache_by`.
    cache_anonymize:
        Controls how sensitive fields are stripped before the cache lookup
        and before writing new entries.  Uses the same
        ``"remove"`` / ``"redact"`` semantics as :attr:`SendConfig.anonymize`.
    received_from:
        See :attr:`SendConfig.received_from`.
    received_to:
        See :attr:`SendConfig.received_to`.
    wait_cache:
        See :attr:`SendConfig.wait_cache`.
    normalize:
        When ``True`` URLs are normalised before the request is dispatched.
        ``None`` (default) normalises automatically when ``cache`` is set.
    batch_size:
        Number of requests per cache-lookup batch.  ``None`` defaults to
        ``pool_maxsize × 100``.  Has no effect when ``cache`` is ``None``.
    ordered:
        When ``True`` responses are yielded in the same order as the input
        requests.  ``False`` (default) yields in completion order for higher
        throughput.
    max_in_flight:
        Maximum number of concurrent in-flight requests in the thread pool.
        ``None`` uses the pool's natural concurrency limit.
    """

    # --- shared with SendConfig ---
    wait: WaitingConfigArg = None
    raise_error: bool = True
    stream: bool = True
    cache: Optional["Table"] = field(default=None, hash=False, compare=False)
    cache_by: Optional[list[str]] = field(default=None, hash=False, compare=False)
    cache_anonymize: Literal["remove", "redact"] = "remove"
    received_from: Optional[dt.datetime | dt.date | str] = None
    received_to: Optional[dt.datetime | dt.date | str] = None
    wait_cache: WaitingConfigArg = False

    # --- send_many-specific ---
    normalize: Optional[bool] = None
    batch_size: Optional[int] = None
    ordered: bool = False
    max_in_flight: Optional[int] = None

    @classmethod
    def default(cls) -> "SendManyConfig":
        """Return the default :class:`SendManyConfig` (all fields at their defaults)."""
        return cls()

    def to_send_config(self) -> SendConfig:
        """Return a :class:`SendConfig` built from the shared fields.

        Useful when delegating from :meth:`send_many` to :meth:`send` for
        each cache miss.
        """
        return SendConfig(
            wait=self.wait,
            raise_error=self.raise_error,
            stream=self.stream,
            cache=None,  # individual sends never write to cache directly
            cache_by=self.cache_by,
            anonymize=self.cache_anonymize,
            received_from=self.received_from,
            received_to=self.received_to,
            wait_cache=self.wait_cache,
        )

