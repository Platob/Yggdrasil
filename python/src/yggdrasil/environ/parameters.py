"""Lazy, type-aware process parameter mapping.

:class:`SystemParameters` is a lazy ``Mapping[str, Any]`` over every channel
a runtime exposes to a process:

* ``sys.argv[1:]`` — ``--key=value`` / ``--key value`` / ``--flag`` pairs.
  Positional tokens (no ``--`` prefix) land on :attr:`SystemParameters.args`.
* Databricks notebook bindings — the union of ``dbutils.widgets`` values and
  ``{{job.parameters.*}}`` substitutions via
  ``dbutils.notebook.entry_point.getCurrentBindings()``. Probed **lazily**:
  ``dbutils`` is not touched until a key actually needs it.
* ``os.environ`` — filtered by prefix when the caller asks for it.

Precedence on collision (highest wins): explicit overrides > sys.argv >
Databricks bindings > env.

Typed config via subclassing
----------------------------

Annotate fields on a subclass to get value casting through
:func:`yggdrasil.data.cast.convert` and attribute access::

    class Config(SystemParameters):
        count: int = 1
        name: str = "default"
        verbose: bool = False

    cfg = Config()         # auto-fetches from every channel
    cfg.count              # int(42) from --count=42 / widget / env
    cfg["name"]            # "alice"
    cfg.verbose            # bool, "true"/"false" coerced

Undeclared keys come back as the raw source value (string from argv / env /
widgets). Cast results are cached per-key for the lifetime of the instance.
"""
from __future__ import annotations

import builtins
import datetime as dt
import logging
import os
import sys
from collections.abc import Mapping as MappingABC
from enum import Enum
from typing import Any, ClassVar, Iterator, Mapping, get_args, get_origin, get_type_hints

from yggdrasil.data.cast import convert

__all__ = ["SystemParameters", "WidgetType", "ALL_VALUES_TAG", "LABEL_ACRONYMS", "nice_label"]

#: Sentinel surfaced on Databricks ``multiselect`` widgets when the user
#: hasn't picked anything — split out of the resolved value so callers get
#: an empty list instead of a list containing the literal ``"**all**"``.
ALL_VALUES_TAG = "**all**"

#: Tokens that stay upper-case when :func:`nice_label` prettifies a snake_case
#: field name into a Databricks widget label. Domain acronyms the codebase
#: actually uses — currency / standards / protocols / Databricks / data shapes.
#: Override per subclass by setting :attr:`SystemParameters._LABEL_ACRONYMS`.
LABEL_ACRONYMS: frozenset[str] = frozenset({
    # Time / dates
    "UTC", "TZ", "IANA", "ISO", "DST",
    # Currency / finance
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY", "FX", "OHLCV",
    # Identifiers / standards
    "ID", "UUID", "URL", "URI", "EIC", "MIC", "PK", "FK",
    # Protocols / formats
    "HTTP", "HTTPS", "JSON", "XML", "CSV", "PDF", "SQL", "IO", "TLS", "SSL",
    "TCP", "UDP", "FTP", "SFTP", "SMTP", "DNS", "IP",
    # Cloud / platform
    "AWS", "GCP", "S3", "GCS", "EC2", "IAM", "VPC", "DBFS", "UC",
    # Tech / ML
    "API", "SDK", "CLI", "UI", "DB", "ML", "AI", "BI", "KPI", "GPU", "CPU",
    # Org / energy
    "ENTSO",
})


def nice_label(name: str) -> str:
    """Prettify a snake_case identifier into a Title Case widget label.

    Splits on ``_`` / ``-``, title-cases each piece, and keeps tokens in
    :data:`LABEL_ACRONYMS` upper-case. Empty / all-separator input round-trips.

    Examples::

        nice_label("start_date_utc")    # "Start Date UTC"
        nice_label("user_id")           # "User ID"
        nice_label("api_url")           # "API URL"
        nice_label("bidding_zone_eic")  # "Bidding Zone EIC"
        nice_label("verbose")           # "Verbose"
    """
    parts = [p for p in name.replace("-", "_").split("_") if p]
    if not parts:
        return name
    out: list[str] = []
    for part in parts:
        upper = part.upper()
        if upper in LABEL_ACRONYMS:
            out.append(upper)
        else:
            out.append(part[:1].upper() + part[1:].lower())
    return " ".join(out)


class WidgetType(Enum):
    """Databricks notebook widget kinds used by :meth:`SystemParameters.init_widgets`."""

    TEXT = "text"
    DROPDOWN = "dropdown"
    COMBOBOX = "combobox"
    MULTISELECT = "multiselect"
    DATETIME = "datetime"  # text widget under the hood, dt.* formatted

LOGGER = logging.getLogger(__name__)

# Use Ellipsis as the "unset" sentinel per project conventions.
_UNSET: Any = ...


class _FieldDescriptor:
    """Per-field data descriptor — routes ``cls.name`` through the lazy resolver.

    Installed by :meth:`SystemParameters.__init_subclass__` for each annotated
    attribute on a subclass. ``__set__`` makes it a *data* descriptor so it
    takes precedence over instance attributes, preventing accidental shadowing.
    """

    __slots__ = ("name", "type", "default")

    def __init__(self, name: str, type_: Any, default: Any) -> None:
        self.name = name
        self.type = type_
        self.default = default

    def __get__(self, instance: Any, owner: type) -> Any:
        if instance is None:
            return self
        return instance._resolve(self.name, self.type, self.default)

    def __set__(self, instance: Any, value: Any) -> None:
        instance._explicit[self.name] = value
        instance._cast_cache.pop(self.name, None)


class SystemParameters(MappingABC):
    """Lazy ``Mapping[str, Any]`` over sys.argv, Databricks bindings, and env.

    Build via the ``from_*`` constructors — :meth:`from_argv`,
    :meth:`from_dbutils`, :meth:`from_environ` — or instantiate directly to
    auto-fetch from every channel. Subclass and annotate fields to get typed
    attribute access with cast-through-``convert``.
    """

    _declared_fields: ClassVar[dict[str, tuple[Any, Any]]] = {}

    args: tuple[str, ...]

    # ---------------------------------------------------------------------
    # Subclass schema capture
    # ---------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Inherit parent declarations (subclasses extend, not replace).
        declared: dict[str, tuple[Any, Any]] = dict(getattr(cls, "_declared_fields", {}))
        # Resolve forward-reference annotations to real types — necessary
        # under ``from __future__ import annotations`` where every hint is
        # stored as a string. Restrict to *this class's* annotations so
        # parent fields stay in the inherited ``_declared_fields`` slot.
        own_annotation_names = set(cls.__dict__.get("__annotations__", {}))
        try:
            resolved_hints = get_type_hints(cls)
        except Exception:
            resolved_hints = cls.__dict__.get("__annotations__", {})
        own_annotations = {
            name: resolved_hints.get(name, cls.__dict__["__annotations__"][name])
            for name in own_annotation_names
        }
        for name, type_ in own_annotations.items():
            if name.startswith("_") or name == "args":
                continue
            default = cls.__dict__.get(name, _UNSET)
            if isinstance(default, (classmethod, staticmethod, property)):
                continue
            declared[name] = (type_, default)
            setattr(cls, name, _FieldDescriptor(name, type_, default))
        cls._declared_fields = declared

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------

    def __init__(
        self,
        mapping: Mapping[str, Any] | None = None,
        *,
        argv: list[str] | None = _UNSET,
        env_prefix: str | tuple[str, ...] = (),
        dbutils: Any = _UNSET,
        **kwargs: Any,
    ) -> None:
        """Capture source configuration; nothing is fetched until first access.

        Args:
            mapping: Highest-precedence explicit overrides. Merged with *kwargs*.
            argv: ``...`` (default) → ``sys.argv[1:]``; ``None`` → skip argv;
                a ``list[str]`` → parse those tokens. Argv parsing is eager
                because reading the list is essentially free.
            env_prefix: Empty (default) → ignore env. Pass a prefix string or
                tuple to expose ``os.environ`` keys with those prefixes.
            dbutils: ``...`` (default) → auto-detect via ``builtins.dbutils``
                / IPython on first access; ``None`` → skip Databricks
                bindings; a live handle → use it directly.
            **kwargs: Convenience for ad-hoc explicit overrides.
        """
        self._explicit: dict[str, Any] = {}
        if mapping is not None:
            self._explicit.update(mapping)
        self._explicit.update(kwargs)

        if argv is None:
            self._argv_kwargs: dict[str, str] = {}
            self.args = ()
        else:
            raw = sys.argv[1:] if argv is _UNSET else list(argv)
            self._argv_kwargs, positional = self._parse_argv(raw)
            self.args = tuple(positional)

        if isinstance(env_prefix, str):
            self._env_prefixes: tuple[str, ...] = (env_prefix,) if env_prefix else ()
        else:
            self._env_prefixes = tuple(env_prefix)

        # dbutils — lazy. _UNSET = auto-detect on demand, None = skip.
        self._dbutils_arg: Any = dbutils
        self._dbutils_resolved: Any = _UNSET
        self._dbutils_bindings_cache: dict[str, str] | None = None

        # Cast result cache: declared-field reads memoize here.
        self._cast_cache: dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # Mapping interface (lazy)
    # ---------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        if key in self._declared_fields:
            type_, default = self._declared_fields[key]
            return self._resolve(key, type_, default)
        value = self._lookup_raw(key)
        if value is _UNSET:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._all_keys())

    def __len__(self) -> int:
        return len(self._all_keys())

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._declared_fields:
            return True
        return self._lookup_raw(key) is not _UNSET

    # ---------------------------------------------------------------------
    # Resolution
    # ---------------------------------------------------------------------

    def _resolve(self, name: str, type_: Any, default: Any) -> Any:
        """Look up *name* across sources, cast to *type_*, fall back to *default*.

        Cached per-key. Casting routes through
        :func:`yggdrasil.data.cast.convert` — declared annotations get coerced
        from string to int / bool / datetime / enum / dataclass / etc.
        """
        if name in self._cast_cache:
            return self._cast_cache[name]
        raw = self._lookup_raw(name)
        if raw is _UNSET:
            if default is _UNSET:
                raise KeyError(name)
            result = default
        elif type_ is Any or type_ is None:
            result = raw
        else:
            # CSV split for iterable target types — Databricks multiselect
            # widgets and ``--names=a,b,c`` argv tokens hand back one string
            # the cast registry would reject (it bans ``str`` → ``list``).
            # Drop the ``**all**`` placeholder dbutils emits when nothing is
            # selected, and drop empty parts: an unselected multiselect / a
            # bare ``--content_ids=`` hands back ``""``, whose ``split(",")``
            # is ``[""]`` — left in, that single empty token casts to a
            # spurious element (``tuple[int, ...]`` → ``(0,)``) instead of the
            # empty collection the caller meant.
            if isinstance(raw, str) and self._is_iterable_type(type_):
                raw = [
                    part
                    for part in (p.strip() for p in raw.split(","))
                    if part and part != ALL_VALUES_TAG
                ]
            try:
                result = convert(raw, type_)
            except Exception as exc:
                raise ValueError(
                    f"SystemParameters: cannot cast {name!r}={raw!r} (type "
                    f"{type(raw).__name__}) to declared type {type_!r}. "
                    f"Source returned the raw value; the cast registry "
                    f"rejected it. Check the annotation or fix the input."
                ) from exc
        self._cast_cache[name] = result
        return result

    @staticmethod
    def _is_iterable_type(type_: Any) -> bool:
        """Return ``True`` when *type_* is a list/set-shaped annotation.

        Handles bare ``list`` / ``set``, parameterised ``list[int]`` /
        ``set[str]``, and nested ``Optional[list[...]]`` via origin lookup.
        ``str`` / ``bytes`` are *not* iterable for this purpose — we don't
        want a declared ``str`` field to be split on commas.
        """
        if type_ in (list, set, tuple, frozenset):
            return True
        origin = get_origin(type_)
        if origin is None:
            return False
        return origin in (list, set, tuple, frozenset)

    def _lookup_raw(self, key: str) -> Any:
        """Resolve raw value across the source stack. Returns ``...`` on miss.

        Precedence (highest first): explicit overrides → sys.argv → Databricks
        bindings → env (filtered by *env_prefix*).
        """
        if key in self._explicit:
            return self._explicit[key]
        if key in self._argv_kwargs:
            return self._argv_kwargs[key]
        bindings = self._dbutils_bindings()
        if key in bindings:
            return bindings[key]
        if self._env_prefixes and any(key.startswith(p) for p in self._env_prefixes):
            env_value = os.environ.get(key)
            if env_value is not None:
                return env_value
        return _UNSET

    def _all_keys(self) -> set[str]:
        keys: set[str] = set(self._declared_fields)
        keys.update(self._explicit)
        keys.update(self._argv_kwargs)
        keys.update(self._dbutils_bindings())
        if self._env_prefixes:
            for k in os.environ:
                if any(k.startswith(p) for p in self._env_prefixes):
                    keys.add(k)
        return keys

    # ---------------------------------------------------------------------
    # Lazy dbutils access
    # ---------------------------------------------------------------------

    def _dbutils(self) -> Any:
        """Resolve the live ``dbutils`` handle once; cached afterwards."""
        if self._dbutils_resolved is not _UNSET:
            return self._dbutils_resolved
        if self._dbutils_arg is None:
            self._dbutils_resolved = None
        elif self._dbutils_arg is _UNSET:
            self._dbutils_resolved = self._get_dbutils()
        else:
            self._dbutils_resolved = self._dbutils_arg
        return self._dbutils_resolved

    def _dbutils_bindings(self) -> dict[str, str]:
        if self._dbutils_bindings_cache is not None:
            return self._dbutils_bindings_cache
        self._dbutils_bindings_cache = self._read_dbutils_bindings(self._dbutils())
        return self._dbutils_bindings_cache

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------

    @classmethod
    def from_(cls, value: Any = _UNSET) -> SystemParameters:
        """Generic dispatch — route by input shape to the right constructor.

        * ``...`` / ``None`` → fresh instance (auto-fetch from every channel).
        * existing ``SystemParameters`` → identity.
        * ``Mapping`` → explicit-only (argv + dbutils + env skipped).
        * ``list`` / ``tuple`` of strings → :meth:`from_argv`.
        """
        if value is _UNSET or value is None:
            return cls()
        if isinstance(value, SystemParameters):
            return value
        if isinstance(value, MappingABC):
            return cls(value, argv=None, dbutils=None)
        if isinstance(value, (list, tuple)):
            return cls.from_argv(list(value))
        raise TypeError(
            f"Cannot build SystemParameters from {type(value).__name__!r}: "
            f"expected ..., None, Mapping, list[str], or SystemParameters."
        )

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> SystemParameters:
        """Build from argv only — skips Databricks and env."""
        return cls(argv=sys.argv[1:] if argv is None else list(argv), dbutils=None)

    @classmethod
    def from_dbutils(cls, *names: str) -> SystemParameters:
        """Read Databricks notebook widget bindings via ``dbutils``.

        With no *names*: the full union from
        ``dbutils.notebook.entry_point.getCurrentBindings()``. With *names*:
        only those widgets via ``dbutils.widgets.get(name)``.

        Raises :class:`RuntimeError` when ``dbutils`` is not available so the
        miss is loud — instantiate :class:`SystemParameters` directly for the
        silent-fallback shape.
        """
        dbutils = cls._get_dbutils()
        if dbutils is None:
            raise RuntimeError(
                "SystemParameters.from_dbutils: dbutils is not available — "
                "this constructor only works inside a Databricks runtime. "
                "Use SystemParameters.from_argv() for command-line parameters "
                "or instantiate SystemParameters() for a silent multi-channel fetch."
            )
        if names:
            return cls(
                {n: dbutils.widgets.get(n) for n in names},
                argv=None,
                dbutils=None,
            )
        return cls(argv=None, dbutils=dbutils)

    @classmethod
    def from_environ(cls, *prefixes: str) -> SystemParameters:
        """Snapshot ``os.environ``, optionally filtered by key prefix."""
        if not prefixes:
            return cls(dict(os.environ), argv=None, dbutils=None)
        return cls(argv=None, dbutils=None, env_prefix=prefixes)

    @classmethod
    def from_environment(cls) -> SystemParameters:
        """Auto-fetch from every channel — alias for ``cls()``.

        Kept as the canonical entry point for the historical
        ``NotebookConfig.from_environment()`` shape.
        """
        return cls()

    # ---------------------------------------------------------------------
    # Databricks widget surface
    # ---------------------------------------------------------------------

    @classmethod
    def init_widgets(cls, *, skip_existing: bool = True) -> None:
        """Create a Databricks notebook widget for each declared field.

        Resolves the widget shape from the field's annotation:
        ``bool`` → dropdown(``"true"`` / ``"false"``), ``Enum`` →
        dropdown over enum values, ``list`` / ``set`` → multiselect,
        :class:`datetime.datetime` / :class:`datetime.date` → text widget
        with ISO 8601 default, everything else → text widget.

        Silent no-op outside a Databricks notebook (no ``dbutils``).
        Pass ``skip_existing=False`` to recreate widgets already present.
        """
        dbutils = cls._get_dbutils()
        if dbutils is None or not hasattr(dbutils, "widgets"):
            LOGGER.info(
                "SystemParameters.init_widgets: dbutils.widgets unavailable — "
                "widgets are only created inside a Databricks notebook"
            )
            return

        for name, (type_, default) in cls._declared_fields.items():
            if skip_existing and cls._widget_exists(dbutils, name):
                continue
            widget_type = cls._determine_widget_type(type_)
            cls._create_widget(dbutils, name, type_, default, widget_type)

    @classmethod
    def init_job(
        cls,
        logging: bool | int | None = logging.INFO,
    ) -> SystemParameters:
        """Initialize widgets, tweak the active Spark session, return the populated config.

        Mirrors the historical ``NotebookConfig.init_job()`` entry point.
        Spark tweaks are silently skipped when PySpark isn't importable or
        no session is active.

        ``logging`` controls runtime log activation on the ``yggdrasil``
        logger:

        * ``int`` (default ``logging.INFO``) — set the level to that
          numeric value;
        * ``True`` — alias for ``logging.INFO``;
        * ``False`` / ``None`` — leave the logger untouched.

        A :class:`logging.StreamHandler` is attached only when nothing
        upstream is already going to render records (checked via
        :meth:`logging.Logger.hasHandlers`, which walks the propagation
        chain). The Databricks job runtime (and pytest harnesses) usually
        wire the root logger at startup; in those cases propagation alone
        carries the messages to the existing handler, so adding our own
        would double-log.
        """
        import logging as _logging

        cls.init_widgets()

        if logging is not None and logging is not False:
            level = _logging.INFO if logging is True else int(logging)
            ygg_logger = _logging.getLogger("yggdrasil")
            ygg_logger.setLevel(level)
            if not ygg_logger.hasHandlers():
                handler = _logging.StreamHandler()
                handler.setFormatter(_logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s"
                ))
                ygg_logger.addHandler(handler)

        try:
            from pyspark.sql import SparkSession
        except ImportError:
            return cls()
        try:
            spark = SparkSession.getActiveSession()
        except Exception:
            spark = None
        if spark is not None:
            try:
                spark.conf.set("spark.sql.session.timeZone", "UTC")
            except Exception:
                pass
        return cls()

    @staticmethod
    def _widget_exists(dbutils: Any, name: str) -> bool:
        """``True`` when *name* is already wired on the notebook's widget panel."""
        if not hasattr(dbutils, "widgets"):
            return False
        try:
            dbutils.widgets.get(name)
            return True
        except Exception:
            return False

    @staticmethod
    def _determine_widget_type(field_type: Any) -> WidgetType:
        """Resolve the widget shape for a declared annotation."""
        if field_type is dt.datetime or field_type is dt.date:
            return WidgetType.DATETIME
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            return WidgetType.DROPDOWN
        if field_type is bool:
            return WidgetType.DROPDOWN
        if SystemParameters._is_iterable_type(field_type):
            return WidgetType.MULTISELECT
        return WidgetType.TEXT

    @staticmethod
    def _format_widget_default(value: Any, widget_type: WidgetType) -> str:
        """Render *value* into the string shape the dbutils widget API expects."""
        if value is _UNSET or value is None:
            return ""
        if widget_type is WidgetType.DATETIME:
            if isinstance(value, dt.datetime):
                return value.isoformat()
            if isinstance(value, dt.date):
                return value.strftime("%Y-%m-%d")
        if widget_type is WidgetType.MULTISELECT and isinstance(value, (list, tuple, set, frozenset)):
            return ",".join(
                v.name if isinstance(v, Enum) else str(v)
                for v in value
            )
        if isinstance(value, Enum):
            return value.name
        return str(value)

    @staticmethod
    def _element_enum_type(field_type: Any) -> type[Enum] | None:
        """Extract the ``Enum`` subclass from a parameterized iterable type.

        ``list[Color]`` → ``Color``, ``set[Mode]`` → ``Mode``,
        ``list[str]`` → ``None``.
        """
        args = get_args(field_type)
        if args:
            elem = args[0]
            if isinstance(elem, type) and issubclass(elem, Enum):
                return elem
        return None

    @classmethod
    def _create_widget(
        cls,
        dbutils: Any,
        name: str,
        field_type: Any,
        default: Any,
        widget_type: WidgetType,
    ) -> None:
        """Issue the ``dbutils.widgets.*`` call matching *widget_type*."""
        # Build the option list for dropdown / multiselect widgets.
        if field_type is bool:
            options = ["true", "false"]
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            options = [e.name for e in field_type]
        else:
            elem_enum = cls._element_enum_type(field_type)
            if elem_enum is not None:
                options = [e.name for e in elem_enum]
            elif isinstance(default, (list, set, tuple, frozenset)) and default:
                options = [
                    v.name if isinstance(v, Enum) else str(v)
                    for v in default
                ]
            else:
                options = [cls._format_widget_default(default, widget_type)]
        if not options:
            options = [ALL_VALUES_TAG]

        default_str = cls._format_widget_default(default, widget_type)
        if default_str not in options:
            default_str = options[0]
        label = nice_label(name)
        if widget_type is WidgetType.DROPDOWN:
            dbutils.widgets.dropdown(name, default_str, options, label)
        elif widget_type is WidgetType.COMBOBOX:
            dbutils.widgets.combobox(name, default_str, options, label)
        elif widget_type is WidgetType.MULTISELECT:
            dbutils.widgets.multiselect(name, default_str, options, label)
        else:  # TEXT and DATETIME both use text widgets.
            dbutils.widgets.text(
                name,
                cls._format_widget_default(default, widget_type),
                label,
            )

    # ---------------------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Materialise every known key into a plain ``dict``, applying casts."""
        return {k: self[k] for k in self._all_keys()}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"explicit={self._explicit!r}, argv={self._argv_kwargs!r}, "
            f"args={self.args!r})"
        )

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _parse_argv(argv: list[str]) -> tuple[dict[str, str], list[str]]:
        """Split argv into ``--key=value`` kwargs and positional args."""
        kwargs: dict[str, str] = {}
        positional: list[str] = []
        i = 0
        n = len(argv)
        while i < n:
            token = argv[i]
            if token.startswith("--"):
                tail = token[2:]
                if "=" in tail:
                    key, value = tail.split("=", 1)
                    kwargs[key] = value
                    i += 1
                    continue
                if i + 1 < n and not argv[i + 1].startswith("--"):
                    kwargs[tail] = argv[i + 1]
                    i += 2
                    continue
                kwargs[tail] = "true"
                i += 1
                continue
            positional.append(token)
            i += 1
        return kwargs, positional

    @staticmethod
    def _get_dbutils() -> Any:
        """Locate a live ``dbutils`` instance, or return ``None``.

        Probes ``builtins.dbutils`` then the IPython user namespace. Stays
        self-contained so the ``environ`` module doesn't pull in the
        ``yggdrasil.databricks`` SDK import chain.
        """
        if hasattr(builtins, "dbutils"):
            return builtins.dbutils  # type: ignore[attr-defined]
        try:
            from IPython import get_ipython
        except ImportError:
            return None
        try:
            ip = get_ipython()
        except Exception:
            return None
        if ip is None:
            return None
        user_ns = getattr(ip, "user_ns", None)
        if not user_ns:
            return None
        return user_ns.get("dbutils")

    @classmethod
    def _read_dbutils_bindings(cls, dbutils: Any) -> dict[str, str]:
        """Return ``getCurrentBindings()`` as a string dict, or ``{}`` on miss.

        Silent on every failure path — outside Databricks, inside a
        ``SparkPythonTask`` (no notebook entry point), or when the Py4J
        bridge raises. Callers needing a loud miss go through
        :meth:`from_dbutils` instead.
        """
        if dbutils is None:
            return {}
        try:
            bindings = dbutils.notebook.entry_point.getCurrentBindings()
        except Exception:
            return {}
        if bindings is None:
            return {}
        # getCurrentBindings hands back a Py4J Java Map; in-process fakes /
        # unit tests can hand back a plain dict.
        if hasattr(bindings, "keySet"):
            return {str(k): str(bindings.get(k)) for k in bindings.keySet()}
        return {str(k): str(v) for k, v in dict(bindings).items()}
