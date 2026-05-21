"""Process parameter snapshot from sys.argv, env vars, and Databricks notebook kwargs.

:class:`SystemParameters` is a ``dict[str, str]`` populated from every channel
the runtime exposes:

* ``sys.argv[1:]`` — ``--key=value`` / ``--key value`` / ``--flag`` pairs.
  Positional tokens (no ``--`` prefix) land on :attr:`SystemParameters.args`.
* Databricks notebook bindings — the union of ``dbutils.widgets`` values and
  ``{{job.parameters.*}}`` substitutions, read via
  ``dbutils.notebook.entry_point.getCurrentBindings()``.
* ``os.environ`` — filtered by prefix when the caller asks for it.

Precedence on collision (highest wins): sys.argv > Databricks bindings > env.
An explicit ``--key=cli`` on the command line means the caller is overriding
the job-parameter binding for this run.
"""
from __future__ import annotations

import builtins
import logging
import os
import sys
from typing import Any, Iterable, Mapping

__all__ = ["SystemParameters"]

LOGGER = logging.getLogger(__name__)


class SystemParameters(dict):
    """Process parameter snapshot, merged from every available channel.

    Behaves as a ``dict[str, str]`` for ``--key=value`` / widget / job-parameter
    bindings. Positional ``sys.argv`` tokens are kept on :attr:`args`.

    Build via the ``from_*`` constructors — :meth:`from_argv`,
    :meth:`from_dbutils`, :meth:`from_environ` — or :meth:`current` to
    auto-fetch from every channel.
    """

    args: tuple[str, ...]

    def __init__(
        self,
        mapping: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        *,
        args: Iterable[str] = (),
        **kwargs: Any,
    ) -> None:
        if mapping is None:
            super().__init__(**kwargs)
        else:
            super().__init__(mapping, **kwargs)
        self.args = tuple(args)

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------

    @classmethod
    def from_(cls, value: Any = ...) -> SystemParameters:
        """Generic dispatch — route by input shape to the right constructor.

        * ``...`` / ``None`` → :meth:`current` (auto-fetch).
        * existing ``SystemParameters`` → identity.
        * ``Mapping`` → wrap as bindings.
        * ``list`` / ``tuple`` of strings → parse as argv.
        """
        if value is ... or value is None:
            return cls.current()
        if isinstance(value, SystemParameters):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, (list, tuple)):
            return cls.from_argv(list(value))
        raise TypeError(
            f"Cannot build SystemParameters from {type(value).__name__!r}: "
            f"expected ..., None, Mapping, list[str], or SystemParameters."
        )

    @classmethod
    def current(
        cls,
        *,
        argv: list[str] | None = None,
        env_prefix: str | tuple[str, ...] = (),
    ) -> SystemParameters:
        """Auto-fetch the merged snapshot from argv + Databricks + (optional) env.

        Layers (lowest precedence first):

        1. ``os.environ`` filtered by *env_prefix* (skipped when empty).
        2. Databricks notebook bindings (silently empty outside Databricks).
        3. ``sys.argv[1:]`` ``--key=value`` pairs (or *argv* when supplied).

        Positional argv tokens are surfaced on :attr:`args`. Pass *argv*
        explicitly in tests so the snapshot doesn't pick up the test
        harness's own command line.
        """
        merged: dict[str, str] = {}

        prefixes: tuple[str, ...] = (
            (env_prefix,) if isinstance(env_prefix, str) and env_prefix else
            tuple(env_prefix) if not isinstance(env_prefix, str) else ()
        )
        if prefixes:
            for k, v in os.environ.items():
                if any(k.startswith(p) for p in prefixes):
                    merged[k] = v

        merged.update(cls._read_dbutils_bindings())

        kwargs, positional = cls._parse_argv(sys.argv[1:] if argv is None else list(argv))
        merged.update(kwargs)

        return cls(merged, args=positional)

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> SystemParameters:
        """Parse ``--key=value`` / ``--key value`` / ``--flag`` pairs out of *argv*.

        *argv* defaults to ``sys.argv[1:]``. Recognized shapes:

        * ``--key=value`` — single token, split on first ``=``.
        * ``--key value`` — two tokens; value must not itself start with ``--``.
        * ``--flag`` — bare flag, stored as the string ``"true"`` so a
          ``bool``-annotated parameter coerces cleanly.

        Bare positional tokens land on :attr:`args` in input order.
        """
        kwargs, positional = cls._parse_argv(
            sys.argv[1:] if argv is None else list(argv)
        )
        return cls(kwargs, args=positional)

    @classmethod
    def from_dbutils(cls, *names: str) -> SystemParameters:
        """Read Databricks notebook widget bindings via ``dbutils``.

        With no *names*: the full union from
        ``dbutils.notebook.entry_point.getCurrentBindings()`` (widget values +
        ``{{job.parameters.*}}`` substitutions). With *names*: only those
        widgets via ``dbutils.widgets.get(name)``.

        Raises :class:`RuntimeError` when ``dbutils`` is not available so the
        miss is loud — use :meth:`current` for the silent-fallback shape.
        """
        dbutils = cls._get_dbutils()
        if dbutils is None:
            raise RuntimeError(
                "SystemParameters.from_dbutils: dbutils is not available — "
                "this constructor only works inside a Databricks runtime. "
                "Use SystemParameters.from_argv() for command-line parameters "
                "or SystemParameters.current() for a silent multi-channel fetch."
            )
        if names:
            return cls({n: dbutils.widgets.get(n) for n in names})
        return cls(cls._read_dbutils_bindings(dbutils))

    @classmethod
    def from_environ(cls, *prefixes: str) -> SystemParameters:
        """Snapshot ``os.environ``, optionally filtered to keys starting with *prefixes*."""
        if not prefixes:
            return cls(dict(os.environ))
        return cls({
            k: v for k, v in os.environ.items()
            if any(k.startswith(p) for p in prefixes)
        })

    # ---------------------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------------------

    def as_dict(self) -> dict[str, str]:
        """Return a plain ``dict`` copy of the bindings (drops :attr:`args`)."""
        return dict(self)

    def __repr__(self) -> str:
        return f"SystemParameters({dict.__repr__(self)}, args={self.args!r})"

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
    def _get_dbutils() -> Any | None:
        """Locate a live ``dbutils`` instance, or return ``None``.

        Probes the same injection points Databricks itself uses:
        ``builtins.dbutils``, then the IPython user namespace. Intentionally
        self-contained so the environ module stays free of the
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
    def _read_dbutils_bindings(cls, dbutils: Any = None) -> dict[str, str]:
        """Return ``getCurrentBindings()`` as a string dict, or ``{}`` on miss.

        Silent on every failure path — outside Databricks, inside a
        ``SparkPythonTask`` (no notebook entry point), or when the Py4J
        bridge raises. Callers that need a loud miss should go through
        :meth:`from_dbutils` instead.
        """
        if dbutils is None:
            dbutils = cls._get_dbutils()
        if dbutils is None:
            return {}
        try:
            bindings = dbutils.notebook.entry_point.getCurrentBindings()
        except Exception:
            return {}
        if bindings is None:
            return {}
        # getCurrentBindings hands back a Py4J Java Map; in-process fakes
        # / unit tests can hand back a plain dict.
        if hasattr(bindings, "keySet"):
            return {str(k): str(bindings.get(k)) for k in bindings.keySet()}
        return {str(k): str(v) for k, v in dict(bindings).items()}
