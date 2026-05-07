""":class:`SqlTestCase` — engine-aware base for :mod:`yggdrasil.sql` tests.

Mirrors the per-engine TestCase pattern used elsewhere in the
package (:class:`yggdrasil.arrow.tests.ArrowTestCase`,
:class:`yggdrasil.polars.tests.PolarsTestCase`, …):

- Skips on missing optional deps (``sqlglot``, ``polars``).
- Provides a fresh :class:`SqlContext` per test plus convenience
  helpers (``self.register``, ``self.sql``) so tests don't have
  to reach into the process-wide :data:`default_context`.
- Cleans up the per-test context at ``tearDown`` so a leaky
  registration in test A can't poison test B.

Use :class:`SqlPolarsTestCase` (the default) when the test
exercises anything beyond a bare ``SELECT cols FROM src
[WHERE] [LIMIT]``; :class:`SqlArrowTestCase` exercises the
fallback-only path explicitly.
"""

from __future__ import annotations

import unittest
from typing import Any, Mapping

from yggdrasil.arrow.tests import ArrowTestCase

from .catalog import SqlContext
from .lib import has_polars, has_sqlglot


__all__ = ["SqlTestCase", "SqlPolarsTestCase", "SqlArrowTestCase"]


class SqlTestCase(ArrowTestCase):
    """Base class for SQL tests.

    Skips when sqlglot isn't importable. Subclasses tighten the
    skip with their own backend check (``polars``).
    """

    def setUp(self) -> None:
        super().setUp()
        if not has_sqlglot():
            raise unittest.SkipTest(
                "sqlglot is required for yggdrasil.sql tests; "
                "install with `pip install sqlglot` or `pip install ygg[sql]`."
            )
        self.ctx: SqlContext = SqlContext()

    def register(self, name: str, source: Any) -> SqlContext:
        return self.ctx.register(name, source)

    def register_many(self, sources: Mapping[str, Any]) -> SqlContext:
        return self.ctx.register_many(sources)

    def sql(self, query: str, **kwargs: Any):
        from . import sql as _sql

        kwargs.setdefault("context", self.ctx)
        return _sql(query, **kwargs)


class SqlPolarsTestCase(SqlTestCase):
    """SQL tests that need the polars backend (joins / aggregations / …)."""

    def setUp(self) -> None:
        super().setUp()
        if not has_polars():
            raise unittest.SkipTest(
                "polars is required for the SQL polars-backed tests; "
                "install with `pip install polars` or `pip install ygg[data]`."
            )

    def sql(self, query: str, **kwargs: Any):
        from .executor import PolarsSqlExecutor

        kwargs.setdefault("executor", PolarsSqlExecutor(context=self.ctx))
        return super().sql(query, **kwargs)


class SqlArrowTestCase(SqlTestCase):
    """SQL tests that exercise the Arrow-only fallback explicitly."""

    def sql(self, query: str, **kwargs: Any):
        from .executor import ArrowSqlExecutor

        kwargs.setdefault("executor", ArrowSqlExecutor(context=self.ctx))
        return super().sql(query, **kwargs)
