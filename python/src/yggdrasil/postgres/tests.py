"""Unittest base class for Postgres backend tests.

Two orientations:

* **Pure-unit tests** — verify type mapping, SQL string assembly,
  dotted-name parsing, etc., without touching a live database.
  These don't need a Postgres URI; the base class handles missing-
  dependency skips for the optional ``psycopg`` / ``adbc`` modules.

* **Integration tests** — exercise the catalog / schema / table /
  Tabular surface against a live Postgres. Skipped automatically
  when ``POSTGRES_URI`` is unset *or* either driver is missing.
  Decorate the test class with the ``integration`` pytest marker
  (matches the existing Databricks pattern in the repo) so the
  default ``pytest`` run skips them.

Quick start
-----------
::

    from yggdrasil.postgres.tests import PostgresTestCase

    class TestQuoting(PostgresTestCase):
        require_live = False  # pure-unit; doesn't need POSTGRES_URI

        def test_ident(self):
            from yggdrasil.postgres import quote_ident
            self.assertEqual(quote_ident("a"), '"a"')

    @pytest.mark.integration
    class TestRoundtrip(PostgresTestCase):
        def test_roundtrip(self):
            tbl = self.pa.table({"id": [1, 2, 3]})
            t = self.engine.table("public.t").create(tbl.schema)
            t.write_arrow_table(tbl)
            out = t.read_arrow_table()
            self.assertEqual(out.num_rows, 3)
"""

from __future__ import annotations

import logging
import os
import unittest
import uuid
from typing import Any, ClassVar, Optional

from yggdrasil.lazy_imports import has_adbc, has_psycopg

LOGGER = logging.getLogger(__name__)

__all__ = ["PostgresTestCase"]


_LIVE_URI_ENV = "POSTGRES_URI"


class PostgresTestCase(unittest.TestCase):
    """Base class for Postgres backend tests.

    Class attributes
    ----------------
    require_live
        Skip the class entirely when no live database is reachable
        (``POSTGRES_URI`` unset or psycopg missing). Default
        ``True``; flip to ``False`` for pure-unit tests that only
        exercise string / type mapping helpers.
    require_adbc
        Also skip when the ADBC driver is unavailable. Default
        ``True`` — the postgres backend's tabular I/O assumes ADBC
        for full functionality, so a live test without it would be
        misleading.
    test_schema_prefix
        Prefix used for the per-test transient schema. Each test
        gets its own ``{prefix}_{uuid4}`` schema, dropped on
        ``tearDown``, so parallel runs don't collide.
    """

    require_live: ClassVar[bool] = True
    require_adbc: ClassVar[bool] = True
    test_schema_prefix: ClassVar[str] = "ygg_test"

    pa: ClassVar[Any]
    engine: Optional[Any] = None  # PostgresEngine
    test_schema_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            import pyarrow as _pa
        except ImportError as exc:  # pragma: no cover — pyarrow is a hard dep
            raise unittest.SkipTest(f"pyarrow unavailable: {exc}") from exc
        cls.pa = _pa

        if not cls.require_live:
            return

        if not has_psycopg():
            raise unittest.SkipTest(
                "psycopg (psycopg 3) is not installed; install with "
                "`pip install ygg[postgres]`."
            )
        if cls.require_adbc and not has_adbc():
            raise unittest.SkipTest(
                "adbc-driver-postgresql is not installed; install with "
                "`pip install ygg[postgres]`."
            )

        uri = os.environ.get(_LIVE_URI_ENV)
        if not uri:
            raise unittest.SkipTest(
                f"Set {_LIVE_URI_ENV} to run live Postgres integration tests."
            )

        from .engine import PostgresEngine
        cls.engine = PostgresEngine(uri)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.engine is not None:
            try:
                cls.engine.connection.close()
            except Exception:
                LOGGER.exception("Closing engine connection failed; continuing.")
            cls.engine = None
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        if self.engine is None:
            return
        # Per-test scratch schema — parallel-safe.
        self.test_schema_name = f"{self.test_schema_prefix}_{uuid.uuid4().hex[:12]}"
        self.engine.schema(self.test_schema_name).create(missing_ok=True)

    def tearDown(self) -> None:
        if self.engine is not None and self.test_schema_name:
            try:
                self.engine.schema(self.test_schema_name).delete(
                    if_exists=True, cascade=True,
                )
            except Exception:
                LOGGER.exception(
                    "Dropping per-test schema %r failed; leaving for cleanup.",
                    self.test_schema_name,
                )
            self.test_schema_name = None
        super().tearDown()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def table(self, name: str):
        """Resolve a :class:`Table` inside the per-test scratch schema."""
        if self.engine is None or self.test_schema_name is None:
            raise RuntimeError(
                "PostgresTestCase has no live engine; set require_live=False "
                "for unit tests or POSTGRES_URI for integration tests."
            )
        return self.engine.table(
            schema_name=self.test_schema_name,
            table_name=name,
        )
