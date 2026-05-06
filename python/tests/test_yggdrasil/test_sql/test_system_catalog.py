"""Tests for :mod:`yggdrasil.sql.system_catalog`.

The system catalog is a process-wide :class:`DynamicCatalog`
singleton; tests use ``setUp`` / ``tearDown`` to keep its state
clean between cases.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.sql import (
    Engine,
    SYSTEM_CATALOG,
    DynamicCatalog,
    system_catalog,
)
from yggdrasil.sql.catalog import default_context


def _t() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})


class _SystemCatalogTestCase(ArrowTestCase):
    """Common reset hook so test order doesn't matter."""

    def setUp(self) -> None:
        super().setUp()
        system_catalog.clear()
        # Drop legacy bindings too — some other test in the suite may
        # have written to default_context.
        for n in list(default_context.names()):
            default_context.deregister(n)

    def tearDown(self) -> None:
        system_catalog.clear()
        super().tearDown()


class TestRegistration(_SystemCatalogTestCase):
    def test_register_and_get(self) -> None:
        system_catalog.register("trades", _t())
        self.assertIn("trades", system_catalog.names())
        self.assertIsNotNone(system_catalog.get("trades"))

    def test_deregister(self) -> None:
        system_catalog.register("trades", _t())
        prior = system_catalog.deregister("trades")
        self.assertIsNotNone(prior)
        self.assertNotIn("trades", system_catalog.names())

    def test_register_many(self) -> None:
        system_catalog.register_many({"a": _t(), "b": _t()})
        self.assertEqual(set(system_catalog.names()), {"a", "b"})

    def test_clear_only_drops_locals_not_parents(self) -> None:
        # Write through the legacy SqlContext (parent of SYSTEM_CATALOG).
        default_context.register("legacy", _t())
        system_catalog.register("modern", _t())
        self.assertIn("legacy", system_catalog.names())
        self.assertIn("modern", system_catalog.names())

        system_catalog.clear()
        # ``modern`` (local) gone; ``legacy`` (parent) still visible.
        self.assertNotIn("modern", system_catalog.names())
        self.assertIn("legacy", system_catalog.names())


class TestEngineInheritance(_SystemCatalogTestCase):
    def test_engine_with_no_sources_inherits_system_catalog(self) -> None:
        system_catalog.register("trades", _t())
        eng = Engine()
        self.assertIn("trades", eng.names())
        out = eng.execute("SELECT * FROM trades").read_arrow_table()
        self.assertEqual(out.num_rows, 3)

    def test_engine_local_register_does_not_pollute_system(self) -> None:
        eng = Engine(sources={"only_here": _t()})
        # Visible on the engine.
        self.assertIn("only_here", eng.names())
        # Not on the system catalog.
        self.assertNotIn("only_here", system_catalog.names())

    def test_system_register_visible_to_existing_engines(self) -> None:
        eng = Engine()
        system_catalog.register("late", _t())
        # The engine resolves the source through the parent chain at
        # query time, so registering after construction still works.
        out = eng.execute("SELECT id FROM late").read_arrow_table()
        self.assertEqual(out.num_rows, 3)

    def test_explicit_empty_parents_isolates_engine(self) -> None:
        # Catalogs / engines that pass ``parents=[]`` opt out of the
        # system catalog — useful for tests and sandboxed evaluators.
        system_catalog.register("globally_visible", _t())
        catalog = DynamicCatalog(parents=[])
        self.assertNotIn("globally_visible", catalog.names())


class TestLegacyBridge(_SystemCatalogTestCase):
    def test_legacy_register_visible_via_system_catalog(self) -> None:
        # The legacy ``yggdrasil.sql.register`` writes to
        # ``default_context``, which is wired in as a parent of
        # SYSTEM_CATALOG, so the new path picks it up too.
        from yggdrasil.sql import register as legacy_register

        legacy_register("legacy_table", _t())
        self.assertIn("legacy_table", system_catalog.names())
        eng = Engine()
        out = eng.execute("SELECT * FROM legacy_table").read_arrow_table()
        self.assertEqual(out.num_rows, 3)

    def test_system_register_does_not_leak_into_legacy(self) -> None:
        # We deliberately keep writes one-way: registering on
        # SYSTEM_CATALOG locals does NOT mutate the legacy
        # default_context. Otherwise the two surfaces would race.
        system_catalog.register("only_modern", _t())
        self.assertNotIn("only_modern", default_context.names())


class TestSnapshot(_SystemCatalogTestCase):
    def test_snapshot_includes_locals_and_parents(self) -> None:
        default_context.register("from_legacy", _t())
        system_catalog.register("from_system", _t())
        snap = system_catalog.snapshot()
        self.assertEqual(set(snap), {"from_legacy", "from_system"})
