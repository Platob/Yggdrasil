"""Pickle round-trip for :class:`SQLWarehouse`.

The warehouse handle holds a :class:`threading.Lock` and a lazily-built
:class:`urllib3.PoolManager` to guard external-link fetches. Both are
process-local resources that can't cross a pickle boundary; pickling the
warehouse must drop them and rebuild fresh ones on the receiving side.

Bare-bones service / client stand-ins keep the test off the network —
:class:`SQLWarehouse` only needs ``service`` to be present (it's stored
verbatim) and to expose a ``client`` attribute (used by ``__del__`` /
``_release`` if GC fires during the test).
"""
from __future__ import annotations

import pickle
import threading

# Importing the SQL package first sidesteps the circular import between
# ``warehouse`` and ``sql.engine`` — same workaround as
# ``test_warehouse_empty_result``.
from yggdrasil.databricks.sql import SQLEngine  # noqa: F401  -- import-order fix
from yggdrasil.databricks.warehouse import SQLWarehouse
from yggdrasil.databricks.warehouse.service import Warehouses


class _StubService(Warehouses):
    """Picklable stand-in for the live ``Warehouses`` service."""

    def __init__(self) -> None:
        self.client = None

    def find_warehouse(self, **kwargs):  # pragma: no cover - not exercised
        return None


def _make_warehouse() -> SQLWarehouse:
    """Build an SQLWarehouse with both id + name so ``__init__`` doesn't
    trigger the ``find_warehouse`` lookup."""
    return SQLWarehouse(
        service=_StubService(),
        warehouse_id="wh-1",
        warehouse_name="wh",
    )


class TestSQLWarehousePickle:
    def test_round_trip_preserves_identity_fields(self) -> None:
        wh = _make_warehouse()
        restored: SQLWarehouse = pickle.loads(pickle.dumps(wh))

        assert restored.warehouse_id == "wh-1"
        assert restored.warehouse_name == "wh"
        assert isinstance(restored.service, _StubService)

    def test_round_trip_drops_lock_and_pool(self) -> None:
        """Lock + pool are process-local — restored with fresh defaults."""
        wh = _make_warehouse()
        # Mark the pool slot non-None to ensure the dropped value isn't
        # carried verbatim across the boundary.
        wh._external_link_pool_instance = object()

        restored: SQLWarehouse = pickle.loads(pickle.dumps(wh))

        assert isinstance(restored._external_link_pool_lock, type(threading.Lock()))
        assert restored._external_link_pool_lock is not wh._external_link_pool_lock
        assert restored._external_link_pool_instance is None

    def test_round_trip_preserves_disposable_slots(self) -> None:
        """Disposable's open/close bookkeeping lives in ``__slots__`` and
        must travel even though it isn't in ``__dict__``."""
        wh = _make_warehouse()
        wh._acquired = True
        wh._dirty = True
        wh._depth = 3

        restored: SQLWarehouse = pickle.loads(pickle.dumps(wh))

        assert restored._acquired is True
        assert restored._dirty is True
        assert restored._depth == 3

    def test_pool_rebuilds_after_unpickle(self) -> None:
        """The fresh lock + None pool slot let ``external_link_pool``
        lazily rebuild on the receiving side without raising."""
        wh = _make_warehouse()
        restored: SQLWarehouse = pickle.loads(pickle.dumps(wh))

        pool = restored.external_link_pool(max_workers=2)
        assert pool is not None
        # Calling it again returns the same cached pool.
        assert restored.external_link_pool(max_workers=2) is pool
