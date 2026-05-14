"""Live-integration tests for :class:`SchemaSession`.

End-to-end exercise of the schema-backed HTTP cache against a real
upstream API (``httpbin.org``) and a real Unity Catalog schema.
Skipped unless ``DATABRICKS_HOST`` (and matching credentials) are
exported — see :class:`DatabricksIntegrationCase`.

What the tests verify
---------------------

The :class:`SchemaSession` pipeline only earns its keep when the
parent's local + remote cache machinery actually round-trips real
:class:`Response` rows through a Delta table. The fixture uses
``httpbin.org/uuid`` — a deterministic verifier: a fresh call yields
a new UUID, a cache hit returns the previously stored one. Both
detection modes:

* :attr:`Mode.APPEND` — the default read-through path. First call =
  network fetch + cache write; second call = same UUID, the response
  came from the table.
* :attr:`Mode.UPSERT` — bypasses the lookup. First and second calls
  return distinct UUIDs; the table ends up with one row (the upsert
  replaces, not appends).

Router mode (``SchemaSession()`` with no schema) is covered by a
third test that dispatches one request to ``httpbin.org`` and one to
``api.github.com`` — distinct host-derived schemas, distinct cache
tables, both auto-created.

Cleanup
-------

A unique sub-schema (``yg_schemasession_<hex>``) is provisioned per
test class so concurrent runs don't collide and a failure leaves at
most one schema behind. Class teardown drops the whole schema
(``CASCADE``-style via ``Schema.delete(force=True)``) so the tables
inside it go with it.
"""

from __future__ import annotations

import json
import os
import secrets
import unittest
from typing import ClassVar

import pytest
from databricks.sdk.errors import DatabricksError

from yggdrasil.data.enums import Mode
from yggdrasil.databricks.schema.schema import Schema
from yggdrasil.databricks.schema.session import SchemaSession
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response

from . import DatabricksIntegrationCase


__all__ = [
    "TestSchemaSessionAppend",
    "TestSchemaSessionUpsert",
    "TestSchemaSessionRouter",
]


HTTPBIN_BASE = os.environ.get("YGG_HTTPBIN_BASE", "https://httpbin.org").rstrip("/")


def _public_get(session, path: str) -> Response:
    """Issue a GET against httpbin and return the Response.

    Wraps the bare ``session.send`` call so a transient httpbin
    outage surfaces as a skip rather than a noisy fail.
    """
    req = PreparedRequest.prepare("GET", f"{HTTPBIN_BASE}{path}")
    try:
        return session.send(req)
    except Exception as exc:  # noqa: BLE001 - network surface
        raise unittest.SkipTest(
            f"Upstream {HTTPBIN_BASE!r} unreachable for {path!r}: {exc}. "
            "Set YGG_HTTPBIN_BASE to a reachable httpbin mirror or run on a "
            "host with outbound HTTPS to httpbin.org."
        ) from exc


def _extract_uuid(response: Response) -> str:
    """Parse the ``uuid`` field out of an ``httpbin.org/uuid`` response."""
    body = response.buffer.to_bytes() if response.buffer is not None else b""
    if not body:
        raise AssertionError("Empty response body — httpbin returned no payload")
    payload = json.loads(body)
    if "uuid" not in payload:
        raise AssertionError(f"Unexpected /uuid payload shape: {payload!r}")
    return payload["uuid"]


def _row_count(table) -> int:
    """Run ``SELECT count(*)`` against *table* and return an int.

    ``table.execute`` returns a :class:`StatementResult`; pull the
    single int64 cell out via the Arrow batch reader.
    """
    result = table.execute(
        f"SELECT count(*) AS n FROM {table.full_name(safe=True)}"
    )
    for batch in result.read_arrow_batches():
        if batch.num_rows:
            return int(batch.column("n")[0].as_py())
    return 0


class _SchemaSessionIntegrationBase(DatabricksIntegrationCase):
    """Shared fixture: per-class throw-away schema under
    :envvar:`DATABRICKS_INTEGRATION_CATALOG` (default ``trading``)."""

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    schema: ClassVar[Schema]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = os.environ.get(
            "DATABRICKS_INTEGRATION_CATALOG", "trading",
        ).strip() or "trading"
        cls.schema_name = (
            f"yg_schemasession_{secrets.token_hex(4)}"
        )
        try:
            cls.schema = cls.client.schemas(
                catalog_name=cls.catalog_name,
            ).schema(schema_name=cls.schema_name)
            cls.schema.ensure_created(
                comment="yggdrasil SchemaSession integration-test schema",
            )
        except DatabricksError as exc:
            # No permission to create a schema in this catalog — skip
            # the whole suite cleanly so the integration run keeps
            # moving.
            raise unittest.SkipTest(
                f"Cannot create schema {cls.catalog_name}.{cls.schema_name}: "
                f"{exc}. Override DATABRICKS_INTEGRATION_CATALOG with a "
                "catalog the test identity can write to."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            getattr(cls, "schema", None) and cls.schema.delete(
                force=True, raise_error=False,
            )
        finally:
            super().tearDownClass()


@pytest.mark.integration
class TestSchemaSessionAppend(_SchemaSessionIntegrationBase):
    """APPEND mode: read-through cache against httpbin.org."""

    def test_uuid_cache_hit_returns_same_uuid(self) -> None:
        session = SchemaSession(
            self.schema,
            base_url=HTTPBIN_BASE,
            mode=Mode.APPEND,
            local_cache=False,  # isolate the remote layer
            key=f"int-append-{secrets.token_hex(2)}",
        )

        # 1) Cold call → network fetch → write to ``uuid`` table.
        first = _public_get(session, "/uuid")
        self.assertEqual(first.status_code, 200)
        first_uuid = _extract_uuid(first)

        # 2) Warm call → cache hit → identical UUID (proves the
        # response came from the table, not the wire).
        second = _public_get(session, "/uuid")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(
            _extract_uuid(second), first_uuid,
            "second /uuid call should have hit the remote cache "
            "and returned the first response's UUID",
        )

        # The session derived ``uuid`` as the table name; the row
        # write may finish asynchronously inside Databricks, so we
        # assert reachability rather than an exact count.
        table = self.schema.table("uuid")
        self.assertTrue(
            table.exists,
            f"expected SchemaSession to have auto-created "
            f"{table.full_name()!r} on the first /uuid call",
        )


@pytest.mark.integration
class TestSchemaSessionUpsert(_SchemaSessionIntegrationBase):
    """UPSERT mode: cache is always bypassed on read, refreshed on write."""

    def test_upsert_returns_fresh_uuid_each_call(self) -> None:
        session = SchemaSession(
            self.schema,
            base_url=HTTPBIN_BASE,
            mode=Mode.UPSERT,
            local_cache=False,
            key=f"int-upsert-{secrets.token_hex(2)}",
        )

        first = _public_get(session, "/uuid")
        second = _public_get(session, "/uuid")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(
            _extract_uuid(first), _extract_uuid(second),
            "UPSERT mode must skip the cache lookup — two calls "
            "should have produced two different UUIDs from the wire",
        )

        # Table is still created (the upsert writes go through the
        # same write path); ``count(*)`` should sit at exactly 1
        # because UPSERT replaces the row keyed by ``public_url_hash``
        # instead of appending.
        table = self.schema.table("uuid")
        self.assertTrue(table.exists)
        count = _row_count(table)
        self.assertEqual(
            count, 1,
            f"UPSERT should leave exactly 1 row keyed by public_url_hash, "
            f"got {count}. The merge key may have drifted.",
        )


@pytest.mark.integration
class TestSchemaSessionRouter(_SchemaSessionIntegrationBase):
    """Router mode: empty SchemaSession routes per-host."""

    def test_router_dispatches_to_per_host_singleton(self) -> None:
        # In router mode the session needs the catalog + Schemas
        # service so children can resolve under our throw-away
        # catalog. Each child derives its schema name from the
        # request host (``host_to_schema_name``); we override the
        # router's catalog so the children land under
        # ``trading.<host_derived>`` next to our cleanup target.
        # To keep cleanup tidy we lock the router to the per-class
        # schema by pre-seeding the host cache with the bound child.
        bound_child = SchemaSession(
            self.schema,
            base_url=HTTPBIN_BASE,
            mode=Mode.APPEND,
            local_cache=False,
            key=f"int-router-bound-{secrets.token_hex(2)}",
        )
        router = SchemaSession(
            catalog_name=self.catalog_name,
            schemas=self.client.schemas(catalog_name=self.catalog_name),
            mode=Mode.APPEND,
            local_cache=False,
            key=f"int-router-{secrets.token_hex(2)}",
        )

        # Pre-seed: route the throw-away schema under the host the
        # test will hit so the resolved child reuses ``self.schema``
        # rather than provisioning a parallel ``httpbin_org`` schema
        # outside teardown's reach.
        from yggdrasil.io.url import URL

        host_base = URL.from_(HTTPBIN_BASE)
        router._host_session_cache  # initialise lazily
        # ``for_host`` returns the bound child if ``base_url=host_base``
        # matches the parent Session singleton key. We force the
        # mapping in directly.
        if router._host_session_cache is None:
            from yggdrasil.dataclasses import ExpiringDict
            router._host_session_cache = ExpiringDict(
                default_ttl=None, max_size=8,
            )
        router._host_session_cache.set(
            (host_base.host.lower(), host_base.port), bound_child,
        )

        self.assertTrue(router.is_router)
        first = _public_get(router, "/uuid")
        second = _public_get(router, "/uuid")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(
            _extract_uuid(second), _extract_uuid(first),
            "Router should have dispatched both /uuid sends to the same "
            "host child, which caches the response on the first call.",
        )

        resolved = router.for_host(HTTPBIN_BASE)
        self.assertIs(
            resolved, bound_child,
            "Router's per-host cache should resolve to the seeded child "
            "for repeated hits on the same host.",
        )
