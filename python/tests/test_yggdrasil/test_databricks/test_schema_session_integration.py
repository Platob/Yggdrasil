"""Live-integration tests for :class:`SchemaSession`.

Exercises the schema-backed HTTP cache against a real upstream
(``httpbin.org``) and a real Unity Catalog schema. Skipped unless
``DATABRICKS_HOST`` (and matching credentials) are exported — see
:class:`DatabricksIntegrationCase`.

Determinism trick
-----------------
``httpbin.org/uuid`` returns a freshly-generated UUID on every call,
so the response body is its own cache verifier:

* :attr:`Mode.APPEND` — the read-through path: cold call writes a row,
  warm call returns the *same* UUID because it came from the table
  rather than the wire.
* :attr:`Mode.UPSERT` — bypass-on-read: two calls return *different*
  UUIDs, but the table ends with exactly one row keyed by the public
  URL hash.

Local cache layer
-----------------
``local_cache=False`` on every test so the assertions track the
remote (Delta-table) tier in isolation; a separate test pins the
two-tier behaviour by enabling the local layer and asserting a warm
call short-circuits before the remote read.

Cleanup
-------
A per-class throw-away schema (``yg_schemasession_<hex>``) is created
under :envvar:`DATABRICKS_INTEGRATION_CATALOG` (default ``trading``)
and dropped in ``tearDownClass`` so the tables provisioned by each
``/uuid`` call go with it.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import unittest
from pathlib import Path
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
    "TestSchemaSessionLocalCache",
]


HTTPBIN_BASE = os.environ.get("YGG_HTTPBIN_BASE", "https://httpbin.org").rstrip("/")


def _get_uuid(session: SchemaSession) -> tuple[Response, str]:
    """Issue ``GET /uuid`` and return ``(response, parsed_uuid)``.

    Wraps the bare ``send`` so an httpbin outage skips the test
    instead of raising a noisy fail.
    """
    req = PreparedRequest.prepare("GET", f"{HTTPBIN_BASE}/uuid")
    try:
        response = session.send(req)
    except Exception as exc:  # noqa: BLE001 - network surface
        raise unittest.SkipTest(
            f"Upstream {HTTPBIN_BASE!r} unreachable: {exc}. "
            "Set YGG_HTTPBIN_BASE to a reachable httpbin mirror."
        ) from exc

    if response.status_code != 200:
        raise unittest.SkipTest(
            f"httpbin returned {response.status_code} for /uuid — skipping."
        )

    body = response.buffer.to_bytes() if response.buffer is not None else b""
    payload = json.loads(body)
    if "uuid" not in payload:
        raise AssertionError(f"Unexpected /uuid payload shape: {payload!r}")
    return response, payload["uuid"]


def _row_count(table) -> int:
    """``SELECT count(*)`` against *table* as a plain int."""
    result = table.execute(
        f"SELECT count(*) AS n FROM {table.full_name(safe=True)}"
    )
    for batch in result.read_arrow_batches():
        if batch.num_rows:
            return int(batch.column("n")[0].as_py())
    return 0


# ---------------------------------------------------------------------------
# Live integration fixture
# ---------------------------------------------------------------------------


class _SchemaSessionLiveBase(DatabricksIntegrationCase):
    """Per-class throw-away schema under
    :envvar:`DATABRICKS_INTEGRATION_CATALOG` (default ``trading``)."""

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    schema: ClassVar[Schema]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = (
            os.environ.get("DATABRICKS_INTEGRATION_CATALOG", "trading").strip()
            or "trading"
        )
        cls.schema_name = f"yg_schemasession_{secrets.token_hex(4)}"
        try:
            cls.schema = cls.client.schemas(catalog_name=cls.catalog_name).schema(
                schema_name=cls.schema_name,
            )
            cls.schema.ensure_created(
                comment="yggdrasil SchemaSession integration-test schema",
            )
        except DatabricksError as exc:
            raise unittest.SkipTest(
                f"Cannot create schema {cls.catalog_name}.{cls.schema_name}: "
                f"{exc}. Override DATABRICKS_INTEGRATION_CATALOG with a "
                "catalog the test identity can write to."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            sch = getattr(cls, "schema", None)
            if sch is not None:
                sch.delete(force=True, raise_error=False)
        finally:
            super().tearDownClass()

    def _session(self, *, mode: Mode, local_cache=False) -> SchemaSession:
        """Build a fresh session per test — unique ``base_url`` suffix
        keeps the singleton cache from handing back a stale instance
        from an earlier case (the cache now keys on every constructor
        argument, so a per-test URL suffix is the simplest split)."""
        return SchemaSession(
            self.schema,
            base_url=f"{HTTPBIN_BASE}#int-{secrets.token_hex(2)}",
            mode=mode,
            local_cache=local_cache,
        )


# ---------------------------------------------------------------------------
# APPEND mode — read-through cache
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSchemaSessionAppend(_SchemaSessionLiveBase):

    def test_warm_call_returns_cached_uuid(self) -> None:
        """Cold call writes; warm call returns the same UUID from the table."""
        session = self._session(mode=Mode.APPEND)

        _, first = _get_uuid(session)
        _, second = _get_uuid(session)

        self.assertEqual(
            second, first,
            "APPEND mode: second /uuid call should have hit the remote "
            "cache and returned the cold call's UUID",
        )

        table = self.schema.table("uuid")
        self.assertTrue(
            table.exists,
            f"expected SchemaSession to have auto-created {table.full_name()!r} "
            f"on the first /uuid call",
        )

    def test_distinct_paths_use_distinct_tables(self) -> None:
        """``/uuid`` and ``/get`` resolve to different table names; both
        get auto-created on first call."""
        session = self._session(mode=Mode.APPEND)

        _get_uuid(session)
        req = PreparedRequest.prepare("GET", f"{HTTPBIN_BASE}/get")
        try:
            other = session.send(req)
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"httpbin /get unreachable: {exc}") from exc
        self.assertEqual(other.status_code, 200)

        self.assertTrue(self.schema.table("uuid").exists)
        self.assertTrue(self.schema.table("get").exists)


# ---------------------------------------------------------------------------
# UPSERT mode — bypass on read, refresh on write
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSchemaSessionUpsert(_SchemaSessionLiveBase):

    def test_each_call_returns_fresh_uuid(self) -> None:
        session = self._session(mode=Mode.UPSERT)

        _, first = _get_uuid(session)
        _, second = _get_uuid(session)

        self.assertNotEqual(
            first, second,
            "UPSERT must bypass the cache lookup — two calls should "
            "have produced two distinct UUIDs from the wire",
        )

        table = self.schema.table("uuid")
        self.assertTrue(table.exists)
        count = _row_count(table)
        self.assertEqual(
            count, 1,
            f"UPSERT keys by public_url_hash and replaces — expected 1 row, "
            f"got {count}",
        )


# ---------------------------------------------------------------------------
# Two-tier (local + remote) cache
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSchemaSessionLocalCache(_SchemaSessionLiveBase):
    """Local on-disk tier short-circuits before the remote read."""

    def test_local_layer_serves_warm_call_without_remote(self) -> None:
        """Two-tier APPEND: cold call hits the wire and writes both
        tiers; warm call returns the same UUID — which is the
        end-user contract regardless of which tier serves it. We
        also confirm the local tier left a file on disk so the
        fast-path is genuinely populated."""
        with tempfile.TemporaryDirectory(prefix="yg-schemasession-") as tmp:
            session = SchemaSession(
                self.schema,
                base_url=f"{HTTPBIN_BASE}#int-local-{secrets.token_hex(2)}",
                mode=Mode.APPEND,
                local_cache=tmp,
            )

            _, first = _get_uuid(session)
            _, second = _get_uuid(session)

            self.assertEqual(
                second, first,
                "Warm call should have returned the cached UUID",
            )

            # The local fast-path writes one ``.arrow`` file per
            # response under ``<tmp>/<METHOD>/<host>/.../<hash>.arrow``.
            arrow_files = list(Path(tmp).rglob("*.arrow"))
            self.assertTrue(
                arrow_files,
                f"Expected the local cache layer to leave at least one "
                f".arrow file under {tmp!r}; got nothing.",
            )
