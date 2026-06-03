"""Live-integration tests for the :class:`DatabricksPath` dispatcher
plus :class:`Schema` / :class:`Volume` / :class:`VolumePath` identity.

What this file pins down end-to-end against a real workspace:

- ``DatabricksPath("/Volumes/<cat>/<sch>")`` resolves to a :class:`Schema`
  (and ``DatabricksPath("/Volumes/<cat>/<sch>/<vol>")`` to a
  :class:`Volume`, ``/.../x`` to a :class:`VolumePath`) — same
  segment-depth dispatch the unit tests exercise, but proven against
  the live SDK so the URL → resource → SDK call chain doesn't drift.
- :class:`Schema` lifecycle: get_or_create / use / delete inside a
  catalog the test identity already owns. The test does **not** create
  a catalog — that requires permissions most service principals don't
  have, and the user's environment can't grant them; the catalog is
  read from :envvar:`DATABRICKS_INTEGRATION_CATALOG` and assumed
  pre-existing with CREATE SCHEMA on it.
- :class:`Volume` lifecycle inside the throw-away schema:
  read-after-write, sub-directory creation, idempotent ``ensure_*``
  paths.
- :class:`VolumePath` round-trip via the dispatcher: writing through
  ``DatabricksPath("/Volumes/<cat>/<sch>/<vol>/<file>")`` must land at
  the same UC location as a direct :class:`VolumePath` constructor.

Skip rules
----------

Skipped wholesale unless ``DATABRICKS_HOST`` is set. The catalog is
read from :envvar:`DATABRICKS_INTEGRATION_CATALOG` (default
``trading``); if the test identity can't ``CREATE SCHEMA`` on it,
``setUpClass`` raises :class:`unittest.SkipTest` with the exact
permission error so the run keeps moving cleanly.

Cleanup
-------

A unique schema (``yg_dispatch_<hex>``) is provisioned per test class
so concurrent runs don't collide and a failure leaves at most one
schema behind. Class teardown drops the whole schema (cascade-style
via ``Schema.delete(force=True)``) so any volumes / volume paths /
files written underneath it go with it.
"""

from __future__ import annotations

import os
import secrets
import unittest
from typing import ClassVar

import pytest
from databricks.sdk.errors import DatabricksError, PermissionDenied

from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.databricks.path import DatabricksPath
from yggdrasil.databricks.schema.schema import UCSchema
from yggdrasil.databricks.volume.volume import Volume
from yggdrasil.io.io_stats import IOKind

from . import DatabricksIntegrationCase


__all__ = [
    "TestSchemaIntegration",
    "TestVolumeIntegration",
    "TestVolumePathDispatchIntegration",
]


def _resolve_catalog() -> str:
    """Catalog the test identity is allowed to create schemas inside.

    Defaults to ``trading`` (the project's standard integration
    catalog) but a contributor can point at any other catalog they
    own via the env var.
    """
    name = os.environ.get(
        "DATABRICKS_INTEGRATION_CATALOG", "trading",
    ).strip()
    if not name:
        raise unittest.SkipTest(
            "DATABRICKS_INTEGRATION_CATALOG is empty — set it to a "
            "catalog the test identity has CREATE SCHEMA on."
        )
    return name


class _SchemaFixture(DatabricksIntegrationCase):
    """Shared fixture: per-class throw-away schema.

    Subclasses get ``cls.catalog_name`` / ``cls.schema_name`` /
    ``cls.schema`` populated, with cleanup wired so a forgotten test
    method can't leave UC state behind.
    """

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    schema: ClassVar[UCSchema]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = _resolve_catalog()
        cls.schema_name = f"yg_dispatch_{secrets.token_hex(4)}"
        try:
            cls.schema = cls.client.schemas(
                catalog_name=cls.catalog_name,
            ).schema(schema_name=cls.schema_name)
            cls.schema.get_or_create(
                comment="yggdrasil DatabricksPath dispatch integration",
            )
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Cannot create schema "
                f"{cls.catalog_name}.{cls.schema_name}: {exc}. Override "
                f"DATABRICKS_INTEGRATION_CATALOG with a catalog the test "
                f"identity can CREATE SCHEMA on."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            schema = getattr(cls, "schema", None)
            if schema is not None:
                schema.delete(force=True, raise_error=False)
        finally:
            super().tearDownClass()


@pytest.mark.integration
class TestSchemaIntegration(_SchemaFixture):
    """:class:`Schema` lifecycle + dispatcher round-trip."""

    def test_dispatch_resolves_to_schema(self) -> None:
        """``DatabricksPath("/Volumes/<cat>/<sch>")`` returns a live
        :class:`Schema` bound to the right catalog + name."""
        resolved = DatabricksPath(
            f"/Volumes/{self.catalog_name}/{self.schema_name}",
            service=self.client.schemas,
        )
        self.assertIsInstance(resolved, UCSchema)
        self.assertEqual(resolved.catalog_name, self.catalog_name)
        self.assertEqual(resolved.schema_name, self.schema_name)
        # The dispatcher's Schema must collapse onto the same Singleton
        # entry the fixture created so the cached SchemaInfo carries.
        direct = self.client.schemas(
            catalog_name=self.catalog_name,
        ).schema(schema_name=self.schema_name)
        self.assertIs(resolved, direct)

    def test_schema_exists_after_create(self) -> None:
        self.assertTrue(self.schema.exists())

    def test_schema_full_name(self) -> None:
        self.assertEqual(
            self.schema.full_name(),
            f"{self.catalog_name}.{self.schema_name}",
        )


@pytest.mark.integration
class TestVolumeIntegration(_SchemaFixture):
    """:class:`Volume` lifecycle inside the throw-away schema."""

    volume_name: ClassVar[str]
    volume: ClassVar[Volume]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.volume_name = f"yg_disp_{secrets.token_hex(3)}"
        try:
            cls.volume = cls.client.volumes(
                catalog_name=cls.catalog_name,
                schema_name=cls.schema_name,
            ).volume(volume_name=cls.volume_name)
            cls.volume.create()
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Cannot create volume "
                f"{cls.catalog_name}.{cls.schema_name}.{cls.volume_name}: "
                f"{exc}."
            ) from exc

    def test_dispatch_resolves_to_volume(self) -> None:
        """``DatabricksPath("/Volumes/<cat>/<sch>/<vol>")`` returns the
        same live :class:`Volume` the fixture created."""
        resolved = DatabricksPath(
            f"/Volumes/{self.catalog_name}/{self.schema_name}/{self.volume_name}",
            service=self.client.volumes,
        )
        self.assertIsInstance(resolved, Volume)
        self.assertEqual(resolved.catalog_name, self.catalog_name)
        self.assertEqual(resolved.schema_name, self.schema_name)
        self.assertEqual(resolved.volume_name, self.volume_name)
        self.assertIs(resolved, self.volume)

    def test_volume_full_name(self) -> None:
        self.assertEqual(
            self.volume.full_name(),
            f"{self.catalog_name}.{self.schema_name}.{self.volume_name}",
        )

    def test_volume_info_round_trip(self) -> None:
        """``read_info`` should hit the live SDK and return a usable
        ``VolumeInfo`` shaped for our throw-away volume."""
        info = self.volume.read_info()
        self.assertIsNotNone(info)
        self.assertEqual(info.catalog_name, self.catalog_name)
        self.assertEqual(info.schema_name, self.schema_name)
        self.assertEqual(info.name, self.volume_name)


@pytest.mark.integration
class TestVolumePathDispatchIntegration(_SchemaFixture):
    """:class:`VolumePath` round-trip through the dispatcher."""

    volume_name: ClassVar[str]
    volume: ClassVar[Volume]
    root: ClassVar[VolumePath]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.volume_name = f"yg_disp_{secrets.token_hex(3)}"
        try:
            cls.volume = cls.client.volumes(
                catalog_name=cls.catalog_name,
                schema_name=cls.schema_name,
            ).volume(volume_name=cls.volume_name)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Cannot create volume for VolumePath dispatch test: {exc}."
            ) from exc
        cls.root = VolumePath(
            f"/Volumes/{cls.catalog_name}/{cls.schema_name}/{cls.volume_name}/scratch",
            service=cls.client.volumes,
        )
        cls.root.mkdir(parents=True, exist_ok=True)

    def test_dispatch_resolves_to_volume_path(self) -> None:
        """``DatabricksPath("/Volumes/<cat>/<sch>/<vol>/<rest>")`` returns
        a :class:`VolumePath` (not a :class:`Volume`) — the depth-4+
        branch of the dispatcher."""
        resolved = DatabricksPath(
            f"/Volumes/{self.catalog_name}/{self.schema_name}/"
            f"{self.volume_name}/scratch/probe.bin",
            service=self.client.volumes,
        )
        self.assertIsInstance(resolved, VolumePath)
        # ``catalog_name`` / ``schema_name`` / ``volume_name`` resolve
        # back to the originating UC volume so a downstream
        # ``resolved.volume`` / ``resolved.schema`` / ``resolved.catalog``
        # collapses onto the same singletons we already own.
        self.assertEqual(resolved.catalog_name, self.catalog_name)
        self.assertEqual(resolved.schema_name, self.schema_name)
        self.assertEqual(resolved.volume_name, self.volume_name)

    def test_dispatch_round_trip_writes_against_files_api(self) -> None:
        """A path obtained through the dispatcher must round-trip
        bytes through the same Files-API path a hand-built
        :class:`VolumePath` would."""
        leaf = f"dispatch-{secrets.token_hex(4)}.bin"
        payload = b"hello-dispatch-" + secrets.token_bytes(8)

        dispatched = DatabricksPath(
            f"/Volumes/{self.catalog_name}/{self.schema_name}/"
            f"{self.volume_name}/scratch/{leaf}",
            service=self.client.volumes,
        )
        dispatched.write_bytes(payload)

        # Read back through a freshly-built ``VolumePath`` (no shared
        # in-memory state) so the assertion is "the bytes are at the
        # canonical UC path", not "the dispatched object remembered them".
        direct = VolumePath(
            f"/Volumes/{self.catalog_name}/{self.schema_name}/"
            f"{self.volume_name}/scratch/{leaf}",
            service=self.client.volumes,
        )
        try:
            stat = direct.stat()
            self.assertEqual(stat.kind, IOKind.FILE)
            self.assertEqual(stat.size, len(payload))
            self.assertEqual(direct.read_bytes(), payload)
        finally:
            direct.unlink(missing_ok=True)
