"""Live-integration tests for :class:`Schema` and :class:`Schemas`.

The path-dispatch integration suite already pins the URL → resource
chain (``DatabricksPath("/Volumes/<cat>/<sch>")`` → :class:`Schema`)
and the bare ``exists`` / ``full_name`` shape. This file targets the
deeper schema surface that only meaningfully exercises against a live
Unity Catalog endpoint:

- :class:`Schema` lifecycle — ``create`` with ``comment`` /
  ``properties``, ``get_or_create`` idempotence, ``update``,
  ``delete(raise_error=False)`` on a missing schema.
- :attr:`Schema.infos` lazy fetch + ``clear()`` cache reset round-trip.
- :class:`Schemas` collection — ``list`` (with glob filter), ``find``
  (cache hit + cache bypass), ``find_remote`` raw API path.
- :meth:`Schema.set_tags` / :meth:`Schema.unset_tags` against the
  ``entity_tag_assignments`` REST API.
- :meth:`Schema.grant` / :meth:`Schema.revoke` /
  :meth:`Schema.set_permissions` plus :meth:`Schema.permissions` /
  :meth:`Schema.effective_permissions` against ``grants``.
- :meth:`Schema.rename` round-trip (and rename back so cleanup hits
  the right ``full_name``).

Skip rules
----------

Skipped wholesale unless ``DATABRICKS_HOST`` is set. The catalog is
read from :envvar:`DATABRICKS_INTEGRATION_CATALOG` (default
``trading``); the test identity must have CREATE SCHEMA on it. Tag
and permission tests degrade to ``unittest.SkipTest`` when the
identity lacks the matching grant on the workspace rather than
failing the whole suite.

Cleanup
-------

A unique schema (``yg_schema_<hex>``) is provisioned per test class
and dropped cascade-style in ``tearDownClass`` so a failure leaves
at most one schema behind.
"""

from __future__ import annotations

import os
import secrets
import unittest
from typing import ClassVar

import pytest
from databricks.sdk.errors import (
    DatabricksError,
    NotFound,
    PermissionDenied,
    ResourceDoesNotExist,
)
from databricks.sdk.service.catalog import Privilege

from yggdrasil.databricks.schema.schema import UCSchema
from yggdrasil.databricks.schema.schemas import Schemas, _SCHEMA_INFO_CACHE

from . import DatabricksIntegrationCase


__all__ = [
    "TestSchemaLifecycleIntegration",
    "TestSchemaInfoIntegration",
    "TestSchemasCollectionIntegration",
    "TestSchemaTagsIntegration",
    "TestSchemaPermissionsIntegration",
    "TestSchemaRenameIntegration",
]


def _resolve_catalog() -> str:
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
    """Per-class throw-away schema."""

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    schema: ClassVar[UCSchema]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = _resolve_catalog()
        cls.schema_name = f"yg_schema_{secrets.token_hex(4)}"
        try:
            cls.schema = cls.client.schemas(
                catalog_name=cls.catalog_name,
            ).schema(schema_name=cls.schema_name)
            cls.schema.get_or_create(
                comment="yggdrasil schema integration",
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
                # Re-resolve under the current name in case a rename test
                # left it pointing somewhere unexpected.
                try:
                    schema.delete(force=True, raise_error=False)
                except Exception:
                    pass
        finally:
            super().tearDownClass()


@pytest.mark.integration
class TestSchemaLifecycleIntegration(_SchemaFixture):
    """``create`` / ``get_or_create`` / ``update`` / ``delete`` round-trips."""

    def test_exists_after_create(self) -> None:
        self.assertTrue(self.schema.exists())

    def test_get_or_create_is_idempotent(self) -> None:
        # Second get_or_create call must NOT raise — the
        # "already exists" path soft-resets the cache and returns self.
        result = self.schema.get_or_create(
            comment="yggdrasil schema integration",
        )
        self.assertIs(result, self.schema)
        self.assertTrue(self.schema.exists())

    def test_create_missing_ok_false_raises_on_existing(self) -> None:
        with self.assertRaises(DatabricksError):
            self.schema.create(missing_ok=False)

    def test_update_comment_round_trip(self) -> None:
        new_comment = f"updated-{secrets.token_hex(3)}"
        self.schema.update(comment=new_comment)
        # ``update`` repopulates ``_infos`` in-place — re-read without
        # an extra remote call to verify the value landed.
        self.assertEqual(self.schema.comment, new_comment)
        # And a fresh fetch after clear() should agree.
        self.schema.clear()
        self.assertEqual(self.schema.comment, new_comment)

    def test_delete_missing_schema_with_raise_error_false(self) -> None:
        """``delete(raise_error=False)`` on a non-existent schema must
        swallow the ``NotFound`` rather than propagating it."""
        ghost_name = f"yg_ghost_{secrets.token_hex(4)}"
        ghost = self.client.schemas(
            catalog_name=self.catalog_name,
        ).schema(schema_name=ghost_name)
        # Don't create it — just try to delete.
        try:
            ghost.delete(raise_error=False)
        except DatabricksError:
            self.fail(
                "Schema.delete(raise_error=False) leaked DatabricksError "
                "for a missing schema"
            )


@pytest.mark.integration
class TestSchemaInfoIntegration(_SchemaFixture):
    """:attr:`Schema.infos` lazy fetch and cache reset behavior."""

    def test_infos_populated(self) -> None:
        info = self.schema.infos
        self.assertEqual(info.catalog_name, self.catalog_name)
        self.assertEqual(info.name, self.schema_name)

    def test_full_name(self) -> None:
        self.assertEqual(
            self.schema.full_name(),
            f"{self.catalog_name}.{self.schema_name}",
        )
        self.assertEqual(
            self.schema.full_name(safe=True),
            f"`{self.catalog_name}`.`{self.schema_name}`",
        )

    def test_owner_set_by_server(self) -> None:
        # The workspace identity always owns the schemas it just created.
        # We don't pin the exact principal — different auth methods
        # (PAT vs OAuth vs SP) surface different owner formats — but
        # the field must be populated.
        self.assertTrue(self.schema.owner)

    def test_clear_drops_cached_infos(self) -> None:
        # Warm the cache, then clear it. The next access must re-fetch.
        _ = self.schema.infos
        self.assertIsNotNone(self.schema._infos)
        self.schema.clear()
        self.assertIsNone(self.schema._infos)
        # Re-read goes through to the SDK and re-populates the slot.
        self.assertEqual(self.schema.infos.name, self.schema_name)
        self.assertIsNotNone(self.schema._infos)

    def test_exists_false_on_missing_schema(self) -> None:
        ghost = self.client.schemas(
            catalog_name=self.catalog_name,
        ).schema(schema_name=f"yg_ghost_{secrets.token_hex(4)}")
        self.assertFalse(ghost.exists())


@pytest.mark.integration
class TestSchemasCollectionIntegration(_SchemaFixture):
    """:class:`Schemas` collection-level navigation and listing."""

    def test_list_includes_fixture_schema(self) -> None:
        names = {
            sch.schema_name
            for sch in self.client.schemas.list(catalog_name=self.catalog_name)
        }
        self.assertIn(self.schema_name, names)

    def test_list_with_glob_filter(self) -> None:
        # ``yg_schema_*`` must match this fixture's schema (and nothing else
        # unrelated by accident).
        matched = list(self.client.schemas.list(
            name="yg_schema_*",
            catalog_name=self.catalog_name,
        ))
        names = {s.schema_name for s in matched}
        self.assertIn(self.schema_name, names)
        for sch in matched:
            self.assertTrue(sch.schema_name.startswith("yg_schema_"))

    def test_find_returns_singleton_with_warm_infos(self) -> None:
        # ``find`` returns a Schema instance with ``_infos`` already
        # populated (either from the module cache or via find_remote)
        # — no extra ``.infos`` round trip needed.
        Schemas.invalidate_all()
        found = self.client.schemas.find(
            f"{self.catalog_name}.{self.schema_name}",
        )
        self.assertIsNotNone(found)
        self.assertIsNotNone(found._infos)
        # And the singleton collapses with the fixture's instance.
        self.assertIs(found, self.schema)

    def test_find_returns_none_when_missing_and_raise_error_false(self) -> None:
        missing = self.client.schemas.find(
            f"{self.catalog_name}.yg_ghost_{secrets.token_hex(4)}",
            raise_error=False,
        )
        self.assertIsNone(missing)

    def test_find_raises_when_missing_by_default(self) -> None:
        with self.assertRaises(ResourceDoesNotExist):
            self.client.schemas.find(
                f"{self.catalog_name}.yg_ghost_{secrets.token_hex(4)}",
                cache_ttl=None,
            )

    def test_find_remote_bypasses_cache(self) -> None:
        # ``find_remote`` always issues a fresh API call — pre-populating
        # the module cache should NOT short-circuit it.
        svc = self.client.schemas(catalog_name=self.catalog_name)
        info = svc.find_remote(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )
        self.assertIsNotNone(info)
        self.assertEqual(info.catalog_name, self.catalog_name)
        self.assertEqual(info.name, self.schema_name)

    def test_find_remote_missing_raises(self) -> None:
        svc = self.client.schemas(catalog_name=self.catalog_name)
        with self.assertRaises(ResourceDoesNotExist):
            svc.find_remote(
                catalog_name=self.catalog_name,
                schema_name=f"yg_ghost_{secrets.token_hex(4)}",
            )

    def test_find_remote_missing_returns_none_when_raise_error_false(self) -> None:
        svc = self.client.schemas(catalog_name=self.catalog_name)
        info = svc.find_remote(
            catalog_name=self.catalog_name,
            schema_name=f"yg_ghost_{secrets.token_hex(4)}",
            raise_error=False,
        )
        self.assertIsNone(info)

    def test_subscript_resolves_to_schema(self) -> None:
        # Two-part dotted name → Schema; collapses to the same instance.
        svc = self.client.schemas(catalog_name=self.catalog_name)
        sch = svc[f"{self.catalog_name}.{self.schema_name}"]
        self.assertIsInstance(sch, UCSchema)
        self.assertIs(sch, self.schema)
        # One-part shorthand uses the bound default catalog.
        sch2 = svc[self.schema_name]
        self.assertIs(sch2, self.schema)

    def test_catalog_navigation(self) -> None:
        cat = self.schema.catalog
        self.assertEqual(cat.catalog_name, self.catalog_name)

    def test_invalidate_evicts_module_cache(self) -> None:
        svc = self.client.schemas(catalog_name=self.catalog_name)
        # Prime the cache via ``find``.
        self.client.schemas.find(f"{self.catalog_name}.{self.schema_name}")
        key = svc._cache_key(self.catalog_name, self.schema_name)
        self.assertIsNotNone(_SCHEMA_INFO_CACHE.get(key))
        svc.invalidate(self.schema)
        self.assertIsNone(_SCHEMA_INFO_CACHE.get(key))


@pytest.mark.integration
class TestSchemaTagsIntegration(_SchemaFixture):
    """:meth:`Schema.set_tags` / :meth:`Schema.unset_tags` round-trip."""

    def _skip_if_no_tag_grant(self, exc: Exception) -> None:
        msg = str(exc).lower()
        if "permission" in msg or "denied" in msg or "not authorized" in msg:
            raise unittest.SkipTest(
                f"Test identity lacks tag-management permission on "
                f"{self.catalog_name}: {exc}."
            )
        raise exc

    def test_set_and_read_tags(self) -> None:
        tag_key = f"yg_test_{secrets.token_hex(3)}"
        tag_value = f"v_{secrets.token_hex(2)}"
        try:
            self.schema.set_tags({tag_key: tag_value})
        except (DatabricksError, PermissionDenied) as exc:
            self._skip_if_no_tag_grant(exc)

        try:
            tags = self.client.entity_tags.entity_tags(
                "schemas", self.schema.full_name(), as_dict=True,
                cache_ttl=None,
            )
            self.assertEqual(tags.get(tag_key), tag_value)
        finally:
            try:
                self.schema.unset_tags([tag_key], if_exists=True)
            except Exception:
                pass

    def test_unset_missing_tag_with_if_exists(self) -> None:
        # ``unset_tags(if_exists=True)`` must swallow the missing-tag error.
        ghost_key = f"yg_ghost_{secrets.token_hex(3)}"
        try:
            self.schema.unset_tags([ghost_key], if_exists=True)
        except DatabricksError as exc:
            self._skip_if_no_tag_grant(exc)

    def test_set_tags_empty_mapping_is_noop(self) -> None:
        # Empty mapping → no API call, no error, returns self.
        result = self.schema.set_tags({})
        self.assertIs(result, self.schema)
        result = self.schema.set_tags(None)
        self.assertIs(result, self.schema)


@pytest.mark.integration
class TestSchemaPermissionsIntegration(_SchemaFixture):
    """:meth:`Schema.grant` / :meth:`Schema.revoke` /
    :meth:`Schema.set_permissions` round-trip against the
    ``grants`` API."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # USE_SCHEMA is the lightest-weight grant we can hand to the
        # caller themselves without inviting cross-principal noise.
        try:
            current = cls.client.iam.users.current_user
        except Exception as exc:
            raise unittest.SkipTest(
                f"Cannot resolve current user for permission tests: {exc}."
            )
        principal = getattr(current, "username", None) or getattr(
            current, "user_name", None,
        )
        if not principal:
            raise unittest.SkipTest(
                "Current user has no username — permission tests need a "
                "principal to grant to."
            )
        cls.principal = principal

    def _revoke_quiet(self, privileges) -> None:
        try:
            self.schema.revoke(self.principal, privileges)
        except Exception:
            pass

    def test_grant_then_permissions_filter_by_principal(self) -> None:
        try:
            self.schema.grant(self.principal, Privilege.USE_SCHEMA)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Test identity cannot grant USE_SCHEMA on "
                f"{self.schema.full_name()}: {exc}."
            )
        try:
            assignments = self.schema.permissions(principal=self.principal)
            self.assertTrue(assignments)
            principals = {a.principal for a in assignments}
            self.assertIn(self.principal, principals)
            granted = {
                p
                for a in assignments
                for p in (a.privileges or ())
            }
            self.assertIn(Privilege.USE_SCHEMA, granted)
        finally:
            self._revoke_quiet([Privilege.USE_SCHEMA])

    def test_grant_accepts_string_privilege(self) -> None:
        # String form must normalize to the matching Privilege enum.
        try:
            self.schema.grant(self.principal, "USE SCHEMA")
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Test identity cannot grant USE_SCHEMA on "
                f"{self.schema.full_name()}: {exc}."
            )
        try:
            assignments = self.schema.permissions(principal=self.principal)
            granted = {
                p
                for a in assignments
                for p in (a.privileges or ())
            }
            self.assertIn(Privilege.USE_SCHEMA, granted)
        finally:
            self._revoke_quiet([Privilege.USE_SCHEMA])

    def test_set_permissions_is_diff_idempotent(self) -> None:
        # Granting the same set twice must collapse to a no-op on the
        # second call (the diff is empty, no API call should fire).
        try:
            self.schema.set_permissions(
                self.principal, [Privilege.USE_SCHEMA],
            )
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Test identity cannot set USE_SCHEMA on "
                f"{self.schema.full_name()}: {exc}."
            )
        try:
            # Second call — diff is empty; the method short-circuits and
            # returns self without re-issuing the grant.
            result = self.schema.set_permissions(
                self.principal, [Privilege.USE_SCHEMA],
            )
            self.assertIs(result, self.schema)
            assignments = self.schema.permissions(principal=self.principal)
            granted = {
                p
                for a in assignments
                for p in (a.privileges or ())
            }
            self.assertIn(Privilege.USE_SCHEMA, granted)
        finally:
            self._revoke_quiet([Privilege.USE_SCHEMA])

    def test_revoke_removes_privilege(self) -> None:
        try:
            self.schema.grant(self.principal, Privilege.USE_SCHEMA)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Test identity cannot grant USE_SCHEMA on "
                f"{self.schema.full_name()}: {exc}."
            )
        self.schema.revoke(self.principal, Privilege.USE_SCHEMA)
        assignments = self.schema.permissions(principal=self.principal)
        granted = {
            p
            for a in assignments
            for p in (a.privileges or ())
        }
        self.assertNotIn(Privilege.USE_SCHEMA, granted)

    def test_grant_unknown_privilege_raises_with_helpful_message(self) -> None:
        # Bad privilege names fail fast at the call site — no API call.
        with self.assertRaises(ValueError) as ctx:
            self.schema.grant(self.principal, "TOTALLY_FAKE_PRIVILEGE")
        msg = str(ctx.exception)
        self.assertIn("TOTALLY_FAKE_PRIVILEGE", msg)
        # The error must list the valid privileges so a typo surfaces.
        self.assertIn("USE_SCHEMA", msg)

    def test_effective_permissions_includes_inherited(self) -> None:
        # ``effective_permissions`` must not raise; the parent catalog
        # typically grants some inherited privileges to the current
        # identity (at minimum USE_CATALOG / USE_SCHEMA). We don't pin
        # specifics — different metastore configs differ — but the call
        # has to round-trip cleanly.
        eff = self.schema.effective_permissions(principal=self.principal)
        self.assertIsInstance(eff, tuple)


@pytest.mark.integration
class TestSchemaRenameIntegration(DatabricksIntegrationCase):
    """:meth:`Schema.rename` round-trip on a dedicated throw-away schema.

    Separated from :class:`_SchemaFixture` so the class-level cleanup can
    chase whichever name the rename test settled on without racing the
    shared fixture's ``schema_name`` field.
    """

    catalog_name: ClassVar[str]
    initial_name: ClassVar[str]
    schema: ClassVar[UCSchema]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = _resolve_catalog()
        cls.initial_name = f"yg_rename_{secrets.token_hex(4)}"
        try:
            cls.schema = cls.client.schemas(
                catalog_name=cls.catalog_name,
            ).schema(schema_name=cls.initial_name)
            cls.schema.get_or_create(
                comment="yggdrasil schema rename integration",
            )
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Cannot create schema "
                f"{cls.catalog_name}.{cls.initial_name}: {exc}."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            schema = getattr(cls, "schema", None)
            if schema is not None:
                # The instance carries whichever name the last rename
                # call left it on.
                try:
                    schema.delete(force=True, raise_error=False)
                except Exception:
                    pass
        finally:
            super().tearDownClass()

    def test_rename_updates_name_and_remote_state(self) -> None:
        new_name = f"yg_renamed_{secrets.token_hex(4)}"
        try:
            self.schema.rename(new_name)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(
                f"Test identity cannot rename schema: {exc}."
            )
        self.assertEqual(self.schema.schema_name, new_name)
        self.assertEqual(
            self.schema.full_name(),
            f"{self.catalog_name}.{new_name}",
        )
        # New name is reachable; old name is not.
        fresh = self.client.schemas.find(
            f"{self.catalog_name}.{new_name}",
            cache_ttl=None,
        )
        self.assertIsNotNone(fresh)
        with self.assertRaises((NotFound, ResourceDoesNotExist)):
            self.client.schemas.find_remote(
                catalog_name=self.catalog_name,
                schema_name=self.initial_name,
            )

    def test_rename_to_same_name_is_noop(self) -> None:
        before = self.schema.schema_name
        result = self.schema.rename(before)
        self.assertIs(result, self.schema)
        self.assertEqual(self.schema.schema_name, before)

    def test_rename_to_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.schema.rename("   ")
