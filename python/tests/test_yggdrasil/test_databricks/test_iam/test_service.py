"""Unit tests for :mod:`yggdrasil.databricks.iam.service`.

Covers :class:`IAMUsers` and :class:`IAMGroups`:

* lazy ``current_user`` lookup, caching, and the runtime-fallback path;
* ``list`` paging / filtering via the workspace / account SDK;
* ``create`` / ``delete`` routing per ``ClientType``;
* OAuth local-cache helpers.

All SDK boundaries are autospec'd mocks — no network is touched.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from databricks.sdk.client_types import ClientType
from databricks.sdk.errors import DatabricksError, PermissionDenied
from databricks.sdk.service.iam import (
    ComplexValue,
    Group as GroupV1,
    User as UserV1,
)

from yggdrasil.databricks.iam.resource import IAMGroup, IAMUser
from yggdrasil.databricks.iam.service import IAMGroups, IAMUsers
from yggdrasil.databricks.tests import DatabricksTestCase


class _IAMServiceTestCase(DatabricksTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.users: IAMUsers = self.client.iam.users
        self.groups: IAMGroups = self.client.iam.groups
        # Lock down the default client type so the test isn't subject to
        # MagicMock's default-attribute behaviour on the config mocks.
        self.workspace_config.client_type = ClientType.WORKSPACE
        self.account_config.client_type = ClientType.WORKSPACE


# =========================================================================
# IAMUsers.current_user
# =========================================================================


class TestCurrentUser(_IAMServiceTestCase):
    """``IAMUsers.current_user`` resolves and caches the calling principal."""

    def test_resolves_from_workspace_client_current_user_me(self) -> None:
        self.workspace_client.current_user.me.return_value = UserV1(
            id="u-1",
            display_name="Alice",
            user_name="alice@example.com",
            active=True,
        )
        user = self.users.current_user
        self.assertIsInstance(user, IAMUser)
        self.assertEqual(user.id, "u-1")
        self.assertEqual(user.username, "alice@example.com")
        self.workspace_client.current_user.me.assert_called_once_with()

    def test_caches_result_via_lazy_property(self) -> None:
        self.workspace_client.current_user.me.return_value = UserV1(
            id="u-1", display_name="Alice", user_name="alice@x.com",
        )
        first = self.users.current_user
        second = self.users.current_user
        self.assertIs(first, second)
        # Only the first access hits the SDK.
        self.workspace_client.current_user.me.assert_called_once()

    def test_falls_back_to_runtime_principal_on_databricks_error(self) -> None:
        # Non-browser auth path: log + return synthetic principal so
        # callers don't crash on local dev / CI where ``me()`` may 403.
        object.__setattr__(self.client, "auth_type", "pat")
        self.workspace_client.current_user.me.side_effect = DatabricksError("boom")
        user = self.users.current_user
        self.assertEqual(user.id, "databricks-runtime")

    def test_external_browser_auth_re_raises_after_resetting_cache(self) -> None:
        # External-browser auth: a stale OAuth token is the most likely
        # culprit, so the service wipes the local cache and re-raises so
        # the caller can re-auth.
        object.__setattr__(self.client, "auth_type", "external-browser")
        self.workspace_client.current_user.me.side_effect = DatabricksError("boom")
        with patch.object(self.users, "reset_local_cache") as reset:
            with self.assertRaises(DatabricksError):
                _ = self.users.current_user
            reset.assert_called_once()


# =========================================================================
# IAMUsers.list
# =========================================================================


class TestIAMUsersList(_IAMServiceTestCase):

    def test_list_no_filters_calls_workspace_users_list(self) -> None:
        self.workspace_client.users.list.return_value = iter([
            UserV1(id="u-1", display_name="Alice", user_name="alice@x.com"),
            UserV1(id="u-2", display_name="Bob", user_name="bob@x.com"),
        ])
        out = list(self.users.list())
        self.assertEqual([u.id for u in out], ["u-1", "u-2"])
        self.workspace_client.users.list.assert_called_once_with(filter=None)

    def test_list_with_name_and_user_name_builds_scim_filter(self) -> None:
        self.workspace_client.users.list.return_value = iter([])
        list(self.users.list(name="Alice", user_name="alice@x.com"))
        kwargs = self.workspace_client.users.list.call_args.kwargs
        self.assertIn('displayName eq "Alice"', kwargs["filter"])
        self.assertIn('userName eq "alice@x.com"', kwargs["filter"])
        self.assertIn(" and ", kwargs["filter"])

    def test_list_account_client_type_dispatches_to_account_client(self) -> None:
        self.account_client.users.list.return_value = iter([
            UserV1(id="u-1", display_name="Alice", user_name="alice@x.com"),
        ])
        out = list(self.users.list(client_type=ClientType.ACCOUNT))
        self.assertEqual(out[0].id, "u-1")
        self.account_client.users.list.assert_called_once_with(filter=None)
        self.workspace_client.users.list.assert_not_called()

    def test_list_respects_limit(self) -> None:
        self.workspace_client.users.list.return_value = iter([
            UserV1(id=f"u-{i}", display_name=f"User{i}", user_name=f"u{i}@x.com")
            for i in range(10)
        ])
        out = list(self.users.list(limit=3))
        self.assertEqual(len(out), 3)


# =========================================================================
# IAMUsers.create
# =========================================================================


class TestIAMUsersCreate(_IAMServiceTestCase):

    def test_create_workspace_calls_workspace_users_create(self) -> None:
        self.workspace_client.users.create.return_value = UserV1(
            id="u-1", display_name="Alice", user_name="alice@x.com", active=True,
        )
        user = self.users.create(
            "Alice",
            client_type=ClientType.WORKSPACE,
            user_name="alice@x.com",
        )
        self.workspace_client.users.create.assert_called_once()
        kwargs = self.workspace_client.users.create.call_args.kwargs
        self.assertEqual(kwargs["display_name"], "Alice")
        self.assertTrue(kwargs["active"])
        self.assertEqual(user.id, "u-1")

    def test_create_account_calls_account_users_create(self) -> None:
        self.account_client.users.create.return_value = UserV1(
            id="u-1", display_name="Alice", user_name="alice@x.com", active=True,
        )
        self.users.create(
            "Alice",
            client_type=ClientType.ACCOUNT,
            user_name="alice@x.com",
        )
        self.account_client.users.create.assert_called_once()
        self.workspace_client.users.create.assert_not_called()

    def test_create_with_user_like_object_uses_parse_branch(self) -> None:
        self.workspace_client.users.create.return_value = UserV1(
            id="u-1", display_name="Alice", user_name="alice@x.com",
        )
        seed = IAMUser(
            service=self.users,
            name="Alice",
            username="alice@x.com",
            external_id="ext-1",
        )
        self.users.create(
            "ignored-when-user-passed",
            client_type=ClientType.WORKSPACE,
            user=seed,
        )
        kwargs = self.workspace_client.users.create.call_args.kwargs
        # ``user`` overrides the positional ``name`` — display_name comes
        # from the parsed seed, not the bare argument.
        self.assertEqual(kwargs["display_name"], "Alice")
        self.assertEqual(kwargs["external_id"], "ext-1")


# =========================================================================
# IAMGroups.create
# =========================================================================


class TestIAMGroupsCreate(_IAMServiceTestCase):

    def _wire_current_user(self) -> None:
        """Make ``current_user`` resolve cleanly so ``create`` can default-fill members."""
        self.workspace_client.current_user.me.return_value = UserV1(
            id="me", display_name="Me", user_name="me@x.com",
        )

    def test_create_workspace_uses_workspace_groups_create(self) -> None:
        self._wire_current_user()
        self.workspace_client.groups.create.return_value = GroupV1(
            id="g-1", display_name="Admins",
        )
        group = self.groups.create("Admins", client_type=ClientType.WORKSPACE)
        self.workspace_client.groups.create.assert_called_once()
        kwargs = self.workspace_client.groups.create.call_args.kwargs
        self.assertEqual(kwargs["display_name"], "Admins")
        # Workspace path stamps the legacy ``WorkspaceGroup`` resource_type.
        self.assertEqual(kwargs["meta"].resource_type, "WorkspaceGroup")
        self.assertEqual(group.id, "g-1")

    def test_create_account_uses_account_groups_create(self) -> None:
        self._wire_current_user()
        self.account_client.groups.create.return_value = GroupV1(
            id="g-1", display_name="Admins",
        )
        self.groups.create(
            "Admins",
            client_type=ClientType.ACCOUNT,
            members=[IAMUser(service=self.users, id="u-1", name="Alice")],
        )
        self.account_client.groups.create.assert_called_once()
        kwargs = self.account_client.groups.create.call_args.kwargs
        self.assertEqual(kwargs["meta"].resource_type, "Group")
        self.assertEqual(kwargs["display_name"], "Admins")

    def test_create_defaults_members_to_current_user(self) -> None:
        # When ``members`` is omitted, the calling user is auto-added —
        # otherwise the freshly-created workspace group could be ownerless.
        self._wire_current_user()
        self.workspace_client.groups.create.return_value = GroupV1(
            id="g-1", display_name="Admins",
        )
        self.groups.create("Admins", client_type=ClientType.WORKSPACE)
        kwargs = self.workspace_client.groups.create.call_args.kwargs
        self.assertIsNotNone(kwargs["members"])
        self.assertEqual(kwargs["members"][0].value, "me")

    def test_create_explicit_members_skip_current_user_default(self) -> None:
        self.workspace_client.groups.create.return_value = GroupV1(
            id="g-1", display_name="Admins",
        )
        explicit = IAMUser(service=self.users, id="u-1", name="Alice")
        self.groups.create(
            "Admins",
            client_type=ClientType.WORKSPACE,
            members=[explicit],
        )
        kwargs = self.workspace_client.groups.create.call_args.kwargs
        self.assertEqual([m.value for m in kwargs["members"]], ["u-1"])
        # The current-user shortcut is NOT consulted on this path.
        self.workspace_client.current_user.me.assert_not_called()

    def test_create_permission_denied_re_wrapped_with_context(self) -> None:
        # The raw SDK ``PermissionDenied`` carries the bare error string;
        # the service rewrites it with the group name + client_type so
        # callers don't have to grep the message to figure out which
        # operation failed.
        self._wire_current_user()
        self.workspace_client.groups.create.side_effect = PermissionDenied("nope")
        with self.assertRaises(PermissionDenied) as info:
            self.groups.create("Admins", client_type=ClientType.WORKSPACE)
        self.assertIn("'Admins'", str(info.exception))
        self.assertIn("workspace", str(info.exception).lower())


# =========================================================================
# IAMGroups.delete
# =========================================================================


class TestIAMGroupsDelete(_IAMServiceTestCase):

    def test_delete_with_group_object_uses_its_client_type(self) -> None:
        group = IAMGroup(
            service=self.groups,
            id="g-1",
            name="Admins",
            client_type=ClientType.ACCOUNT,
        )
        self.groups.delete(group)
        self.account_client.groups.delete.assert_called_once_with("g-1")
        self.workspace_client.groups.delete.assert_not_called()

    def test_delete_with_group_id_uses_default_client_type(self) -> None:
        # ``default_client_type`` is read from the workspace config we set
        # in ``setUp`` (WORKSPACE), so the bare-id path routes there.
        self.groups.delete("g-1", group_id="g-1")
        self.workspace_client.groups.delete.assert_called_once_with("g-1")

    def test_delete_with_explicit_client_type_overrides_object_default(self) -> None:
        group = IAMGroup(
            service=self.groups,
            id="g-1",
            name="Admins",
            client_type=ClientType.ACCOUNT,
        )
        self.groups.delete(group, client_type=ClientType.WORKSPACE)
        self.workspace_client.groups.delete.assert_called_once_with("g-1")
        self.account_client.groups.delete.assert_not_called()

    def test_delete_without_obj_or_id_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "obj or group_id"):
            self.groups.delete(None)


# =========================================================================
# IAMGroups.list
# =========================================================================


class TestIAMGroupsList(_IAMServiceTestCase):

    def test_list_no_filter_uses_workspace_groups_list(self) -> None:
        self.workspace_client.groups.list.return_value = iter([
            GroupV1(id="g-1", display_name="Admins"),
        ])
        out = list(self.groups.list())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].id, "g-1")
        self.workspace_client.groups.list.assert_called_once_with(filter=None)

    def test_list_name_filter_builds_displayname_eq(self) -> None:
        self.workspace_client.groups.list.return_value = iter([])
        list(self.groups.list(name="Admins"))
        kwargs = self.workspace_client.groups.list.call_args.kwargs
        self.assertEqual(kwargs["filter"], 'displayName eq "Admins"')

    def test_list_account_client_type_dispatches_to_account_client(self) -> None:
        self.account_client.groups.list.return_value = iter([
            GroupV1(id="g-1", display_name="Admins"),
        ])
        list(self.groups.list(client_type=ClientType.ACCOUNT))
        self.account_client.groups.list.assert_called_once_with(filter=None)
        self.workspace_client.groups.list.assert_not_called()

    def test_list_respects_limit(self) -> None:
        self.workspace_client.groups.list.return_value = iter([
            GroupV1(id=f"g-{i}", display_name=f"G{i}") for i in range(5)
        ])
        out = list(self.groups.list(limit=2))
        self.assertEqual(len(out), 2)

    def test_list_swallows_databricks_error_when_raise_error_false(self) -> None:
        # Callers that just want "best effort" listings can suppress the
        # SDK error and get an empty iterator instead.
        self.workspace_client.groups.list.side_effect = DatabricksError("boom")
        out = list(self.groups.list(raise_error=False))
        self.assertEqual(out, [])

    def test_list_raises_databricks_error_by_default(self) -> None:
        self.workspace_client.groups.list.side_effect = DatabricksError("boom")
        with self.assertRaises(DatabricksError):
            list(self.groups.list())


# =========================================================================
# IAMUsers OAuth cache helpers
# =========================================================================


class TestLocalOAuthCache(_IAMServiceTestCase):
    """``local_cache_token_path`` / ``reset_local_cache`` manage on-disk tokens."""

    def _patch_local_config_folder(self, root):
        return patch(
            "yggdrasil.databricks.client.DatabricksClient.local_config_folder",
            new_callable=lambda: property(lambda _self: root),
        )

    def test_local_cache_token_path_returns_none_when_oauth_dir_missing(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            with self._patch_local_config_folder(Path(td)):
                self.assertIsNone(self.users.local_cache_token_path())

    def test_local_cache_token_path_returns_first_file_sorted(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "oauth").mkdir()
            (root / "oauth" / "b.json").write_text("{}")
            first = root / "oauth" / "a.json"
            first.write_text("{}")
            with self._patch_local_config_folder(root):
                self.assertEqual(self.users.local_cache_token_path(), str(first))

    def test_reset_local_cache_removes_token_file(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "oauth").mkdir()
            token = root / "oauth" / "a.json"
            token.write_text("{}")
            with self._patch_local_config_folder(root):
                self.users.reset_local_cache()
            self.assertFalse(token.exists())

    def test_reset_local_cache_noop_when_no_token(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            with self._patch_local_config_folder(Path(td)):
                # No oauth subdir → silently no-op, must not raise.
                self.users.reset_local_cache()


# =========================================================================
# IAM root service
# =========================================================================


class TestIAMRootService(_IAMServiceTestCase):
    """``IAM.users`` / ``IAM.groups`` lazy-cache their sub-services."""

    def test_users_property_returns_iamusers_bound_to_same_client(self) -> None:
        self.assertIsInstance(self.client.iam.users, IAMUsers)
        self.assertIs(self.client.iam.users.client, self.client)

    def test_groups_property_returns_iamgroups_bound_to_same_client(self) -> None:
        self.assertIsInstance(self.client.iam.groups, IAMGroups)
        self.assertIs(self.client.iam.groups.client, self.client)

    def test_users_property_is_cached(self) -> None:
        self.assertIs(self.client.iam.users, self.client.iam.users)

    def test_groups_property_is_cached(self) -> None:
        self.assertIs(self.client.iam.groups, self.client.iam.groups)
