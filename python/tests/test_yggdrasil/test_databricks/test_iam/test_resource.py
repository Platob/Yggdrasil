"""Unit tests for :mod:`yggdrasil.databricks.iam.resource`.

Covers :class:`IAMUser` and :class:`IAMGroup`:

* construction (including the ``id=`` → ``object.__init__`` regression);
* parsing from SDK v1/v2 dataclasses, SCIM ``ComplexValue``, mappings,
  and strings;
* ``set_details`` field mapping per source type;
* membership semantics on :meth:`IAMGroup.add_member`.

No network is hit — the workspace / account SDK clients are autospec'd
mocks installed by :class:`DatabricksTestCase`. Tests that exercise
``sync`` / ``complex_value`` wire the relevant SDK return value
explicitly.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from databricks.sdk.client_types import ClientType
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.iam import (
    ComplexValue,
    Group as GroupV1,
    ResourceMeta,
    User as UserV1,
)
from databricks.sdk.service.iamv2 import Group as GroupV2, User as UserV2

from yggdrasil.databricks.iam.resource import IAMGroup, IAMUser
from yggdrasil.databricks.iam.service import IAMGroups, IAMUsers
from yggdrasil.databricks.tests import DatabricksTestCase


class _IAMResourceTestCase(DatabricksTestCase):
    """Shared fixture: real ``IAMUsers`` / ``IAMGroups`` bound to the mock client."""

    def setUp(self) -> None:
        super().setUp()
        self.users_service: IAMUsers = self.client.iam.users
        self.groups_service: IAMGroups = self.client.iam.groups
        # ``default_client_type`` is read from the (mocked) config in a few
        # parse paths; force a deterministic value so tests don't depend on
        # MagicMock's default attribute behavior.
        self.workspace_config.client_type = ClientType.WORKSPACE


# =========================================================================
# IAMUser construction
# =========================================================================


class TestIAMUserInit(_IAMResourceTestCase):
    """``IAMUser.__init__`` stores fields and survives ``id=`` regression."""

    def test_init_with_id_does_not_call_object_init_with_kwargs(self) -> None:
        # Regression: ``DatabricksResource.__init__`` used to forward
        # ``id=`` to ``object.__init__``, raising
        # ``TypeError: object.__init__() takes exactly one argument``.
        user = IAMUser(service=self.users_service, id="abc-123")
        self.assertEqual(user.id, "abc-123")
        self.assertIs(user.service, self.users_service)

    def test_init_stores_all_fields(self) -> None:
        user = IAMUser(
            service=self.users_service,
            id="u-1",
            name="Alice",
            username="alice@example.com",
            emails=["alice@example.com"],
            external_id="ext-1",
            active=False,
            client_type=ClientType.WORKSPACE,
        )
        self.assertEqual(user.id, "u-1")
        self.assertEqual(user.name, "Alice")
        self.assertEqual(user.username, "alice@example.com")
        self.assertEqual(user.emails, ["alice@example.com"])
        self.assertEqual(user.external_id, "ext-1")
        self.assertFalse(user.active)
        self.assertEqual(user.client_type, ClientType.WORKSPACE)

    def test_init_defaults_service_to_iamusers_current(self) -> None:
        # When ``service`` is omitted, the resource binds to the current
        # IAMUsers — and reaches the live ``DatabricksClient`` through it.
        IAMUsers._current = None
        try:
            user = IAMUser(id="u-2")
            self.assertIsInstance(user.service, IAMUsers)
            self.assertIs(user.service.client, self.client)
        finally:
            IAMUsers._current = None

    def test_email_returns_first_email(self) -> None:
        user = IAMUser(
            service=self.users_service,
            emails=["primary@x.com", "alt@x.com"],
        )
        self.assertEqual(user.email, "primary@x.com")

    def test_email_returns_none_when_no_emails(self) -> None:
        user = IAMUser(service=self.users_service)
        self.assertIsNone(user.email)

    def test_str_prefers_name_then_username_then_id(self) -> None:
        self.assertEqual(
            str(IAMUser(service=self.users_service, name="N", username="U", id="I")),
            "N",
        )
        self.assertEqual(
            str(IAMUser(service=self.users_service, username="U", id="I")),
            "U",
        )
        self.assertEqual(
            str(IAMUser(service=self.users_service, id="I")),
            "I",
        )
        self.assertEqual(
            str(IAMUser(service=self.users_service)),
            "unknown-user",
        )

    def test_repr_includes_classname_and_display(self) -> None:
        user = IAMUser(service=self.users_service, name="Alice")
        self.assertEqual(repr(user), "IAMUser<'Alice'>")


class TestIAMUserDatabricksRuntime(_IAMResourceTestCase):
    """``databricks_runtime()`` produces the synthetic runtime principal."""

    def test_returns_synthetic_principal(self) -> None:
        IAMUsers._current = None
        try:
            user = IAMUser.databricks_runtime()
            self.assertEqual(user.id, "databricks-runtime")
            self.assertEqual(user.name, "Databricks Runtime")
            self.assertEqual(user.username, "databricks-runtime")
            self.assertEqual(user.client_type, ClientType.ACCOUNT)
            self.assertTrue(user.active)
        finally:
            IAMUsers._current = None


# =========================================================================
# IAMUser.parse
# =========================================================================


class TestIAMUserParse(_IAMResourceTestCase):

    def test_parse_returns_iamuser_unchanged(self) -> None:
        original = IAMUser(service=self.users_service, id="u-1", name="Alice")
        result = IAMUser.from_(original, service=self.users_service)
        self.assertIs(result, original)

    def test_parse_userv1_copies_fields_via_set_details(self) -> None:
        sdk_user = UserV1(
            id="u-1",
            display_name="Alice",
            user_name="alice@example.com",
            external_id="ext-1",
            active=True,
            emails=[ComplexValue(value="alice@example.com")],
        )
        user = IAMUser.from_(
            sdk_user,
            service=self.users_service,
            client_type=ClientType.WORKSPACE,
        )
        self.assertEqual(user.id, "u-1")
        self.assertEqual(user.name, "Alice")
        self.assertEqual(user.username, "alice@example.com")
        self.assertEqual(user.emails, ["alice@example.com"])
        self.assertEqual(user.external_id, "ext-1")
        self.assertTrue(user.active)
        self.assertEqual(user.client_type, ClientType.WORKSPACE)

    def test_parse_userv1_active_none_treated_as_true(self) -> None:
        # ``UserV1.active is None`` is interpreted as active (the SCIM
        # default) — anything else would silently disable real users.
        sdk_user = UserV1(id="u-1", display_name="X", user_name="x@x.com", active=None)
        user = IAMUser.from_(sdk_user, service=self.users_service)
        self.assertTrue(user.active)

    def test_parse_userv2_maps_internal_id_and_username(self) -> None:
        sdk_user = UserV2(
            internal_id="u-2",
            username="bob@example.com",
            external_id="ext-2",
        )
        user = IAMUser.from_(sdk_user, service=self.users_service)
        self.assertEqual(user.id, "u-2")
        # v2 has no display_name → name is cleared.
        self.assertIsNone(user.name)
        self.assertEqual(user.username, "bob@example.com")
        self.assertEqual(user.emails, ["bob@example.com"])
        self.assertEqual(user.external_id, "ext-2")

    def test_parse_userv2_non_email_username_clears_emails(self) -> None:
        sdk_user = UserV2(internal_id="u-2", username="bob")
        user = IAMUser.from_(sdk_user, service=self.users_service)
        self.assertEqual(user.username, "bob")
        self.assertIsNone(user.emails)

    def test_parse_complex_value_uses_display_as_name_and_username(self) -> None:
        cv = ComplexValue(value="u-3", display="alice@example.com", ref="Users/u-3")
        user = IAMUser.from_(cv, service=self.users_service)
        self.assertEqual(user.id, "u-3")
        self.assertEqual(user.name, "alice@example.com")
        self.assertEqual(user.username, "alice@example.com")
        self.assertEqual(user.emails, ["alice@example.com"])

    def test_parse_complex_value_non_email_display_clears_emails(self) -> None:
        cv = ComplexValue(value="u-3", display="alice")
        user = IAMUser.from_(cv, service=self.users_service)
        self.assertIsNone(user.emails)

    def test_parse_mapping_uses_v1_fields(self) -> None:
        user = IAMUser.from_(
            {
                "id": "u-1",
                "display_name": "Alice",
                "user_name": "alice@example.com",
                "external_id": "ext-1",
            },
            service=self.users_service,
        )
        self.assertEqual(user.id, "u-1")
        self.assertEqual(user.name, "Alice")
        self.assertEqual(user.username, "alice@example.com")
        self.assertEqual(user.emails, ["alice@example.com"])
        self.assertEqual(user.external_id, "ext-1")

    def test_parse_mapping_uses_v2_fields(self) -> None:
        user = IAMUser.from_(
            {"internal_id": "u-2", "username": "bob@example.com"},
            service=self.users_service,
        )
        self.assertEqual(user.id, "u-2")
        self.assertEqual(user.username, "bob@example.com")

    def test_parse_mapping_uses_scim_fields(self) -> None:
        user = IAMUser.from_(
            {"value": "u-3", "display": "Alice"},
            service=self.users_service,
        )
        self.assertEqual(user.id, "u-3")
        self.assertEqual(user.name, "Alice")

    def test_parse_mapping_email_promoted_to_emails(self) -> None:
        user = IAMUser.from_(
            {"id": "u-1", "email": "alice@example.com"},
            service=self.users_service,
        )
        self.assertEqual(user.username, "alice@example.com")
        self.assertEqual(user.emails, ["alice@example.com"])

    def test_parse_mapping_emails_string_normalized_to_list(self) -> None:
        user = IAMUser.from_(
            {"id": "u-1", "emails": "alice@example.com"},
            service=self.users_service,
        )
        self.assertEqual(user.emails, ["alice@example.com"])

    def test_parse_string_email(self) -> None:
        user = IAMUser.from_("alice@example.com", service=self.users_service)
        self.assertEqual(user.username, "alice@example.com")
        self.assertEqual(user.emails, ["alice@example.com"])

    def test_parse_string_non_email(self) -> None:
        user = IAMUser.from_("alice", service=self.users_service)
        self.assertEqual(user.username, "alice")
        self.assertIsNone(user.emails)

    def test_parse_str_empty_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Value cannot be empty"):
            IAMUser.from_str("", service=self.users_service)

    def test_parse_unsupported_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported object type"):
            IAMUser.from_(12345, service=self.users_service)


# =========================================================================
# IAMUser.set_details / sync / complex_value
# =========================================================================


class TestIAMUserSetDetails(_IAMResourceTestCase):

    def test_set_details_from_iamuser_copies_every_field(self) -> None:
        src = IAMUser(
            service=self.users_service,
            id="u-1",
            name="Alice",
            username="alice@example.com",
            emails=["alice@example.com"],
            external_id="ext-1",
            active=False,
            client_type=ClientType.WORKSPACE,
        )
        dst = IAMUser(service=self.users_service)
        dst.set_details(src)
        self.assertEqual(dst.id, "u-1")
        self.assertEqual(dst.name, "Alice")
        self.assertEqual(dst.username, "alice@example.com")
        self.assertEqual(dst.emails, ["alice@example.com"])
        self.assertEqual(dst.external_id, "ext-1")
        self.assertFalse(dst.active)
        self.assertEqual(dst.client_type, ClientType.WORKSPACE)

    def test_set_details_unsupported_raises(self) -> None:
        user = IAMUser(service=self.users_service)
        with self.assertRaisesRegex(ValueError, "Unsupported user details type"):
            user.set_details(object())


class TestIAMUserSync(_IAMResourceTestCase):
    """``sync()`` resolves details against the relevant SDK client."""

    def test_sync_requires_id_or_username(self) -> None:
        user = IAMUser(service=self.users_service)
        with self.assertRaisesRegex(ValueError, "ID or username"):
            user.sync()

    def test_sync_by_username_resolves_via_list(self) -> None:
        resolved = UserV1(id="u-1", display_name="Alice", user_name="alice@x.com")
        self.workspace_client.users.list.return_value = iter([resolved])
        user = IAMUser(
            service=self.users_service,
            username="alice@x.com",
            client_type=ClientType.WORKSPACE,
        )
        out = user.sync()
        self.assertIs(out, user)
        self.assertEqual(user.id, "u-1")
        self.assertEqual(user.name, "Alice")

    def test_sync_by_username_raises_when_not_found(self) -> None:
        self.workspace_client.users.list.return_value = iter([])
        user = IAMUser(
            service=self.users_service,
            username="ghost@x.com",
            client_type=ClientType.WORKSPACE,
        )
        with self.assertRaises(ResourceDoesNotExist):
            user.sync()

    def test_sync_by_id_workspace_uses_workspace_client(self) -> None:
        self.workspace_client.users.get.return_value = UserV1(
            id="u-1", display_name="Alice", user_name="alice@x.com",
        )
        user = IAMUser(
            service=self.users_service,
            id="u-1",
            client_type=ClientType.WORKSPACE,
        )
        user.sync()
        self.workspace_client.users.get.assert_called_once_with(id="u-1")
        self.account_client.users.get.assert_not_called()

    def test_sync_by_id_account_uses_account_client(self) -> None:
        self.account_client.users.get.return_value = UserV1(
            id="u-1", display_name="Alice", user_name="alice@x.com",
        )
        user = IAMUser(
            service=self.users_service,
            id="u-1",
            client_type=ClientType.ACCOUNT,
        )
        user.sync()
        self.account_client.users.get.assert_called_once_with(id="u-1")
        self.workspace_client.users.get.assert_not_called()


class TestIAMUserComplexValue(_IAMResourceTestCase):
    """``complex_value`` rebuilds a SCIM payload, syncing if id is missing."""

    def test_returns_complex_value_when_id_present(self) -> None:
        user = IAMUser(
            service=self.users_service,
            id="u-1",
            name="Alice",
            username="alice@x.com",
        )
        cv = user.complex_value
        self.assertEqual(cv.value, "u-1")
        self.assertEqual(cv.display, "Alice")
        self.assertEqual(cv.ref, "Users/u-1")

    def test_falls_back_to_username_when_no_name(self) -> None:
        user = IAMUser(
            service=self.users_service,
            id="u-1",
            username="alice@x.com",
        )
        self.assertEqual(user.complex_value.display, "alice@x.com")

    def test_syncs_when_id_missing(self) -> None:
        self.workspace_client.users.list.return_value = iter([
            UserV1(id="u-9", display_name="Alice", user_name="alice@x.com"),
        ])
        user = IAMUser(
            service=self.users_service,
            username="alice@x.com",
            client_type=ClientType.WORKSPACE,
        )
        cv = user.complex_value
        self.assertEqual(cv.value, "u-9")


# =========================================================================
# IAMGroup
# =========================================================================


class TestIAMGroupInit(_IAMResourceTestCase):

    def test_init_stores_all_fields(self) -> None:
        member = IAMUser(service=self.users_service, id="u-1", name="Alice")
        group = IAMGroup(
            service=self.groups_service,
            id="g-1",
            name="Admins",
            account_id="acct-1",
            external_id="ext-g",
            client_type=ClientType.ACCOUNT,
            entitlements=["allow-cluster-create"],
            members=[member],
        )
        self.assertEqual(group.id, "g-1")
        self.assertEqual(group.name, "Admins")
        self.assertEqual(group.account_id, "acct-1")
        self.assertEqual(group.external_id, "ext-g")
        self.assertEqual(group.client_type, ClientType.ACCOUNT)
        self.assertEqual(group.entitlements, ["allow-cluster-create"])
        self.assertEqual(group.members, [member])

    def test_init_defaults_service_to_iamgroups_current(self) -> None:
        IAMGroups._current = None
        try:
            group = IAMGroup(id="g-1")
            self.assertIsInstance(group.service, IAMGroups)
            self.assertIs(group.service.client, self.client)
        finally:
            IAMGroups._current = None

    def test_str_prefers_name_then_id(self) -> None:
        self.assertEqual(
            str(IAMGroup(service=self.groups_service, name="N", id="I")),
            "N",
        )
        self.assertEqual(
            str(IAMGroup(service=self.groups_service, id="I")),
            "I",
        )
        self.assertEqual(
            str(IAMGroup(service=self.groups_service)),
            "unknown-group",
        )


class TestIAMGroupParse(_IAMResourceTestCase):

    def test_parse_returns_iamgroup_unchanged(self) -> None:
        original = IAMGroup(service=self.groups_service, id="g-1", name="Admins")
        result = IAMGroup.from_(original, service=self.groups_service)
        self.assertIs(result, original)

    def test_parse_groupv1_workspace_meta(self) -> None:
        sdk_group = GroupV1(
            id="g-1",
            display_name="Admins",
            external_id="ext-g",
            meta=ResourceMeta(resource_type="WorkspaceGroup"),
        )
        group = IAMGroup.from_(sdk_group, service=self.groups_service)
        self.assertEqual(group.id, "g-1")
        self.assertEqual(group.name, "Admins")
        self.assertEqual(group.external_id, "ext-g")
        self.assertEqual(group.client_type, ClientType.WORKSPACE)

    def test_parse_groupv1_account_meta(self) -> None:
        sdk_group = GroupV1(
            id="g-1",
            display_name="Admins",
            meta=ResourceMeta(resource_type="Group"),
        )
        group = IAMGroup.from_(sdk_group, service=self.groups_service)
        self.assertEqual(group.client_type, ClientType.ACCOUNT)

    def test_parse_groupv1_members_converted_to_iamusers(self) -> None:
        sdk_group = GroupV1(
            id="g-1",
            display_name="Admins",
            members=[ComplexValue(value="u-1", display="alice@x.com")],
        )
        group = IAMGroup.from_(sdk_group, service=self.groups_service)
        self.assertEqual(len(group.members), 1)
        self.assertIsInstance(group.members[0], IAMUser)
        self.assertEqual(group.members[0].id, "u-1")

    def test_parse_groupv2_maps_internal_id_and_group_name(self) -> None:
        sdk_group = GroupV2(
            internal_id="g-2",
            group_name="Admins",
            account_id="acct-1",
            external_id="ext-g",
        )
        group = IAMGroup.from_(sdk_group, service=self.groups_service)
        self.assertEqual(group.id, "g-2")
        self.assertEqual(group.name, "Admins")
        self.assertEqual(group.account_id, "acct-1")
        self.assertEqual(group.external_id, "ext-g")
        # v2 doesn't carry members/entitlements — explicitly cleared.
        self.assertIsNone(group.members)
        self.assertIsNone(group.entitlements)

    def test_parse_complex_value(self) -> None:
        cv = ComplexValue(value="g-3", display="Admins")
        group = IAMGroup.from_(cv, service=self.groups_service)
        self.assertEqual(group.id, "g-3")
        self.assertEqual(group.name, "Admins")
        self.assertIsNone(group.members)

    def test_parse_mapping_uses_scim_and_v1_v2_field_names(self) -> None:
        group = IAMGroup.from_(
            {
                "id": "g-1",
                "display_name": "Admins",
                "external_id": "ext-g",
                "account_id": "acct-1",
                "members": [{"id": "u-1", "display_name": "Alice"}],
            },
            service=self.groups_service,
            client_type=ClientType.ACCOUNT,
        )
        self.assertEqual(group.id, "g-1")
        self.assertEqual(group.name, "Admins")
        self.assertEqual(group.account_id, "acct-1")
        self.assertEqual(group.client_type, ClientType.ACCOUNT)
        self.assertEqual(len(group.members), 1)
        self.assertEqual(group.members[0].name, "Alice")

    def test_parse_mapping_empty_members_normalized_to_none(self) -> None:
        group = IAMGroup.from_(
            {"id": "g-1", "display_name": "Admins", "members": []},
            service=self.groups_service,
        )
        self.assertIsNone(group.members)

    def test_parse_unsupported_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported object type"):
            IAMGroup.from_(object(), service=self.groups_service)


class TestIAMGroupSetDetails(_IAMResourceTestCase):

    def test_set_details_from_iamgroup_copies_every_field(self) -> None:
        member = IAMUser(service=self.users_service, id="u-1")
        src = IAMGroup(
            service=self.groups_service,
            id="g-1",
            name="Admins",
            account_id="acct-1",
            external_id="ext-g",
            client_type=ClientType.ACCOUNT,
            entitlements=["allow-cluster-create"],
            members=[member],
        )
        dst = IAMGroup(service=self.groups_service)
        dst.set_details(src)
        self.assertEqual(dst.id, "g-1")
        self.assertEqual(dst.name, "Admins")
        self.assertEqual(dst.account_id, "acct-1")
        self.assertEqual(dst.external_id, "ext-g")
        self.assertEqual(dst.client_type, ClientType.ACCOUNT)
        self.assertEqual(dst.entitlements, ["allow-cluster-create"])
        self.assertEqual(dst.members, [member])

    def test_set_details_unsupported_raises(self) -> None:
        group = IAMGroup(service=self.groups_service)
        with self.assertRaisesRegex(ValueError, "Unsupported group details type"):
            group.set_details(object())


class TestIAMGroupAddMember(_IAMResourceTestCase):
    """``add_member`` parses, dedups, and (optionally) commits."""

    def test_add_member_initializes_members_list_when_none(self) -> None:
        group = IAMGroup(
            service=self.groups_service,
            id="g-1",
            name="Admins",
            client_type=ClientType.WORKSPACE,
            members=None,
        )
        new = IAMUser(service=self.users_service, id="u-1", name="Alice")
        # Avoid hitting workspace_client.groups.update on commit.
        self.workspace_client.groups.update.return_value = GroupV1(
            id="g-1", display_name="Admins",
        )
        group.add_member(new, commit=False)
        self.assertEqual(group.members, [new])

    def test_add_member_dedups_by_id(self) -> None:
        existing = IAMUser(service=self.users_service, id="u-1")
        group = IAMGroup(
            service=self.groups_service,
            id="g-1",
            name="Admins",
            client_type=ClientType.WORKSPACE,
            members=[existing],
        )
        duplicate = IAMUser(service=self.users_service, id="u-1", name="renamed")
        group.add_member(duplicate, commit=False)
        self.assertEqual(len(group.members), 1)
        self.assertIs(group.members[0], existing)

    def test_add_member_dedups_by_username(self) -> None:
        existing = IAMUser(
            service=self.users_service, id="u-1", username="alice@x.com",
        )
        group = IAMGroup(
            service=self.groups_service,
            id="g-1",
            name="Admins",
            client_type=ClientType.WORKSPACE,
            members=[existing],
        )
        # Same username, different id-less candidate. ``add_member`` would
        # otherwise call ``sync`` to resolve the id, so stub the SDK lookup
        # to return the same existing user — preserving dedup semantics.
        self.workspace_client.users.list.return_value = iter([
            UserV1(id="u-1", display_name="Alice", user_name="alice@x.com"),
        ])
        group.add_member("alice@x.com", commit=False)
        self.assertEqual(len(group.members), 1)

    def test_add_member_commit_calls_groups_update(self) -> None:
        existing_member = ComplexValue(value="u-0", display="Eve")
        self.workspace_client.groups.update.return_value = GroupV1(
            id="g-1",
            display_name="Admins",
            members=[existing_member],
            meta=ResourceMeta(resource_type="WorkspaceGroup"),
        )
        group = IAMGroup(
            service=self.groups_service,
            id="g-1",
            name="Admins",
            client_type=ClientType.WORKSPACE,
            members=None,
        )
        new = IAMUser(
            service=self.users_service,
            id="u-1",
            name="Alice",
        )
        group.add_member(new, commit=True)
        self.workspace_client.groups.update.assert_called_once()
        kwargs = self.workspace_client.groups.update.call_args.kwargs
        self.assertEqual(kwargs["id"], "g-1")
        self.assertEqual(kwargs["display_name"], "Admins")
        # The committed member payload is the user's SCIM ``ComplexValue``.
        self.assertEqual(len(kwargs["members"]), 1)
        self.assertEqual(kwargs["members"][0].value, "u-1")


class TestIAMGroupSyncRequiresId(_IAMResourceTestCase):

    def test_sync_without_id_raises(self) -> None:
        group = IAMGroup(service=self.groups_service, name="Admins")
        with self.assertRaisesRegex(ValueError, "ID to be committed"):
            group.sync()

    def test_sync_unsupported_client_type_raises(self) -> None:
        group = IAMGroup(
            service=self.groups_service,
            id="g-1",
            name="Admins",
            client_type=None,
        )
        # Force the dispatch to fall through to the "unsupported" branch.
        group.client_type = MagicMock(name="unknown")
        with self.assertRaisesRegex(ValueError, "Unsupported client type"):
            group.sync()
