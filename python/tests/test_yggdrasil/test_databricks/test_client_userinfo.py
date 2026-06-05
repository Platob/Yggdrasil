"""Tests for environment-aware defaults on :class:`DatabricksClient`."""
from __future__ import annotations

from unittest.mock import patch

from yggdrasil.databricks.tests import DatabricksTestCase


class TestUserScopedName(DatabricksTestCase):
    """``DatabricksClient.user_scoped_name`` derives a per-user slug."""

    def test_uses_email_local_part_when_available(self):
        with patch(
            "yggdrasil.environ.userinfo.UserInfo.current",
        ) as mock_current:
            mock_info = mock_current.return_value
            mock_info.email = "alice@example.com"
            mock_info.key = "alice"
            mock_info.hostname = "alice-mbp"

            name = self.client.user_scoped_name("Yggdrasil")
            self.assertEqual(name, "Yggdrasil-alice")

    def test_falls_back_to_whoami_when_no_email(self):
        with patch(
            "yggdrasil.environ.userinfo.UserInfo.current",
        ) as mock_current:
            mock_info = mock_current.return_value
            mock_info.email = None
            mock_info.key = "bob"
            mock_info.hostname = "bob-laptop"

            name = self.client.user_scoped_name("All Purpose")
            self.assertEqual(name, "All Purpose-bob")

    def test_falls_back_to_hostname(self):
        with patch(
            "yggdrasil.environ.userinfo.UserInfo.current",
        ) as mock_current:
            mock_info = mock_current.return_value
            mock_info.email = None
            mock_info.key = "unknown"
            mock_info.hostname = "ci-runner"

            name = self.client.user_scoped_name("Yggdrasil")
            self.assertEqual(name, "Yggdrasil-ci-runner")

    def test_truncates_to_max_length(self):
        with patch(
            "yggdrasil.environ.userinfo.UserInfo.current",
        ) as mock_current:
            mock_info = mock_current.return_value
            mock_info.email = None
            mock_info.key = "verylongusernamefoundinsomesystems"
            mock_info.hostname = "host"

            name = self.client.user_scoped_name("X" * 50, max_length=20)
            self.assertLessEqual(len(name), 20)

    def test_returns_base_when_no_user_info(self):
        # When every candidate is empty / "unknown", we fall back to the
        # bare base — better than a broken "Yggdrasil-unknown" pool name.
        with patch(
            "yggdrasil.environ.userinfo.UserInfo.current",
        ) as mock_current:
            mock_info = mock_current.return_value
            mock_info.email = None
            mock_info.key = "unknown"
            mock_info.hostname = None

            name = self.client.user_scoped_name("Yggdrasil")
            self.assertEqual(name, "Yggdrasil")


class TestDefaultTagsEnrichment(DatabricksTestCase):
    """``default_tags(update=False)`` includes Owner / Hostname / User."""

    def test_create_tags_include_userinfo(self):
        with patch(
            "yggdrasil.environ.userinfo.UserInfo.current",
        ) as mock_current:
            mock_info = mock_current.return_value
            mock_info.email = "carol@example.com"
            mock_info.key = "carol"
            mock_info.hostname = "carol-desktop"

            tags = self.client.default_tags(update=False)

            self.assertEqual(tags.get("Owner"), "carol@example.com")
            self.assertEqual(tags.get("User"), "carol")
            self.assertEqual(tags.get("Hostname"), "carol-desktop")

    def test_update_tags_skip_owner_block(self):
        # On update (update=True), default tags is empty — we don't want
        # to overwrite ownership tags set at create time.
        tags = self.client.default_tags(update=True)
        self.assertNotIn("Owner", tags)
        self.assertNotIn("Hostname", tags)
        self.assertNotIn("User", tags)
