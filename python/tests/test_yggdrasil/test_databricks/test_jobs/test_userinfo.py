"""Unit tests for UserInfo-derived job settings."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from databricks.sdk.service.jobs import GitProvider, JobEmailNotifications

from yggdrasil.databricks.jobs.userinfo import (
    userinfo_email_notifications,
    userinfo_git_source,
    userinfo_job_settings,
    userinfo_tags,
)
from yggdrasil.url import URL


def _info(
    *,
    email="alice@example.com",
    git_url="https://github.com/acme/widgets#abc1234567",
    info_url="https://my.databricks.net/?o=42#workspace/Users/me/script.py",
    git_branch="main",
):
    info = MagicMock()
    info.email = email
    info.cwd = "/repo"
    info.git_url = URL.from_str(git_url) if git_url else None
    info.url = URL.from_str(info_url) if info_url else None
    info.first_name = "Alice"
    info.last_name = "Smith"

    # Patch the private _git_info helper used by the userinfo helpers.
    def _branch_patch(_cwd):
        if git_branch:
            return {"git_branch": git_branch, "git_remote": git_url}
        return None

    return info, _branch_patch


class TestUserinfoGitSource:
    def test_builds_github_source_with_sha(self):
        info, branch_patch = _info()
        with patch("yggdrasil.databricks.jobs.userinfo._git_info", branch_patch):
            src = userinfo_git_source(info)
        assert src is not None
        assert src.git_url == "https://github.com/acme/widgets"
        assert src.git_provider == GitProvider.GIT_HUB
        assert src.git_commit == "abc1234567"
        # When commit is set, branch is dropped (commit pins it).
        assert src.git_branch is None

    def test_uses_branch_when_no_sha(self):
        info, branch_patch = _info(git_url="https://github.com/acme/widgets")
        with patch("yggdrasil.databricks.jobs.userinfo._git_info", branch_patch):
            src = userinfo_git_source(info)
        assert src is not None
        assert src.git_branch == "main"
        assert src.git_commit is None

    def test_unsupported_host_returns_none(self):
        info, branch_patch = _info(git_url="https://example.com/acme/widgets")
        with patch("yggdrasil.databricks.jobs.userinfo._git_info", branch_patch):
            assert userinfo_git_source(info) is None

    def test_missing_git_returns_none(self):
        info, branch_patch = _info(git_url=None)
        with patch("yggdrasil.databricks.jobs.userinfo._git_info", branch_patch):
            assert userinfo_git_source(info) is None


class TestUserinfoEmailNotifications:
    def test_returns_failure_notifications_by_default(self):
        info, _ = _info()
        notifs = userinfo_email_notifications(info)
        assert isinstance(notifs, JobEmailNotifications)
        assert notifs.on_failure == ["alice@example.com"]
        assert notifs.on_success is None

    def test_multiple_events(self):
        info, _ = _info()
        notifs = userinfo_email_notifications(
            info, events=("on_failure", "on_success"),
        )
        assert notifs.on_failure == ["alice@example.com"]
        assert notifs.on_success == ["alice@example.com"]

    def test_no_email_returns_none(self):
        info, _ = _info(email=None)
        assert userinfo_email_notifications(info) is None


class TestUserinfoTags:
    def test_includes_git_and_compute_metadata(self):
        info, branch_patch = _info()
        with patch("yggdrasil.databricks.jobs.userinfo._git_info", branch_patch):
            tags = userinfo_tags(info)
        assert tags["GitUrl"] == "https://github.com/acme/widgets"
        assert tags["GitCommit"] == "abc1234567"
        assert tags["GitBranch"] == "main"
        assert tags["StagedFrom"].startswith("https://my.databricks.net/")


class TestUserinfoJobSettings:
    def test_returns_full_dict_when_all_included(self):
        info, branch_patch = _info()
        with patch("yggdrasil.databricks.jobs.userinfo._git_info", branch_patch):
            settings = userinfo_job_settings(info)
        assert set(settings) >= {"git_source", "email_notifications", "tags"}

    def test_toggles_drop_keys(self):
        info, branch_patch = _info()
        with patch("yggdrasil.databricks.jobs.userinfo._git_info", branch_patch):
            settings = userinfo_job_settings(
                info,
                include_git_source=False,
                include_notifications=False,
                include_tags=False,
            )
        assert settings == {}
