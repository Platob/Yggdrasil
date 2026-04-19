"""Unit tests for :func:`yggdrasil.arrow.tzdata.ensure_tzdata`.

These tests monkeypatch ``os.name`` so the Windows-only paths run on any
platform, and stub out :func:`pyarrow.util.download_tzdata_on_windows` to
keep the tests hermetic.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.arrow import tzdata as tzdata_mod


class TestEnsureTzdata(ArrowTestCase):
    require_parquet = False

    def setUp(self):
        super().setUp()
        tzdata_mod._ENSURED = None

    def tearDown(self):
        tzdata_mod._ENSURED = None
        super().tearDown()

    def test_noop_on_non_windows(self):
        """Non-Windows platforms short-circuit before touching pyarrow."""
        downloader = MagicMock()
        with patch.object(tzdata_mod.os, "name", "posix"), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader):
            self.assertTrue(tzdata_mod.ensure_tzdata())
        downloader.assert_not_called()

    def test_canary_shortcircuits(self):
        """Healthy env: canary passes, downloader never runs."""
        downloader = MagicMock()
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", return_value=True), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader):
            self.assertTrue(tzdata_mod.ensure_tzdata())
        downloader.assert_not_called()

    def test_partial_install_is_cleaned(self):
        """A tzdata dir without a 'version' file is removed before redownload."""
        partial = self.tmp_path / "tzdata"
        partial.mkdir()
        (partial / "africa").write_text("stale")

        downloader = MagicMock()
        canary_calls = iter([False, True])
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", side_effect=lambda: next(canary_calls)), \
             patch.object(tzdata_mod, "_default_tzdata_path", return_value=partial), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader):
            self.assertTrue(tzdata_mod.ensure_tzdata())

        downloader.assert_called_once()
        self.assertFalse(partial.exists())

    def test_complete_install_is_preserved(self):
        """A tzdata dir with a 'version' file is not wiped."""
        good = self.tmp_path / "tzdata"
        good.mkdir()
        (good / "version").write_text("2024a")
        (good / "africa").write_text("data")

        canary_calls = iter([False, True])
        downloader = MagicMock()
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", side_effect=lambda: next(canary_calls)), \
             patch.object(tzdata_mod, "_default_tzdata_path", return_value=good), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader):
            self.assertTrue(tzdata_mod.ensure_tzdata())

        self.assertTrue((good / "version").is_file())
        downloader.assert_called_once()

    def test_download_failure_returns_false(self):
        """Downloader raising does not propagate; returns False and logs."""
        downloader = MagicMock(side_effect=RuntimeError("network down"))
        installer = MagicMock(return_value=False)
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", return_value=False), \
             patch.object(tzdata_mod, "_default_tzdata_path", return_value=self.tmp_path / "x"), \
             patch.object(tzdata_mod, "_install_tzdata_package", installer), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader):
            self.assertFalse(tzdata_mod.ensure_tzdata())
        installer.assert_called_once()

    def test_canary_still_failing_returns_false(self):
        """Downloader succeeds but the canary still fails — return False."""
        downloader = MagicMock()
        installer = MagicMock(return_value=False)
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", return_value=False), \
             patch.object(tzdata_mod, "_default_tzdata_path", return_value=self.tmp_path / "x"), \
             patch.object(tzdata_mod, "_install_tzdata_package", installer), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader):
            self.assertFalse(tzdata_mod.ensure_tzdata())
        downloader.assert_called_once()
        installer.assert_called_once()

    def test_idempotent_after_success(self):
        """Once cached, repeat calls skip the downloader."""
        downloader = MagicMock()
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", return_value=True), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader):
            self.assertTrue(tzdata_mod.ensure_tzdata())
            self.assertTrue(tzdata_mod.ensure_tzdata())
            self.assertTrue(tzdata_mod.ensure_tzdata())
        downloader.assert_not_called()

    def test_force_reruns(self):
        """force=True bypasses the cache."""
        downloader = MagicMock()
        canary = MagicMock(side_effect=[True, False, True])
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", canary), \
             patch.object(tzdata_mod, "_default_tzdata_path", return_value=self.tmp_path / "x"), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader):
            self.assertTrue(tzdata_mod.ensure_tzdata())
            self.assertTrue(tzdata_mod.ensure_tzdata(force=True))
        downloader.assert_called_once()

    def test_default_path_env_override(self):
        """PYARROW_TZDATA_PATH overrides the default location."""
        custom = self.tmp_path / "custom-tz"
        with patch.dict(tzdata_mod.os.environ, {"PYARROW_TZDATA_PATH": str(custom)}):
            self.assertEqual(tzdata_mod._default_tzdata_path(), custom)

    def test_default_path_falls_back_to_home(self):
        """Without the env var, the default is ~/Downloads/tzdata."""
        env_without = {
            k: v for k, v in tzdata_mod.os.environ.items()
            if k != "PYARROW_TZDATA_PATH"
        }
        with patch.dict(tzdata_mod.os.environ, env_without, clear=True):
            self.assertEqual(
                tzdata_mod._default_tzdata_path(),
                Path.home() / "Downloads" / "tzdata",
            )
