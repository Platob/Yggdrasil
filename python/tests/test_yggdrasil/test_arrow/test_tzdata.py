"""Unit tests for :func:`yggdrasil.arrow.tzdata.ensure_tzdata`.

These tests monkeypatch ``os.name`` so the Windows-only paths run on any
platform, and stub out :func:`pyarrow.util.download_tzdata_on_windows` plus
the manual-download helper to keep the tests hermetic.
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

    def _windows_patches(self, *, path: Path | None = None):
        """Common context-manager patches for Windows-path tests."""
        stack = [
            patch.object(tzdata_mod.os, "name", "nt"),
        ]
        if path is not None:
            stack.append(
                patch.object(tzdata_mod, "_default_tzdata_path", return_value=path),
            )
        return stack

    def test_noop_on_non_windows(self):
        downloader = MagicMock()
        manual = MagicMock()
        with patch.object(tzdata_mod.os, "name", "posix"), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader), \
             patch.object(tzdata_mod, "_manual_download", manual):
            self.assertTrue(tzdata_mod.ensure_tzdata())
        downloader.assert_not_called()
        manual.assert_not_called()

    def test_canary_shortcircuits(self):
        downloader = MagicMock()
        manual = MagicMock()
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", return_value=True), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader), \
             patch.object(tzdata_mod, "_manual_download", manual):
            self.assertTrue(tzdata_mod.ensure_tzdata())
        downloader.assert_not_called()
        manual.assert_not_called()

    def test_partial_install_is_cleaned(self):
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

    def test_manual_fallback_runs_when_primary_fails(self):
        """pyarrow downloader raises → manual fallback runs and succeeds."""
        path = self.tmp_path / "tz"
        downloader = MagicMock(side_effect=RuntimeError("proxy blocked"))
        manual = MagicMock()
        canary_calls = iter([False, False, True])
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", side_effect=lambda: next(canary_calls)), \
             patch.object(tzdata_mod, "_default_tzdata_path", return_value=path), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader), \
             patch.object(tzdata_mod, "_manual_download", manual):
            self.assertTrue(tzdata_mod.ensure_tzdata())
        downloader.assert_called_once()
        manual.assert_called_once_with(path)

    def test_both_downloads_fail_returns_false(self):
        """Primary AND manual fail → log, return False, do not raise."""
        path = self.tmp_path / "tz"
        downloader = MagicMock(side_effect=RuntimeError("proxy blocked"))
        manual = MagicMock(side_effect=IOError("network down"))
        installer = MagicMock(return_value=False)
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", return_value=False), \
             patch.object(tzdata_mod, "_default_tzdata_path", return_value=path), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader), \
             patch.object(tzdata_mod, "_manual_download", manual), \
             patch.object(tzdata_mod, "_install_tzdata_package", installer):
            self.assertFalse(tzdata_mod.ensure_tzdata())
        downloader.assert_called_once()
        manual.assert_called_once()
        installer.assert_called_once()

    def test_raise_on_failure_raises(self):
        """raise_on_failure=True turns a final-canary failure into an exception."""
        path = self.tmp_path / "tz"
        downloader = MagicMock(side_effect=RuntimeError("proxy blocked"))
        manual = MagicMock(side_effect=IOError("network down"))
        installer = MagicMock(return_value=False)
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", return_value=False), \
             patch.object(tzdata_mod, "_default_tzdata_path", return_value=path), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader), \
             patch.object(tzdata_mod, "_manual_download", manual), \
             patch.object(tzdata_mod, "_install_tzdata_package", installer):
            with self.assertRaises(RuntimeError) as ctx:
                tzdata_mod.ensure_tzdata(raise_on_failure=True)
        self.assertIn("PYARROW_TZDATA_PATH", str(ctx.exception))

    def test_canary_still_failing_after_manual(self):
        """Both downloads 'succeed' but canary still fails — return False."""
        path = self.tmp_path / "tz"
        downloader = MagicMock()
        manual = MagicMock()
        installer = MagicMock(return_value=False)
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", return_value=False), \
             patch.object(tzdata_mod, "_default_tzdata_path", return_value=path), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader), \
             patch.object(tzdata_mod, "_manual_download", manual), \
             patch.object(tzdata_mod, "_install_tzdata_package", installer):
            self.assertFalse(tzdata_mod.ensure_tzdata())
        downloader.assert_called_once()
        manual.assert_called_once()
        installer.assert_called_once()

    def test_idempotent_after_success(self):
        downloader = MagicMock()
        with patch.object(tzdata_mod.os, "name", "nt"), \
             patch.object(tzdata_mod, "_canary", return_value=True), \
             patch.object(tzdata_mod.pyarrow.util, "download_tzdata_on_windows", downloader):
            self.assertTrue(tzdata_mod.ensure_tzdata())
            self.assertTrue(tzdata_mod.ensure_tzdata())
            self.assertTrue(tzdata_mod.ensure_tzdata())
        downloader.assert_not_called()

    def test_force_reruns(self):
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
        custom = self.tmp_path / "custom-tz"
        with patch.dict(tzdata_mod.os.environ, {"PYARROW_TZDATA_PATH": str(custom)}):
            self.assertEqual(tzdata_mod._default_tzdata_path(), custom)

    def test_default_path_falls_back_to_home(self):
        env_without = {
            k: v for k, v in tzdata_mod.os.environ.items()
            if k != "PYARROW_TZDATA_PATH"
        }
        with patch.dict(tzdata_mod.os.environ, env_without, clear=True):
            self.assertEqual(
                tzdata_mod._default_tzdata_path(),
                Path.home() / "Downloads" / "tzdata",
            )

    def test_manual_download_retries_on_transient_error(self):
        """Internal urlretrieve should retry transient failures."""
        out = self.tmp_path / "out.bin"
        opener_calls = {"n": 0}

        class _Resp:
            def read(self):
                return b"x" * 4096
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout):
            opener_calls["n"] += 1
            if opener_calls["n"] < 3:
                raise IOError("flaky")
            return _Resp()

        with patch("urllib.request.urlopen", fake_urlopen), \
             patch.object(tzdata_mod.time, "sleep"):
            tzdata_mod._urlretrieve("http://example/", out, timeout=1, attempts=3)
        self.assertEqual(opener_calls["n"], 3)
        self.assertTrue(out.exists())
