"""Tests for yggdrasil.cli.style — the CLI-styled logging handler / formatter."""

from __future__ import annotations

import logging

from yggdrasil.cli import style


def _plain_formatter() -> style.LogFormatter:
    # Force color off so assertions match plain text regardless of TTY / env.
    style.force_color(False)
    return style.LogFormatter()


class TestLogo:
    def setup_method(self):
        style.force_color(False)

    def test_known_services_have_full_combined_logos(self):
        for key in ("YGG", "YGGLOKI", "YGGDBKS", "YGGAWS"):
            lines = style.logo(key).splitlines()
            assert len(lines) == 4, key                 # full wordmark, 4 rows
            assert all("|" in ln or "_" in ln for ln in lines), key

    def test_no_bare_subtitle_appended(self):
        # The old behavior rendered the YGG art + a plain "YGGLOKI" text line.
        art = style.logo("YGGLOKI")
        assert "YGGLOKI" not in art          # rendered as art, not as a subtitle
        assert len(art.splitlines()) == 4    # no extra subtitle row

    def test_unknown_suffix_falls_back_to_plain_ygg_without_subtitle(self):
        art = style.logo("YGGNOPE")
        assert art == style.logo("YGG")      # plain mark, no "YGGNOPE" subtitle


class TestLogFormatter:
    def test_renders_clock_glyph_name_message(self):
        fmt = _plain_formatter()
        rec = logging.LogRecord(
            "yggdrasil.demo", logging.INFO, __file__, 1,
            "uploaded (size=%s)", ("1.5 KiB",), None,
        )
        out = fmt.format(rec)
        # Glyph for INFO, the logger name, and the %-expanded message.
        assert "●" in out
        assert "yggdrasil.demo" in out
        assert "uploaded (size=1.5 KiB)" in out

    def test_level_glyphs_differ(self):
        fmt = _plain_formatter()

        def glyph(level: int) -> str:
            rec = logging.LogRecord("x", level, __file__, 1, "m", (), None)
            return fmt.format(rec)

        assert "•" in glyph(logging.DEBUG)
        assert "●" in glyph(logging.INFO)
        assert "▲" in glyph(logging.WARNING)
        assert "✗" in glyph(logging.ERROR)

    def test_show_name_false_drops_logger_name(self):
        style.force_color(False)
        fmt = style.LogFormatter(show_name=False)
        rec = logging.LogRecord("yggdrasil.demo", logging.INFO, __file__, 1, "hi", (), None)
        assert "yggdrasil.demo" not in fmt.format(rec)

    def test_exception_traceback_appended(self):
        fmt = _plain_formatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
        out = fmt.format(rec)
        assert "Traceback" in out and "ValueError: boom" in out


class TestInstallLogging:
    def setup_method(self):
        self._root = logging.getLogger()
        self._saved = list(self._root.handlers)
        self._saved_level = self._root.level
        self._root.handlers = []

    def teardown_method(self):
        self._root.handlers = self._saved
        self._root.setLevel(self._saved_level)

    def test_installs_single_styled_handler(self):
        h = style.install_logging(logging.DEBUG)
        styled = [x for x in self._root.handlers if getattr(x, "_ygg_styled", False)]
        assert styled == [h]
        assert isinstance(h.formatter, style.LogFormatter)
        assert self._root.level == logging.DEBUG

    def test_idempotent_reuses_handler(self):
        h1 = style.install_logging(logging.INFO)
        h2 = style.install_logging(logging.DEBUG)
        assert h1 is h2
        styled = [x for x in self._root.handlers if getattr(x, "_ygg_styled", False)]
        assert len(styled) == 1
        assert h2.level == logging.DEBUG     # level refreshed on reuse

    def test_force_removes_other_handlers(self):
        other = logging.StreamHandler()
        self._root.addHandler(other)
        style.install_logging(logging.INFO, force=True)
        assert other not in self._root.handlers
        styled = [x for x in self._root.handlers if getattr(x, "_ygg_styled", False)]
        assert len(styled) == 1


class TestImportLeavesLoggingUnconfigured:
    """``import yggdrasil`` must not configure logging at all — no handler on the
    root logger, no swapped-out ``lastResort`` — so the normal stdlib idioms
    (``logging.basicConfig``, ``setLevel``) fully decide whether ygg's INFO logs
    show. Each case runs in a fresh subprocess so import executes against a
    pristine logging state.
    """

    def _run(self, body: str):
        import os
        import subprocess
        import sys

        return subprocess.run(
            [sys.executable, "-c", body],
            capture_output=True, text=True, env=dict(os.environ),
        )

    def test_import_adds_no_root_handler(self):
        out = self._run(
            "import logging, yggdrasil;"
            "print('ROOT_HANDLERS', len(logging.getLogger().handlers))"
        )
        assert "ROOT_HANDLERS 0" in out.stdout, (out.stdout, out.stderr)

    def test_import_leaves_last_resort_default(self):
        # The styled-default-on-import was removed: lastResort stays the stdlib's
        # own handler, never a ygg-styled one.
        out = self._run(
            "import logging, yggdrasil;"
            "print('STYLED', getattr(logging.lastResort, '_ygg_styled', False))"
        )
        assert "STYLED False" in out.stdout, (out.stdout, out.stderr)

    def test_basicconfig_info_surfaces_ygg_info_once(self):
        out = self._run(
            "import logging, yggdrasil;"
            "logging.basicConfig(level=logging.INFO);"
            "logging.getLogger('yggdrasil.demo').info('YGG_INFO_MARKER')"
        )
        # basicConfig works normally (ygg didn't pre-configure root) ...
        assert "YGG_INFO_MARKER" in out.stderr, (out.stdout, out.stderr)
        # ... and the line shows exactly once.
        assert out.stderr.count("YGG_INFO_MARKER") == 1, out.stderr

    def test_default_quiet_at_info(self):
        out = self._run(
            "import logging, yggdrasil;"
            "logging.getLogger('yggdrasil.demo').info('HIDDEN_INFO');"
            "logging.getLogger('yggdrasil.demo').warning('SHOWN_WARNING')"
        )
        assert "HIDDEN_INFO" not in out.stderr, out.stderr
        assert "SHOWN_WARNING" in out.stderr, out.stderr
