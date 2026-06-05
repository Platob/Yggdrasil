"""Tests for yggdrasil.cli.style — the CLI-styled logging handler / formatter."""

from __future__ import annotations

import logging

from yggdrasil.cli import style


def _plain_formatter() -> style.LogFormatter:
    # Force color off so assertions match plain text regardless of TTY / env.
    style.force_color(False)
    return style.LogFormatter()


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
