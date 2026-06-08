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


class TestSpinnerProgress:
    """The spinner's inline step-budget bar (used by the Loki act monitor)."""

    def setup_method(self):
        style.force_color(False)

    def test_no_bar_until_progress_set(self):
        assert style.Spinner("thinking")._bar() == ""

    def test_bar_fills_proportionally(self):
        sp = style.Spinner("thinking")
        sp.set_progress(3, 12)
        bar = style.strip(sp._bar())
        assert bar.count("█") == 3            # 3/12 of a 12-wide bar
        assert bar.count("░") == 9
        assert "3/12" in bar

    def test_bar_never_overflows_when_complete_or_past(self):
        sp = style.Spinner("x")
        sp.set_progress(20, 12)               # current past total
        bar = style.strip(sp._bar())
        assert bar.count("█") == 12 and bar.count("░") == 0
        assert "12/12" in bar                 # current clamped in the count, too

    def test_zero_total_is_safe(self):
        sp = style.Spinner("x")
        sp.set_progress(0, 0)
        assert sp._bar() == ""


class TestStripAndTitle:
    def test_strip_removes_ansi_color(self):
        assert style.strip(style.red("hi") + " " + style.dim("there")) == "hi there"
        assert style.strip("\033[1;36mbold cyan\033[0m") == "bold cyan"
        assert style.strip("plain") == "plain"

    def test_set_title_emits_osc_on_tty(self, monkeypatch, capsys):
        # OSC 2 sets the terminal title; only meaningful on a TTY.
        monkeypatch.setattr(style, "_IS_TTY", True)
        style.set_title(style.green("10 tok") + "\n$0.00")
        out = capsys.readouterr().out
        assert out.startswith("\033]2;")
        assert out.endswith("\a")
        # ANSI stripped and newlines flattened so the title chrome stays clean.
        assert "\033[" not in out[2:]
        assert "\n" not in out

    def test_set_title_noop_off_tty(self, monkeypatch, capsys):
        monkeypatch.setattr(style, "_IS_TTY", False)
        style.set_title("anything")
        assert capsys.readouterr().out == ""


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


class TestTrackAwaitable:
    """style.track drives an Awaitable behind a spinner/progress bar."""

    def setup_method(self):
        style.force_color(False)

    def _task(self, polls=3, fail=False):
        from yggdrasil.dataclasses.awaitable import Awaitable
        from yggdrasil.enums.state import State

        class _T(Awaitable):
            def __init__(s): s._n = 0
            def _start(s): s._state = State.RUNNING
            def _poll(s):
                s._n += 1
                if s._n >= polls:
                    s._state = State.FAILED if fail else State.SUCCEEDED
            def _error_for_status(s): return RuntimeError("boom")
            def progress(s): return min(s._n / polls, 1.0)
        return _T()

    def test_track_drives_to_done(self, monkeypatch):
        monkeypatch.setattr(style, "_IS_TTY", False)
        task = self._task(polls=2)
        out = style.track(task, "working…", interval=0)
        assert out is task and task.is_done and task.is_succeeded

    def test_track_surfaces_failure(self, monkeypatch):
        import pytest
        monkeypatch.setattr(style, "_IS_TTY", False)
        with pytest.raises(RuntimeError, match="boom"):
            style.track(self._task(polls=1, fail=True), interval=0)


class TestProgressBar:
    def setup_method(self):
        style.force_color(False)

    def test_determinate_fills_and_shows_pct(self, monkeypatch, capsys):
        monkeypatch.setattr(style, "_IS_TTY", True)
        style.ProgressBar(total=10, label="idx").update(4)
        out = style.strip(capsys.readouterr().out)
        assert out.count("█") == int(24 * 0.4)       # 24-wide bar, 40% filled
        assert "40%" in out and "idx" in out

    def test_frac_directly(self, monkeypatch, capsys):
        monkeypatch.setattr(style, "_IS_TTY", True)
        style.ProgressBar(label="x").update(frac=1.0)
        assert "100%" in style.strip(capsys.readouterr().out)

    def test_indeterminate_has_no_pct(self, monkeypatch, capsys):
        monkeypatch.setattr(style, "_IS_TTY", True)
        style.ProgressBar(label="waiting").update()   # no total/frac
        out = style.strip(capsys.readouterr().out)
        assert "█" in out and "░" in out and "%" not in out

    def test_noop_off_tty(self, monkeypatch, capsys):
        monkeypatch.setattr(style, "_IS_TTY", False)
        style.ProgressBar(total=10).update(5)
        assert capsys.readouterr().out == ""
