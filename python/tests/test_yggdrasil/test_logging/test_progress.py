"""Tests for ``yggdrasil.logging.progress``.

Covers the three rendering paths (text TTY, text non-TTY, IPython /
Databricks HTML) by injecting fakes — no real terminal or notebook
required.
"""

from __future__ import annotations

import io
import logging
import sys
from unittest import mock

import pytest

from yggdrasil.logging import ProgressBar
from yggdrasil.logging import logger as _logger_mod
from yggdrasil.logging import progress as _progress_mod


class _FakeStream(io.StringIO):
    """StringIO that lets us pretend we're (not) a TTY."""

    def __init__(self, is_tty: bool = False) -> None:
        super().__init__()
        self._is_tty = is_tty

    def isatty(self) -> bool:  # type: ignore[override]
        return self._is_tty


def test_iterates_and_yields_items_in_order():
    items = list(range(5))
    seen = list(ProgressBar(items, stream=_FakeStream(), min_interval=0))
    assert seen == items


def test_total_inferred_from_len_when_iterable_is_sized():
    bar = ProgressBar([1, 2, 3], stream=_FakeStream(), min_interval=0)
    assert bar.total == 3


def test_total_none_for_unsized_iterable():
    def gen():
        yield from (1, 2, 3)

    bar = ProgressBar(gen(), stream=_FakeStream(), min_interval=0)
    assert bar.total is None


def test_negative_total_rejected_with_helpful_error():
    with pytest.raises(ValueError, match="non-negative"):
        ProgressBar(total=-1)


def test_text_renderer_overwrites_on_tty():
    stream = _FakeStream(is_tty=True)
    with ProgressBar(total=2, stream=stream, min_interval=0) as bar:
        bar.update()
        bar.update()
    out = stream.getvalue()
    # Carriage returns mean we're overwriting in place.
    assert out.count("\r") >= 2
    # And we end on a newline so the next print stays clean.
    assert out.endswith("\n")
    assert "2/2" in out


def test_text_renderer_uses_newlines_when_not_tty():
    stream = _FakeStream(is_tty=False)
    with ProgressBar(total=2, stream=stream, min_interval=0) as bar:
        bar.update()
        bar.update()
    out = stream.getvalue()
    # Non-TTY mode writes one line per update, no carriage returns.
    assert "\r" not in out
    lines = [line for line in out.splitlines() if line]
    # Initial render + 2 ticks (the close re-renders the final state
    # but the throttle gate at total==n already let it through).
    assert len(lines) >= 3
    assert "2/2" in lines[-1]


def test_disabled_bar_emits_nothing_but_still_iterates():
    stream = _FakeStream(is_tty=True)
    out = list(ProgressBar(range(4), stream=stream, disable=True))
    assert out == [0, 1, 2, 3]
    assert stream.getvalue() == ""


def test_manual_update_without_iterable_raises_on_iter():
    bar = ProgressBar(total=3, stream=_FakeStream(), min_interval=0)
    with pytest.raises(TypeError, match="no iterable"):
        iter(bar)


def test_update_past_total_widens_total_instead_of_breaking():
    bar = ProgressBar(total=2, stream=_FakeStream(is_tty=True), min_interval=0)
    bar.update(5)
    assert bar.n == 5
    assert bar.total == 5  # widened, not stuck at 2
    bar.close()


def test_update_negative_step_clamps_at_zero():
    bar = ProgressBar(total=10, stream=_FakeStream(is_tty=True), min_interval=0)
    bar.update(-3)
    assert bar.n == 0
    bar.close()


def test_min_interval_throttles_intermediate_renders():
    stream = _FakeStream(is_tty=True)
    bar = ProgressBar(total=100, stream=stream, min_interval=1.0)
    # Many quick updates — throttle should keep most of them silent.
    for _ in range(50):
        bar.update()
    bar.close()
    # We see at least the opening render and the final render, but
    # nowhere near 50.
    out = stream.getvalue()
    assert out.count("\r") < 10
    assert "50/100" in out


def test_set_description_updates_label():
    stream = _FakeStream(is_tty=True)
    bar = ProgressBar(total=1, stream=stream, min_interval=0, desc="ingest")
    bar.set_description("publish")
    bar.update()
    bar.close()
    assert "publish" in stream.getvalue()


def test_reset_starts_a_fresh_pass():
    bar = ProgressBar(total=5, stream=_FakeStream(is_tty=True), min_interval=0)
    bar.update(3)
    bar.reset(total=10)
    assert bar.n == 0
    assert bar.total == 10
    bar.close()


def test_format_eta_handles_seconds_minutes_hours():
    fmt = _progress_mod._format_eta
    assert fmt(0) == "0s"
    assert fmt(45) == "45s"
    assert fmt(125) == "2m05s"
    assert fmt(3725) == "1h02m"
    assert fmt(-1) == "?"


# ---- HTML / IPython renderer paths ----------------------------------


class _FakeHTML:
    def __init__(self, html: str) -> None:
        self.html = html


class _FakeIPython:
    """Minimal stand-in for ``IPython.get_ipython()``."""

    def __init__(self, user_ns=None) -> None:
        self.user_ns = user_ns or {}


def _patch_ipython(monkeypatch, user_ns=None, display_calls=None):
    """Install a fake IPython into ``progress``'s import path.

    Returns the ``display_calls`` list so the test can inspect what
    was rendered.
    """
    if display_calls is None:
        display_calls = []

    def fake_display(html, display_id=None, update=False):
        display_calls.append({"html": html, "display_id": display_id, "update": update})

    fake_ip = _FakeIPython(user_ns=user_ns)

    fake_module = mock.MagicMock()
    fake_module.get_ipython = lambda: fake_ip
    fake_display_module = mock.MagicMock()
    fake_display_module.HTML = _FakeHTML
    fake_display_module.display = fake_display

    monkeypatch.setitem(sys.modules, "IPython", fake_module)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display_module)
    return display_calls


def test_ipython_renderer_paints_html_progress_element(monkeypatch):
    calls = _patch_ipython(monkeypatch)
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)

    bar = ProgressBar(total=3, desc="rows", min_interval=0)
    bar.update()
    bar.update()
    bar.close()

    # We pushed at least one initial render plus subsequent updates.
    assert len(calls) >= 2
    first = calls[0]
    rest = calls[1:]
    assert first["update"] is False
    assert all(c["update"] is True for c in rest)
    assert all(c["display_id"] == first["display_id"] for c in calls)
    assert "<progress" in first["html"].html
    assert "rows" in first["html"].html


def test_databricks_renderer_uses_ipython_display_when_available(monkeypatch):
    calls = _patch_ipython(monkeypatch)
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")

    bar = ProgressBar(total=2, min_interval=0)
    bar.update()
    bar.update()
    bar.close()

    assert len(calls) >= 2
    # All updates land on the same display id — we're updating in
    # place, not appending fresh cells.
    ids = {c["display_id"] for c in calls}
    assert len(ids) == 1


def test_databricks_renderer_falls_back_to_displayhtml(monkeypatch):
    """When IPython display is unavailable but DBR exposes
    ``displayHTML``, we paint the widget once and use stderr for
    subsequent ticks instead of stacking cells."""

    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")

    # Force IPython renderer to fail to build by removing IPython.
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)

    html_calls = []

    def fake_display_html(html):
        html_calls.append(html)

    monkeypatch.setattr(_progress_mod, "_resolve_display_html", lambda: fake_display_html)

    fake_stderr = _FakeStream(is_tty=True)
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    bar = ProgressBar(total=3, min_interval=0)
    bar.update()
    bar.update()
    bar.update()
    bar.close()

    # Exactly one HTML cell — not one per tick.
    assert len(html_calls) == 1
    assert "<progress" in html_calls[0]
    # Subsequent updates went to stderr as carriage-returned text.
    text = fake_stderr.getvalue()
    assert "\r" in text
    assert "3/3" in text


# ---- Logger setup ---------------------------------------------------


def test_get_logger_namespaces_under_yggdrasil():
    log = _logger_mod.get_logger("data.cast")
    assert log.name == "yggdrasil.data.cast"
    assert _logger_mod.get_logger("yggdrasil.data.cast").name == "yggdrasil.data.cast"
    assert _logger_mod.get_logger().name == "yggdrasil"


def test_setup_logger_is_idempotent():
    # Snapshot existing handlers so we can restore after.
    log = logging.getLogger("yggdrasil")
    saved = list(log.handlers)
    try:
        for h in list(log.handlers):
            log.removeHandler(h)

        first = _logger_mod.setup_logger(level=logging.WARNING)
        second = _logger_mod.setup_logger(level=logging.DEBUG)
        assert first is second
        assert len(log.handlers) == 1
        # Second call should have updated the level on the existing handler.
        assert log.handlers[0].level == logging.DEBUG
    finally:
        for h in list(log.handlers):
            log.removeHandler(h)
        for h in saved:
            log.addHandler(h)


def test_setup_logger_force_replaces_handlers():
    log = logging.getLogger("yggdrasil")
    saved = list(log.handlers)
    try:
        for h in list(log.handlers):
            log.removeHandler(h)

        _logger_mod.setup_logger(level=logging.INFO)
        assert len(log.handlers) == 1
        first_handler = log.handlers[0]

        _logger_mod.setup_logger(level=logging.INFO, force=True)
        assert len(log.handlers) == 1
        assert log.handlers[0] is not first_handler
    finally:
        for h in list(log.handlers):
            log.removeHandler(h)
        for h in saved:
            log.addHandler(h)
