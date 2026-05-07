"""Environment-aware progress bar.

Renders correctly in three places:

- **Plain terminal / log file**: writes a single carriage-returned
  line to ``sys.stderr`` (or whatever stream the caller passes).
- **IPython / Jupyter**: replaces an HTML ``<progress>`` element in
  place via ``IPython.display.update_display``.
- **Databricks notebook**: same HTML widget, displayed through
  ``displayHTML`` when available (the Databricks notebook stripped
  cell stdout helpers, but it keeps DOM updates cheap).

The bar is also a context manager and an iterable wrapper, so the
common shapes all work::

    with ProgressBar(total=len(rows), desc="ingest") as bar:
        for row in rows:
            ingest(row)
            bar.update()

    for row in ProgressBar(rows, desc="ingest"):
        ingest(row)

When ``total`` is unknown (non-sized iterable, no ``total=`` passed),
the bar falls back to a spinner-style counter so it stays useful for
streams.
"""

from __future__ import annotations

import os
import sys
import time
from html import escape
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

_BAR_WIDTH = 40
_MIN_REDRAW_INTERVAL = 0.1  # seconds; cheap throttle so tight loops don't spam

_DBR_RUNTIME_ENV = "DATABRICKS_RUNTIME_VERSION"


def _detect_renderer(stream) -> "_Renderer":
    """Pick the best renderer for the current environment.

    Order matters: Databricks runs inside IPython, so we have to check
    for the Databricks runtime first before falling through to plain
    IPython.
    """
    if _in_databricks():
        dbx = _DatabricksRenderer.try_build()
        if dbx is not None:
            return dbx

    ipy = _IPythonRenderer.try_build()
    if ipy is not None:
        return ipy

    return _TextRenderer(stream if stream is not None else sys.stderr)


def _in_databricks() -> bool:
    return bool(os.environ.get(_DBR_RUNTIME_ENV))


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN guard for first-tick edge
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"


class ProgressBar:
    """Display-aware progress bar.

    Args:
        iterable: Optional iterable to wrap. When passed, iterating
            over the bar yields its items and ticks once per item.
        total: Total number of expected steps. Inferred from
            ``len(iterable)`` when possible. ``None`` means "unknown" —
            the bar shows a counter instead of a percent bar.
        desc: Short label shown next to the bar.
        unit: Unit suffix on the count (``it`` by default, e.g.
            ``rows``, ``files``, ``MB``).
        stream: Where the text renderer writes. Defaults to
            ``sys.stderr`` so it doesn't pollute piped stdout.
        disable: Skip rendering entirely. Handy for non-interactive
            jobs where you still want the same iteration shape.
        min_interval: Minimum seconds between visual updates. Updates
            arriving faster are coalesced. The final tick always
            renders.
    """

    __slots__ = (
        "iterable",
        "total",
        "desc",
        "unit",
        "disable",
        "min_interval",
        "_n",
        "_start",
        "_last_render",
        "_renderer",
        "_finished",
    )

    def __init__(
        self,
        iterable: Optional[Iterable[T]] = None,
        *,
        total: Optional[int] = None,
        desc: str = "",
        unit: str = "it",
        stream=None,
        disable: bool = False,
        min_interval: float = _MIN_REDRAW_INTERVAL,
    ) -> None:
        if total is None and iterable is not None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None

        if total is not None and total < 0:
            raise ValueError(
                f"ProgressBar total must be non-negative, got {total}. "
                "Pass total=None for streams of unknown length."
            )

        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.min_interval = max(0.0, float(min_interval))

        self._n = 0
        self._start = time.monotonic()
        self._last_render = 0.0
        self._finished = False
        self._renderer: _Renderer = _NullRenderer() if disable else _detect_renderer(stream)

    # ---- iteration ----------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        if self.iterable is None:
            raise TypeError(
                "ProgressBar has no iterable to iterate over. "
                "Either pass an iterable to ProgressBar(...) or call "
                "update() manually inside your own loop."
            )
        return self._iter(self.iterable)

    def _iter(self, iterable: Iterable[T]) -> Iterator[T]:
        try:
            self._render(force=True)
            for item in iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    # ---- context manager ---------------------------------------------

    def __enter__(self) -> "ProgressBar":
        self._render(force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- public API ---------------------------------------------------

    @property
    def n(self) -> int:
        """Steps completed so far."""
        return self._n

    def update(self, step: int = 1) -> None:
        """Advance the bar by ``step`` and redraw if it's time.

        ``step`` may be negative (rare, but useful when retrying
        causes work to roll back). The bar clamps at zero so we don't
        flash a negative count.
        """
        if self.disable:
            return
        self._n = max(0, self._n + step)
        if self.total is not None and self._n > self.total:
            # Going past total is not an error — caller might re-enter
            # work — but we do need to widen the displayed total so the
            # rendered fraction stays sane.
            self.total = self._n
        self._render()

    def set_description(self, desc: str) -> None:
        """Update the label and redraw."""
        self.desc = desc
        self._render(force=True)

    def reset(self, total: Optional[int] = None) -> None:
        """Reset the counter. Optionally swap the total for a new pass."""
        self._n = 0
        if total is not None:
            self.total = total
        self._start = time.monotonic()
        self._finished = False
        self._render(force=True)

    def close(self) -> None:
        """Render the final state and release the renderer."""
        if self._finished:
            return
        self._finished = True
        self._render(force=True)
        self._renderer.finalize(self._payload())

    # ---- rendering ----------------------------------------------------

    def _render(self, *, force: bool = False) -> None:
        if self.disable:
            return
        now = time.monotonic()
        if not force and (now - self._last_render) < self.min_interval:
            # Always let the terminal renderer flush the final tick
            # via close(); the throttle only skips intermediate frames.
            if self.total is None or self._n != self.total:
                return
        self._last_render = now
        self._renderer.render(self._payload())

    def _payload(self) -> "_Payload":
        elapsed = max(0.0, time.monotonic() - self._start)
        rate = self._n / elapsed if elapsed > 0 else 0.0
        eta: Optional[float]
        if self.total is None or rate <= 0:
            eta = None
        else:
            eta = max(0.0, (self.total - self._n) / rate)
        return _Payload(
            n=self._n,
            total=self.total,
            desc=self.desc,
            unit=self.unit,
            elapsed=elapsed,
            rate=rate,
            eta=eta,
            finished=self._finished,
        )


# ---------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------


class _Payload:
    """Snapshot of bar state passed to renderers — keeps them stateless."""

    __slots__ = ("n", "total", "desc", "unit", "elapsed", "rate", "eta", "finished")

    def __init__(
        self,
        *,
        n: int,
        total: Optional[int],
        desc: str,
        unit: str,
        elapsed: float,
        rate: float,
        eta: Optional[float],
        finished: bool,
    ) -> None:
        self.n = n
        self.total = total
        self.desc = desc
        self.unit = unit
        self.elapsed = elapsed
        self.rate = rate
        self.eta = eta
        self.finished = finished

    def text_line(self) -> str:
        prefix = f"{self.desc}: " if self.desc else ""
        rate_part = f"{self.rate:.1f} {self.unit}/s" if self.rate > 0 else f"? {self.unit}/s"
        elapsed_part = _format_eta(self.elapsed)
        if self.total is not None and self.total > 0:
            frac = self.n / self.total
            filled = int(_BAR_WIDTH * frac)
            bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
            eta_part = _format_eta(self.eta) if self.eta is not None else "?"
            return (
                f"{prefix}[{bar}] {self.n}/{self.total} "
                f"({frac:6.1%}) [{elapsed_part}<{eta_part}, {rate_part}]"
            )
        return f"{prefix}{self.n} {self.unit} [{elapsed_part}, {rate_part}]"

    def html(self) -> str:
        desc = escape(self.desc) if self.desc else ""
        unit = escape(self.unit)
        elapsed_part = _format_eta(self.elapsed)
        if self.total is not None and self.total > 0:
            frac = self.n / self.total
            eta_part = _format_eta(self.eta) if self.eta is not None else "?"
            rate_part = f"{self.rate:.1f} {unit}/s" if self.rate > 0 else f"? {unit}/s"
            label = (
                f"{self.n}/{self.total} ({frac:6.1%}) "
                f"[{elapsed_part}&lt;{eta_part}, {rate_part}]"
            )
            bar = (
                f'<progress value="{self.n}" max="{self.total}" '
                f'style="width: 320px; height: 14px;"></progress>'
            )
        else:
            rate_part = f"{self.rate:.1f} {unit}/s" if self.rate > 0 else f"? {unit}/s"
            label = f"{self.n} {unit} [{elapsed_part}, {rate_part}]"
            # Indeterminate <progress> draws a moving stripe in most browsers.
            bar = '<progress style="width: 320px; height: 14px;"></progress>'

        head = f'<span style="font-family: monospace; margin-right: 8px;">{desc}</span>' if desc else ""
        return (
            '<div style="font-family: monospace; font-size: 12px; '
            'display: flex; align-items: center; gap: 8px;">'
            f"{head}{bar}"
            f'<span style="font-family: monospace;">{label}</span>'
            "</div>"
        )


class _Renderer:
    def render(self, payload: _Payload) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def finalize(self, payload: _Payload) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _NullRenderer(_Renderer):
    def render(self, payload: _Payload) -> None:
        return

    def finalize(self, payload: _Payload) -> None:
        return


class _TextRenderer(_Renderer):
    """Single-line, carriage-returned text renderer for TTYs and logs."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self._is_tty = bool(getattr(stream, "isatty", lambda: False)())

    def render(self, payload: _Payload) -> None:
        line = payload.text_line()
        if self._is_tty:
            self.stream.write("\r" + line)
        else:
            # Non-TTY (log file, piped stdout): newline per update so
            # lines don't get glued together.
            self.stream.write(line + "\n")
        try:
            self.stream.flush()
        except Exception:
            pass

    def finalize(self, payload: _Payload) -> None:
        # End the carriage-returned line cleanly so the next print
        # doesn't land mid-bar.
        if self._is_tty:
            self.stream.write("\r" + payload.text_line() + "\n")
            try:
                self.stream.flush()
            except Exception:
                pass


class _IPythonRenderer(_Renderer):
    """Updates a single HTML element in place via IPython display IDs."""

    def __init__(self, display_fn, html_cls, display_id: str) -> None:
        self._display = display_fn
        self._html_cls = html_cls
        self._display_id = display_id
        self._published = False

    @classmethod
    def try_build(cls) -> "Optional[_IPythonRenderer]":
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
        except Exception:
            return None
        if get_ipython() is None:
            return None
        # Stable display id per renderer instance — multiple bars can
        # coexist as long as each has its own id.
        display_id = f"yggdrasil-progress-{time.monotonic_ns()}"
        return cls(display, HTML, display_id)

    def render(self, payload: _Payload) -> None:
        html = self._html_cls(payload.html())
        if not self._published:
            self._display(html, display_id=self._display_id)
            self._published = True
        else:
            self._display(html, display_id=self._display_id, update=True)

    def finalize(self, payload: _Payload) -> None:
        # One last paint so the bar shows the final count even if the
        # last update was throttled.
        self.render(payload)


class _DatabricksRenderer(_Renderer):
    """Databricks-friendly renderer.

    Databricks notebooks expose ``displayHTML`` but it appends a new
    output cell on every call instead of updating in place. Where
    possible we still prefer the IPython display protocol (which DBR
    routes through to the same widget surface) and only fall back to
    ``displayHTML`` when IPython display IDs aren't usable. In that
    fallback we emit one widget at the start and then quietly print
    text updates to stderr so the user still gets feedback without
    flooding the notebook with cells.
    """

    def __init__(self, ipy: Optional[_IPythonRenderer], display_html_fn) -> None:
        self._ipy = ipy
        self._display_html = display_html_fn
        self._first_emit = True

    @classmethod
    def try_build(cls) -> "Optional[_DatabricksRenderer]":
        ipy = _IPythonRenderer.try_build()
        display_html = _resolve_display_html()
        if ipy is None and display_html is None:
            return None
        return cls(ipy, display_html)

    def render(self, payload: _Payload) -> None:
        if self._ipy is not None:
            self._ipy.render(payload)
            return
        # Fallback path: paint one HTML widget on the first call and
        # then reuse stderr for subsequent ticks. Spamming displayHTML
        # creates a wall of cells that nobody wants to scroll through.
        if self._first_emit and self._display_html is not None:
            try:
                self._display_html(payload.html())
            except Exception:
                pass
            self._first_emit = False
        sys.stderr.write("\r" + payload.text_line())
        try:
            sys.stderr.flush()
        except Exception:
            pass

    def finalize(self, payload: _Payload) -> None:
        if self._ipy is not None:
            self._ipy.finalize(payload)
            return
        sys.stderr.write("\r" + payload.text_line() + "\n")
        try:
            sys.stderr.flush()
        except Exception:
            pass


def _resolve_display_html():
    """Find ``displayHTML`` from the Databricks runtime, if present.

    DBR injects ``displayHTML`` either as a builtin or into the
    IPython user namespace. We check both before giving up.
    """
    import builtins

    fn = getattr(builtins, "displayHTML", None)
    if callable(fn):
        return fn

    try:
        from IPython import get_ipython
    except Exception:
        return None
    ip = get_ipython()
    if ip is None:
        return None
    fn = getattr(ip, "user_ns", {}).get("displayHTML")
    if callable(fn):
        return fn
    return None
