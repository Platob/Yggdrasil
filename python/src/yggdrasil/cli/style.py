"""Shared ANSI styling, logo, and animations for all ygg CLIs.

No external dependencies — pure sys.stdout ANSI escape sequences.
Auto-disables color when stdout is not a TTY.
"""
from __future__ import annotations

import itertools
import os
import re
import sys
import threading
import time
from typing import Any, Sequence

_CSI = "\033["
_RESET = f"{_CSI}0m"
_IS_TTY = sys.stdout.isatty()


def _color_enabled() -> bool:
    """Whether to emit ANSI color.

    A real TTY gets color; so does Databricks job / notebook output (it
    renders ANSI even though stdout is not a TTY) and anything that opts in
    via ``FORCE_COLOR`` / ``YGG_FORCE_COLOR``. ``NO_COLOR`` (the de-facto
    standard) always wins and turns it off."""
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR") or os.environ.get("YGG_FORCE_COLOR")
    if force and force.lower() not in ("0", "false", "no"):
        return True
    if _IS_TTY:
        return True
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


#: Color is gated separately from :data:`_IS_TTY` — escape sequences render in
#: places that aren't a terminal (Databricks output panels), while cursor
#: animations (spinner, clear-line) stay TTY-only so logs don't fill with
#: ``\r`` junk. Flip it explicitly with :func:`force_color`.
_COLOR = _color_enabled()


def force_color(enabled: bool = True) -> None:
    """Override color autodetection (``NO_COLOR`` still wins when disabling).

    Used by CLIs whose output is consumed in ANSI-rendering surfaces — a
    terminal or a Databricks job / notebook panel — so color shows even off a
    TTY."""
    global _COLOR
    _COLOR = bool(enabled) and not os.environ.get("NO_COLOR")


# -- colors ----------------------------------------------------------------

def _esc(code: str, text: str) -> str:
    if not _COLOR:
        return text
    return f"{_CSI}{code}m{text}{_RESET}"

# One coral-forward theme for every ygg CLI. Coral orange is the brand /
# primary accent; green means good, red means bad, amber means caution, and
# everything secondary recedes to a muted grey. The decorative names kept for
# back-compat (``cyan``/``magenta``/``blue``) are repainted onto this theme, so
# all CLIs share one look — retune the palette here and it changes everywhere.
_CORAL = "38;5;209"   # brand / primary accent (coral orange)
_GREEN = "38;5;42"    # good
_RED   = "38;5;203"   # bad (a coral-red that sits beside the brand)
_AMBER = "38;5;214"   # caution
_MUTED = "38;5;245"   # secondary — labels, paths, hints

def bold(text: str) -> str:  return _esc("1", text)
def dim(text: str) -> str:   return _esc("2", text)

# -- semantic palette (prefer these) --------------------------------------
def brand(text: str) -> str: return _esc(_CORAL, text)   # coral orange
def good(text: str) -> str:  return _esc(_GREEN, text)
def bad(text: str) -> str:   return _esc(_RED, text)
def amber(text: str) -> str: return _esc(_AMBER, text)
def muted(text: str) -> str: return _esc(_MUTED, text)

# -- back-compat aliases — existing call sites map onto the theme ----------
coral = orange = brand           # brand / primary accent
green = good                     # good
red = bad                        # bad
yellow = amber                   # caution
cyan = magenta = brand           # decorative accents → brand coral
blue = muted                     # paths / secondary → muted grey

def clear_line() -> None:
    sys.stdout.write(f"{_CSI}2K\r")
    sys.stdout.flush()

def out(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()

# -- username color --------------------------------------------------------

_USER_CODES = ("31", "32", "33", "34", "35", "36")

def colored_name(name: str) -> str:
    return _esc(_USER_CODES[hash(name) % len(_USER_CODES)], name)

# -- logo ------------------------------------------------------------------

_LOGOS: dict[str, tuple[str, ...]] = {
    "YGG": (
        r" __   __ ___ ___ ",
        r" \ \ / // __/ __|",
        r"  \ V /| (_ | (_ |",
        r"   |_|  \___|\___| ",
    ),
    "YGGNODE": (
        r" __   __ ___ ___  _  _  ___  ___  ___",
        r" \ \ / // __/ __|| \| |/ _ \|   \| __|",
        "  \\ V /| (_ | (_ | .` | (_) | |) | _| ",
        r"   |_|  \___|\___||_|\_|\___/|___/|___|",
    ),
    "YGGCHAT": (
        r" __   __ ___ ___   ___ _  _    _  _____",
        r" \ \ / // __/ __| / __| || |  / \|_   _|",
        r"  \ V /| (_ | (_ | (__| __ | / _ \ | |  ",
        r"   |_|  \___|\___| \___|_||_|/_/ \_\|_|  ",
    ),
    "YGGDBKS": (
        " __   __ ___ ___  ___  ___ _  __ ___",
        r" \ \ / // __/ __||   \| _ ) |/ // __|",
        "  \\ V /| (_ | (_ | |) | _ \\ ' < \\__ \\",
        r"   |_|  \___|\___||___/|___/_|\_\|___/",
    ),
    "GENIE": (
        r"  ___ ___ _  _ ___ ___ ",
        r" / __| __| \| |_ _| __|",
        r"| (_ | _|| .` || || _| ",
        r" \___|___|_|\_|___|___|",
    ),
}


#: Coral brand gradient (256-color) painted top→bottom across the logo rows —
#: peach → coral → orange, anchored on the brand coral.
_LOGO_GRADIENT = ("38;5;216", "38;5;209", "38;5;208", "38;5;202")


def logo(suffix: str = "") -> str:
    """Render the YGG logo with an optional suffix like BOT, CHAT, GENIE."""
    key = suffix or "YGG"
    lines = _LOGOS.get(key, _LOGOS["YGG"])
    if not _COLOR:
        parts = ["  " + ln for ln in lines]
        if key not in _LOGOS and suffix:
            parts.append(f"  {suffix}")
        return "\n".join(parts)

    r = _RESET
    rendered = [
        f"  {_CSI}{_LOGO_GRADIENT[min(i, len(_LOGO_GRADIENT) - 1)]}m{ln}{r}"
        for i, ln in enumerate(lines)
    ]
    if key not in _LOGOS and suffix:
        rendered.append(f"  {_CSI}1m{suffix}{r}")
    return "\n".join(rendered)


def print_logo(suffix: str = "") -> None:
    out("\n" + logo(suffix) + "\n\n")

# -- spinner ---------------------------------------------------------------

_BRAILLE = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_DOTS = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
_BARS = ("▏", "▎", "▍", "▌", "▋", "▊", "▉", "█", "▉", "▊", "▋", "▌", "▍", "▎", "▏")
_BOUNCE = ("⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈")


class Spinner:
    """Animated spinner that runs in a background thread."""

    def __init__(
        self,
        text: str = "",
        frames: tuple[str, ...] = _BRAILLE,
        interval: float = 0.08,
        color: str = "36",
    ) -> None:
        self.text = text
        self.frames = frames
        self.interval = interval
        self.color = color
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "Spinner":
        if not _IS_TTY:
            if self.text:
                out(f"  {self.text}\n")
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        cycle = itertools.cycle(self.frames)
        while not self._stop_event.is_set():
            frame = next(cycle)
            colored = f"{_CSI}{self.color}m{frame}{_RESET}"
            sys.stdout.write(f"\r{_CSI}2K  {colored} {self.text}")
            sys.stdout.flush()
            self._stop_event.wait(self.interval)

    def update(self, text: str) -> None:
        self.text = text

    def stop(self, final: str = "") -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if _IS_TTY:
            clear_line()
        if final:
            out(f"  {final}\n")

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# -- progress bar ----------------------------------------------------------

def progress_bar(current: int, total: int, width: int = 30, label: str = "") -> str:
    if total <= 0:
        return ""
    frac = min(current / total, 1.0)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{frac * 100:5.1f}%"
    if _COLOR:
        return f"  {_CSI}36m{bar}{_RESET} {pct} {dim(label)}"
    return f"  [{bar}] {pct} {label}"


# -- typing animation ------------------------------------------------------

def typing_dots(duration: float = 0.35, frames: int = 3) -> None:
    if not _IS_TTY:
        return
    delay = duration / (frames + 1)
    for i in range(1, frames + 1):
        out(f"\r  {dim('.' * i)}")
        time.sleep(delay)
    clear_line()


# -- pulse effect ----------------------------------------------------------

def pulse_text(text: str, duration: float = 1.5, cycles: int = 2) -> None:
    """Fade text between dim and bright."""
    if not _IS_TTY:
        out(f"  {text}\n")
        return
    steps = 8
    delay = duration / (cycles * steps * 2)
    for _ in range(cycles):
        for code in ("2", "0", "1", "0"):
            out(f"\r{_CSI}2K  {_CSI}{code}m{text}{_RESET}")
            time.sleep(delay)
    out(f"\r{_CSI}2K  {text}\n")


# -- structured log lines --------------------------------------------------

def event(icon: str, text: str, code: str = "36") -> str:
    """A timestamped, glyph-led log line: ``  HH:MM:SS  ● text`` (the glyph
    colored, the clock dimmed)."""
    return f"  {dim(time.strftime('%H:%M:%S'))}  {_esc(code, icon)}  {text}"


def hr(width: int = 46) -> str:
    """A dim horizontal rule."""
    return "  " + dim("─" * width)


def info(text: str) -> None: out(event("●", text, _CORAL) + "\n")
def step(text: str) -> None: out(event("▸", text, _CORAL) + "\n")
def ok(text: str) -> None:   out(event("✓", text, _GREEN) + "\n")
def warn(text: str) -> None: out(event("▲", text, _AMBER) + "\n")
def fail(text: str) -> None: out(event("✗", text, _RED) + "\n")


# -- boxes: panels and tables ----------------------------------------------

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_len(text: str) -> int:
    """Length of ``text`` ignoring ANSI SGR escapes."""
    return len(_ANSI_RE.sub("", text))


def _pad(text: str, width: int) -> str:
    """Right-pad ``text`` to ``width`` visible columns (ANSI-aware)."""
    return text + " " * max(0, width - _visible_len(text))


def panel(body: str, *, title: str = "", width: int = 0, color: str = _CORAL) -> str:
    """A rounded box around ``body`` with an optional colored ``title``.

    ``body`` may contain newlines and inline ANSI; the border is dim and
    the title wears ``color``.
    """
    lines = body.split("\n")
    inner = max([_visible_len(ln) for ln in lines] + [_visible_len(title) + 2, width - 4])
    top = f"╭─ {_esc(color, title)} " + "─" * (inner - _visible_len(title) - 2) + "╮" if title \
        else "╭" + "─" * (inner + 2) + "╮"
    rows = [f"{dim('│')} {_pad(ln, inner)} {dim('│')}" for ln in lines]
    bottom = "╰" + "─" * (inner + 2) + "╯"
    return "\n".join(["  " + dim(top)] + ["  " + r for r in rows] + ["  " + dim(bottom)])


def table(headers: "Sequence[str]", rows: "Sequence[Sequence[Any]]", *, max_rows: int = 0) -> str:
    """A bordered table — bold/brand headers, dim borders, padded cells.

    Cells are stringified; values are not ANSI-styled so the column widths
    stay exact. Pass ``max_rows`` to cap the body (a footer notes the rest).
    """
    cols = [str(h) for h in headers]
    body = [[str(c) for c in r] for r in rows]
    shown = body[:max_rows] if max_rows and len(body) > max_rows else body
    widths = [
        max([_visible_len(cols[i])] + [_visible_len(r[i]) for r in shown if i < len(r)])
        for i in range(len(cols))
    ]

    def _row(cells: "Sequence[str]", *, header: bool = False) -> str:
        out_cells = []
        for i, w in enumerate(widths):
            cell = cells[i] if i < len(cells) else ""
            out_cells.append(bold(brand(_pad(cell, w))) if header else _pad(cell, w))
        return f"{dim('│')} " + f" {dim('│')} ".join(out_cells) + f" {dim('│')}"

    bar = lambda l, m, r: dim(l + m.join("─" * (w + 2) for w in widths) + r)  # noqa: E731
    lines = [
        "  " + bar("╭", "┬", "╮"),
        "  " + _row(cols, header=True),
        "  " + bar("├", "┼", "┤"),
    ]
    lines += ["  " + _row(r) for r in shown]
    lines.append("  " + bar("╰", "┴", "╯"))
    if max_rows and len(body) > max_rows:
        lines.append("  " + muted(f"… {len(body) - max_rows} more rows"))
    return "\n".join(lines)
