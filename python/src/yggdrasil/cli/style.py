"""Shared ANSI styling, logo, and animations for all ygg CLIs.

No external dependencies — pure sys.stdout ANSI escape sequences.
Auto-disables color when stdout is not a TTY.
"""
from __future__ import annotations

import itertools
import logging
import os
import re
import sys
import threading
import time
from typing import Callable

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

#: Escape prefix per SGR code (``"2"`` → ``"\033[2m"``), built once and reused.
#: The colour helpers run several times per rendered line, so each concatenates
#: a cached prefix + text + reset rather than re-formatting the CSI every call.
_PREFIX: dict[str, str] = {}


def _esc(code: str, text: str) -> str:
    if not _COLOR:
        return text
    prefix = _PREFIX.get(code)
    if prefix is None:
        prefix = _PREFIX[code] = f"{_CSI}{code}m"
    return prefix + text + _RESET

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

def _painter(code: str) -> Callable[[str], str]:
    """A colour helper bound to *code* with its escape prefix precomputed.

    One call that concatenates ``prefix + text + reset`` — no per-call CSI
    formatting and no delegation through :func:`_esc`. The colour gate
    (:func:`force_color` / ``NO_COLOR`` / TTY) is still read live per call."""
    prefix = _PREFIX.setdefault(code, f"{_CSI}{code}m")

    def paint(text: str) -> str:
        return prefix + text + _RESET if _COLOR else text

    return paint


bold = _painter("1")
dim = _painter("2")

# -- semantic palette (prefer these) --------------------------------------
brand = _painter(_CORAL)   # coral orange
good = _painter(_GREEN)
bad = _painter(_RED)
amber = _painter(_AMBER)
muted = _painter(_MUTED)

# -- back-compat aliases — existing call sites map onto the theme ----------
coral = orange = brand           # brand / primary accent
green = good                     # good
red = bad                        # bad
yellow = amber                   # caution
cyan = magenta = brand           # decorative accents → brand coral
blue = muted                     # paths / secondary → muted grey

#: Matches SGR / CSI escape sequences so a colored string can be rendered plain
#: (e.g. into a terminal title, which doesn't interpret color codes).
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip(text: str) -> str:
    """*text* with ANSI color escapes removed."""
    return _ANSI_RE.sub("", text)

def clear_line() -> None:
    sys.stdout.write(f"{_CSI}2K\r")
    sys.stdout.flush()

def out(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()

def set_title(text: str) -> None:
    """Set the terminal title bar (OSC 2) — a *static* status line that updates
    in place instead of scrolling the transcript.

    Used by the Loki REPL to keep live token/cost KPIs pinned in the terminal
    "head" rather than reprinting a usage line after every turn. TTY-only (the
    escape is meaningless and would be junk in a redirected log); the title is
    stripped of any ANSI so it shows clean in the window/tab chrome."""
    if not _IS_TTY:
        return
    plain = strip(text).replace("\a", " ").replace("\n", " ")
    sys.stdout.write(f"\033]2;{plain}\a")
    sys.stdout.flush()

# -- username color --------------------------------------------------------

_USER_CODES = ("31", "32", "33", "34", "35", "36")

def colored_name(name: str) -> str:
    return _esc(_USER_CODES[hash(name) % len(_USER_CODES)], name)

# -- logo ------------------------------------------------------------------

# Each logo is a rectangle: every row is the same visual width and the
# glyph columns line up across rows (composed from fixed-width per-letter
# cells), so the gradient paints cleanly and nothing drifts horizontally.
_LOGOS: dict[str, tuple[str, ...]] = {
    "YGG": (
        r"__   __ ___   ___  ",
        r"\ \ / // __| / __| ",
        r" \ V / | (_ || (_ |",
        r"  |_|   \___| \___|",
    ),
    "YGGNODE": (
        r"__   __ ___   ___   _  _  ___    ___   ___ ",
        r"\ \ / // __| / __| | \| |/ _ \  |   \ | __|",
        r" \ V / | (_ || (_ || .` || (_) || |) || _| ",
        r"  |_|   \___| \___||_|\_| \___/ |___/ |___|",
    ),
    "YGGCHAT": (
        r"__   __ ___   ___   ___  _  _   _   _____  ",
        r"\ \ / // __| / __| / __|| || | /_\   |_  _|",
        r" \ V / | (_ || (_ || (__| __ |/ _ \   | |  ",
        r"  |_|   \___| \___|\___||_||_|_/ \_\  |_|  ",
    ),
    "YGGDBKS": (
        r"__   __ ___   ___   ___   ___ _  __  ___ ",
        r"\ \ / // __| / __| |   \ | _ )| |/ // __|",
        r" \ V / | (_ || (_ || |) || _ \| ' < \__ \ ",
        r"  |_|   \___| \___||___/ |___/|_|\_\|___/",
    ),
    "YGGLOKI": (
        r"__   __ ___   ___   _      ___   _  __ ___ ",
        r"\ \ / // __| / __| | |    / _ \ | |/ /|_ _|",
        r" \ V / | (_ || (_ || |__ | (_) || ' <  | | ",
        r"  |_|   \___| \___||____| \___/ |_|\_\|___|",
    ),
    "YGGAWS": (
        r"__   __ ___   ___    _   __      __ ___  ",
        r"\ \ / // __| / __|  /_\  \ \    / // __| ",
        r" \ V / | (_ || (_ |/ _ \  \ \/\/ / \__ \ ",
        r"  |_|   \___| \___|_/ \_\  \_/\_/  |___/ ",
    ),
}


#: Coral brand gradient (256-color) painted top→bottom across the logo rows —
#: peach → coral → orange, anchored on the brand coral.
_LOGO_GRADIENT = ("38;5;216", "38;5;209", "38;5;208", "38;5;202")


def logo(suffix: str = "") -> str:
    """Render the full combined CLI logo for *suffix* (``YGGLOKI``, ``YGGDBKS``,
    ``YGGAWS``, …).

    Each CLI gets ONE full ``ygg``+service wordmark — never the bare ``YGG``
    art with a text subtitle tacked underneath. An unknown suffix falls back to
    the plain ``YGG`` mark (no subtitle)."""
    key = suffix or "YGG"
    lines = _LOGOS.get(key, _LOGOS["YGG"])
    # Pad every row to the widest so the block is a true rectangle — guards
    # the gradient / right edge against trailing-space drift in the literals.
    width = max((len(ln) for ln in lines), default=0)
    lines = [ln.ljust(width) for ln in lines]
    if not _COLOR:
        return "\n".join("  " + ln for ln in lines)

    r = _RESET
    return "\n".join(
        f"  {_CSI}{_LOGO_GRADIENT[min(i, len(_LOGO_GRADIENT) - 1)]}m{ln}{r}"
        for i, ln in enumerate(lines)
    )


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
        #: Optional ``(current, total)`` — when set, a compact filled bar renders
        #: between the spinner glyph and the text so a bounded task (an agent's
        #: step budget, a multi-file pass) shows how far along it is, live.
        self._progress: tuple[int, int] | None = None
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
            sys.stdout.write(f"\r{_CSI}2K  {colored} {self._bar()}{self.text}")
            sys.stdout.flush()
            self._stop_event.wait(self.interval)

    def _bar(self) -> str:
        """A compact filled progress bar segment, or ``""`` when no progress is set."""
        prog = self._progress
        if prog is None or prog[1] <= 0:
            return ""
        current, total = prog
        width = 12
        filled = min(int(width * current / total), width)
        bar = f"{_CSI}{self.color}m{'█' * filled}{_RESET}" + dim("░" * (width - filled))
        return f"{bar} {dim(f'{min(current, total)}/{total}')}  "

    def update(self, text: str) -> None:
        self.text = text

    def set_progress(self, current: int, total: int) -> None:
        """Drive the inline progress bar (``current`` of ``total``). Renders on
        the next animation frame; safe to call from the main thread mid-spin."""
        self._progress = (current, total)

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


# -- logging integration ----------------------------------------------------
#
# A drop-in :class:`logging.Formatter` + installer so the stdlib ``logging``
# output wears the same coral theme as the rest of the CLI, instead of the
# default ``WARNING:root:...`` lines. Glyph + color track the level the same
# way the structured one-liners above do (``●`` info, ``▲`` warning, ``✗``
# error), so a log stream and a hand-printed ``style.info(...)`` line read as
# one surface.

#: Level → (glyph, color-code). Anything unmapped falls back to the info look.
_LEVEL_STYLE: "dict[int, tuple[str, str]]" = {
    logging.DEBUG:    ("•", _MUTED),
    logging.INFO:     ("●", _CORAL),
    logging.WARNING:  ("▲", _AMBER),
    logging.ERROR:    ("✗", _RED),
    logging.CRITICAL: ("✗", _RED),
}


class LogFormatter(logging.Formatter):
    """Format log records in the ygg CLI look::

        12:34:56  ●  yggdrasil.databricks.cli  seeding workspace

    A dim clock, a level-colored glyph, the muted logger name, then the
    message — mirroring :func:`event`. Honors the module's color gate
    (:func:`force_color` / ``NO_COLOR`` / TTY autodetect), so a non-ANSI sink
    (file, pipe) degrades to plain text. Tracebacks / stack info append in the
    standard form. Set ``show_name=False`` to drop the logger name for terse
    single-app output.
    """

    def __init__(self, *, show_name: bool = True, datefmt: str = "%H:%M:%S") -> None:
        super().__init__(datefmt=datefmt)
        self.show_name = show_name

    def format(self, record: logging.LogRecord) -> str:
        glyph, code = _LEVEL_STYLE.get(record.levelno, ("●", _CORAL))
        clock = dim(self.formatTime(record, self.datefmt))
        name = f"  {muted(record.name)}" if self.show_name else ""
        line = f"  {clock}  {_esc(code, glyph)}{name}  {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            line += "\n" + self.formatStack(record.stack_info)
        return line


def install_logging(
    level: "int | str" = logging.INFO,
    *,
    logger: "logging.Logger | None" = None,
    show_name: bool = True,
    force: bool = False,
) -> logging.Handler:
    """Install the CLI-styled :class:`LogFormatter` on a stream handler.

    Attaches a single ``StreamHandler`` (stderr, so it never tangles with the
    spinner / structured lines on stdout) carrying :class:`LogFormatter` to
    *logger* (the root logger by default) and sets the level on both. The
    preferred replacement for ``logging.basicConfig(...)`` in ygg CLIs.

    Idempotent: a handler this function previously installed is reused (its
    level + formatter refreshed) rather than stacked, so repeated calls across
    entrypoints don't double every line. ``force=True`` first removes any other
    handlers on *logger* (e.g. a stray ``basicConfig`` default) so the styled
    stream is the only sink. Returns the handler.
    """
    target = logger if logger is not None else logging.getLogger()
    target.setLevel(level)

    existing = next(
        (h for h in target.handlers if getattr(h, "_ygg_styled", False)), None
    )
    if force:
        for h in list(target.handlers):
            if h is not existing:
                target.removeHandler(h)

    handler = existing
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler._ygg_styled = True  # type: ignore[attr-defined]
        target.addHandler(handler)
    handler.setLevel(level)
    handler.setFormatter(LogFormatter(show_name=show_name))
    return handler
