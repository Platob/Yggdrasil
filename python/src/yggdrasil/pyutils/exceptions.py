"""Utilities for parsing and re-raising exceptions from traceback strings."""

import builtins
import dataclasses as dc
import re
from typing import Type


__all__ = [
    "ParsedException",
    "RemoteTraceback",
    "parse_exception_from_traceback",
    "raise_parsed_traceback"
]


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Matches typical "SomeError: message" final line(s)
# Also catches things like "ValueError: ..." or "ModuleNotFoundError: ..."
_EXC_LINE_RE = re.compile(
    r"(?m)^\s*([A-Za-z_]\w*(?:Error|Exception|Warning))\s*:\s*(.*?)\s*$"
)

# Some tracebacks end with just "KeyboardInterrupt" or "SystemExit" (no colon/message)
_BARE_EXC_RE = re.compile(r"(?m)^\s*([A-Za-z_]\w*(?:Error|Exception|Warning|Interrupt|Exit))\s*$")


@dc.dataclass(frozen=True)
class ParsedException:
    """Structured representation of a parsed exception type and message."""
    exc_type: Type[BaseException]
    message: str
    raw_type_name: str


class RemoteTraceback(Exception):
    """Holds a traceback *string* and prints it as the chained cause."""
    def __init__(self, traceback_text: str):
        """Store the traceback text for later display.

        Args:
            traceback_text: Traceback string to store.

        Returns:
            None.
        """
        super().__init__("Remote traceback (text)")
        self.traceback_text = traceback_text

    def __str__(self) -> str:
        """Render the exception with its stored traceback text.

        Returns:
            Rendered exception string with traceback text.
        """
        return f"{self.args[0]}\n\n{self.traceback_text}"


def parse_exception_from_traceback(tb_text: str) -> ParsedException:
    """
    Parse a traceback string, infer exception class + message.
    Falls back to RuntimeError if it can't infer a proper built-in exception.
    """
    clean = _ANSI_RE.sub("", tb_text or "").strip()

    # Prefer the last "Type: message" line in the whole text
    matches = list(_EXC_LINE_RE.finditer(clean))
    if matches:
        name, msg = matches[-1].group(1), matches[-1].group(2)
        exc_cls = getattr(builtins, name, None)
        if isinstance(exc_cls, type) and issubclass(exc_cls, BaseException):
            return ParsedException(exc_cls, msg, name)
        return ParsedException(RuntimeError, f"{name}: {msg}", name)

    # Try a bare exception name at the end (no message)
    bare = list(_BARE_EXC_RE.finditer(clean))
    if bare:
        name = bare[-1].group(1)
        exc_cls = getattr(builtins, name, None)
        if isinstance(exc_cls, type) and issubclass(exc_cls, BaseException):
            return ParsedException(exc_cls, "", name)
        return ParsedException(RuntimeError, name, name)

    # No idea → just wrap the whole thing
    return ParsedException(RuntimeError, clean, "RuntimeError")


def raise_parsed_traceback(tb_text: str, *, attach_as_cause: bool = True) -> None:
    """
    Infer exception from traceback text and raise it.
    Note: cannot recreate original frames; we chain the text as the cause by default.
    """
    parsed = parse_exception_from_traceback(tb_text)
    exc = parsed.exc_type(parsed.message) if parsed.message else parsed.exc_type()

    if attach_as_cause:
        raise exc from RemoteTraceback(tb_text)
    raise exc
