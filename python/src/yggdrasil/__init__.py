from .version import *


def _install_default_log_style() -> None:
    """Paint *unconfigured* log output in the ygg CLI theme — via ``lastResort``.

    Swaps the stdlib's bare last-resort handler (the one ``logging`` falls back
    to only when *nothing* else is configured) for one carrying the coral
    :class:`yggdrasil.cli.style.LogFormatter`. So a plain ``import yggdrasil``
    script shows styled ``WARNING``+ output instead of the stdlib's terse
    ``WARNING:root:...`` — yet the instant an app configures logging
    (``logging.basicConfig(...)``, a handler on root, pytest's ``caplog``, the
    job runner's console handler, …) the last-resort handler steps aside, exactly
    as the stdlib default does.

    This is the whole point of going through ``lastResort`` rather than adding a
    handler to the root logger (what an earlier version did): a root handler
    makes a later ``logging.basicConfig(level=...)`` a silent no-op — the stdlib
    skips it once root has handlers — *and* pins the sink at ``WARNING``, so ygg's
    INFO logs could never be switched on. ``lastResort`` never touches the root
    logger or propagation, so ``logging.basicConfig(level=logging.INFO)`` works
    normally and surfaces ygg's INFO chatter, and log capture keeps working.

    - **Verbosity unchanged** — kept at ``WARNING`` (the stdlib last-resort
      level); raise it the normal way (``logging.basicConfig(level=…)``), which
      then drives the format too.
    - **Only replaces the stdlib default** — never a ``lastResort`` an app or
      another library deliberately installed.
    - Opt out entirely with ``YGG_NO_LOG_STYLE=1``.

    Best-effort: any failure leaves logging untouched.
    """
    import logging
    import os
    import sys

    if os.environ.get("YGG_NO_LOG_STYLE"):
        return
    current = logging.lastResort
    # Only swap out the stdlib's own default (``logging._StderrHandler``) or a
    # missing last-resort — leave a deliberately-installed one alone, and don't
    # double-wrap our own on a re-import.
    if not (current is None or type(current).__name__ == "_StderrHandler"):
        return
    try:
        from yggdrasil.cli.style import LogFormatter

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(LogFormatter())
        handler._ygg_styled = True  # type: ignore[attr-defined]
        logging.lastResort = handler
    except Exception:
        pass


_install_default_log_style()
