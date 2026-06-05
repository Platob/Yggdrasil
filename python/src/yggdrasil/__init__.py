from .version import *


def _install_default_log_style() -> None:
    """Give the root logger the ygg CLI-styled formatter on import.

    So ``import yggdrasil`` already paints log output in the coral CLI theme
    (:class:`yggdrasil.cli.style.LogFormatter`) instead of the stdlib's
    ``WARNING:root:...`` default. Conservative on purpose:

    - **No-op when logging is already configured** (the root logger has
      handlers) — an application that set up its own logging wins, mirroring
      :func:`logging.basicConfig`'s contract so importing ygg as a dependency
      never hijacks a host app's log format.
    - **Doesn't change verbosity** — installs at the existing root level
      (``WARNING`` by default); callers wanting ygg's INFO chatter raise the
      level themselves (or call ``style.install_logging(logging.INFO)``).
    - Opt out entirely with ``YGG_NO_LOG_STYLE=1``.

    Best-effort: any failure leaves logging untouched.
    """
    import logging
    import os

    if os.environ.get("YGG_NO_LOG_STYLE"):
        return
    root = logging.getLogger()
    if root.handlers:
        return
    try:
        from yggdrasil.cli.style import install_logging

        install_logging(root.level or logging.WARNING)
    except Exception:
        pass


_install_default_log_style()
