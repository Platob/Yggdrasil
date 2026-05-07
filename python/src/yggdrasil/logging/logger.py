"""Logger setup for the ``yggdrasil`` namespace.

Mirrors the pattern in ``python/tests/conftest.py`` so callers and
tests get the same handler shape without each module rolling its own.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)

ROOT_NAME = "yggdrasil"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger under the ``yggdrasil`` namespace.

    ``name`` may be a bare submodule name (``"data.cast"``) or a fully
    qualified module path (``"yggdrasil.data.cast"``). Both resolve to
    the same logger, which keeps callers from having to think about
    whether they're inside the package or not.
    """
    if not name or name == ROOT_NAME:
        return logging.getLogger(ROOT_NAME)
    if name.startswith(f"{ROOT_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def setup_logger(
    level: int = logging.INFO,
    *,
    fmt: str = DEFAULT_FORMAT,
    stream=None,
    propagate: bool = False,
    force: bool = False,
) -> logging.Logger:
    """Attach a single :class:`StreamHandler` to the ``yggdrasil`` logger.

    Idempotent by default: if a handler is already attached, this only
    updates the level so calling it twice (in a notebook, in a test)
    doesn't double-log. Pass ``force=True`` to wipe and re-install
    handlers — useful when a parent app has already configured logging
    in a way you want to override.
    """
    logger = logging.getLogger(ROOT_NAME)
    logger.setLevel(level)
    logger.propagate = propagate

    if logger.handlers and not force:
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger
