"""Logging utilities for Yggdrasil.

Thin layer on top of stdlib :mod:`logging` plus environment-aware UI
helpers (progress bars that render correctly in plain TTYs, IPython /
Jupyter, and Databricks notebooks).

Public surface:

- :func:`get_logger` — module logger under the ``yggdrasil`` namespace.
- :func:`setup_logger` — opt-in stream handler with a sane format,
  matching the ``conftest`` setup so tests and notebooks see the same
  output shape.
- :class:`ProgressBar` — display-aware progress component. Use as an
  iterable wrapper (``for x in ProgressBar(items): ...``) or drive it
  manually with :meth:`ProgressBar.update`.
"""

from .logger import get_logger, setup_logger
from .progress import ProgressBar

__all__ = [
    "get_logger",
    "setup_logger",
    "ProgressBar",
]
