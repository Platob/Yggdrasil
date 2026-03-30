"""
yggdrasil.concurrent.threading
==============================
Thread-specific async job primitives.

Canonical exports
-----------------
- :class:`AsyncJob`  — abstract awaitable handle  (``base.py``)
- :class:`ThreadJob` — daemon-thread implementation (``thread_job.py``)

Backward-compatible re-exports
-------------------------------
``Job``, ``JobResult``, and ``JobPoolExecutor`` are also exported here so that
existing ``from yggdrasil.concurrent.threading import …`` statements continue
to work without modification.
"""

from __future__ import annotations

from .base import AsyncJob
from .thread_job import ThreadJob

# Backward-compat re-exports
from ..job import Job
from ..job_result import JobResult
from ..pool import JobPoolExecutor

__all__ = [
    # Thread-specific (canonical)
    "AsyncJob",
    "ThreadJob",
    # Backward-compat re-exports
    "Job",
    "JobResult",
    "JobPoolExecutor",
]

