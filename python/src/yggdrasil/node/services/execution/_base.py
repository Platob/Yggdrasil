from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass
class Executable(ABC):
    timeout: float = 30.0


@dataclass
class Execution(ABC):
    status: str = "pending"
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    duration: float | None = None
