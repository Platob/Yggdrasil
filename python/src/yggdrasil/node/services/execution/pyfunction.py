from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._base import Executable, Execution


@dataclass
class PyFunction(Executable):
    code: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    max_memory_mb: int | None = None


@dataclass
class PyFunctionExecution(Execution):
    result: Any = None
