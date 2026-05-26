from __future__ import annotations

from dataclasses import dataclass, field

from ._base import Executable, Execution


@dataclass
class ShellCommand(Executable):
    command: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    stdin: str | None = None


@dataclass
class ShellCommandExecution(Execution):
    pass
