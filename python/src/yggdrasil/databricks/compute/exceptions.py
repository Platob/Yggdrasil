from yggdrasil.lazy_imports import DatabricksError

__all__ = [
    "ComputeException",
    "ClientTerminatedSession",
    "CommandExecutionException",
    "CommandExecutionError"
]


class ComputeException(DatabricksError):
    pass


class CommandExecutionException(ComputeException):
    pass


class ClientTerminatedSession(CommandExecutionException):
    pass


class CommandExecutionError(ComputeException):
    pass