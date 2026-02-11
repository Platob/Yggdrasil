from ..lib import DatabricksError

__all__ = [
    "ComputeException",
    "ClientTerminatedSession",
    "CommandExecutionException"
]


class ComputeException(DatabricksError):
    pass


class CommandExecutionException(ComputeException):
    pass


class ClientTerminatedSession(CommandExecutionException):
    pass
