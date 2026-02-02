from ...exceptions import YGGException

__all__ = [
    "ComputeException",
    "ClientTerminatedSession",
    "CommandExecutionException"
]


class ComputeException(YGGException):
    pass


class CommandExecutionException(ComputeException):
    pass


class ClientTerminatedSession(CommandExecutionException):
    pass
