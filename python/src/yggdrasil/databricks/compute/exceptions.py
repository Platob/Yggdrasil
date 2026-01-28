from ...exceptions import YGGException

__all__ = [
    "ComputeException",
    "CommandExecutionAborted",
    "CommandExecutionException"
]


class ComputeException(YGGException):
    pass


class CommandExecutionException(ComputeException):
    pass


class CommandExecutionAborted(CommandExecutionException):
    pass
