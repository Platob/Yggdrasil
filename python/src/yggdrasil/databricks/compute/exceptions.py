from ...exceptions import YGGException

__all__ = [
    "ComputeException",
    "CommandAborted"
]


class ComputeException(YGGException):
    pass


class CommandAborted(YGGException):
    pass
