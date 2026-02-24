from dataclasses import dataclass

from yggdrasil.arrow import Field

__all__ = [
    "FieldCastError"
]


@dataclass
class FieldCastError(ValueError):
    source: Field
    target: Field
    message: str = ""

    def __post_init__(self):
        if not self.message:
            self.message = "internal error"
