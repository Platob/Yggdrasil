from __future__ import annotations

__all__ = [
    "SerializationError",
    "HeaderDecodeError",
    "InvalidCodecError",
    "MetadataDecodeError",
]


class SerializationError(Exception):
    """Base exception for yggdrasil.pickle.ser."""


class HeaderDecodeError(SerializationError):
    """Raised when a header cannot be parsed from the buffer."""


class InvalidCodecError(SerializationError):
    """Raised when an unknown codec id is encountered."""


class MetadataDecodeError(SerializationError):
    """Raised when encoded metadata is malformed."""