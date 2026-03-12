from __future__ import annotations

__all__ = ["SerdeTags"]


class SerdeTags:
    OBJECT: int = 0

    NONE: int = 1
    BYTES: int = 2
    STRING: int = 3
    INT: int = 4
    FLOAT: int = 5
    BOOL: int = 6

    DATE: int = 20
    DATETIME: int = 21
    DECIMAL: int = 22
    UUID: int = 23
    MODULE: int = 24
    FUNCTION: int = 25

    LIST: int = 100
    TUPLE: int = 101
    DICT: int = 102
    SET: int = 103
    FROZENSET: int = 104
    ORDEREDDICT: int = 105

    ARROW_IPC: int = 200