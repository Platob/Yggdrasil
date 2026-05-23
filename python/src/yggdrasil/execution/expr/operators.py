"""Operator enums shared across every backend.

Comparison / logical / arithmetic operators are spelled the same way
in the AST regardless of which engine ultimately renders them.
Backends translate to their own dialect: ``EQ`` becomes ``=`` in
SQL, ``__eq__`` in Python, ``pa.compute.equal`` in pyarrow, etc.

Pulled out of ``nodes.py`` so consumers (backends, planners,
pushdown logic) that only need the operator vocabulary don't pay
the cost of importing the full AST module.
"""

from __future__ import annotations

import enum


__all__ = ["CompareOp", "LogicalOp", "ArithmeticOp"]


class CompareOp(str, enum.Enum):
    """Binary comparison operator, target-engine-agnostic.

    Backends translate to their own dialect: ``EQ`` becomes ``=`` in
    SQL, ``__eq__`` in Python, ``pa.compute.equal`` in pyarrow,
    etc.
    """

    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class LogicalOp(str, enum.Enum):
    AND = "AND"
    OR = "OR"


class ArithmeticOp(str, enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
