"""Specialized AWS Loki behaviors.

Importing this module registers one Loki behavior per AWS service into the
global catalog, mirroring the Databricks fleet. They ``require="aws"`` and
ride the project's :class:`~yggdrasil.aws.AWSClient` (its resolved
credentials / region / role).
"""
from __future__ import annotations

from . import skills as _skills  # noqa: F401

__all__: list[str] = []
