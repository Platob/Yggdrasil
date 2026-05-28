from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeRole(StrEnum):
    DRIVER = "driver"
    EXECUTOR = "executor"
    HYBRID = "hybrid"
