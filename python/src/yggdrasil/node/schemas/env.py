from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class EnvVar(StrictModel):
    key: str
    value: str | None = None


class EnvGetResponse(StrictModel):
    node_id: str
    variables: dict[str, str | None]


class EnvSetRequest(StrictModel):
    variables: dict[str, str | None] = Field(
        default_factory=dict,
        description="Keys to set. A None value unsets the variable.",
    )


class EnvSetResponse(StrictModel):
    node_id: str
    applied: dict[str, str | None]
