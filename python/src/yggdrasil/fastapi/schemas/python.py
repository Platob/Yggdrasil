from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .common import StrictModel


class EnvInfo(StrictModel):
    identifier: str
    python_path: str
    cwd: str
    bin_path: str
    root_path: str
    prefer_uv: bool
    has_uv: bool
    is_current: bool
    version: str | None = None


class CommandResult(StrictModel):
    type: str
    returncode: int | None = None
    exit_code: int | None = None
    pid: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    output: str | None = None
    cwd: str | None = None
    cmd: list[str] | None = None
    args: list[str] | None = None
    command: str | list[str] | None = None
    duration: float | None = None


class RequirementItem(StrictModel):
    name: str
    version: str


class DataFrameColumn(StrictModel):
    name: str
    dtype: str


class DataFramePayload(StrictModel):
    df_name: str
    columns: list[str]
    schema: list[DataFrameColumn]
    rows: list[dict[str, Any]]
    row_count: int
    returned_rows: int
    truncated: bool


class EnvRefRequest(StrictModel):
    identifier: str | None = "current"
    cwd: str | None = None
    prefer_uv: bool = True
    create_if_missing: bool = False
    version: str | None = None
    packages: list[str] = Field(default_factory=list)
    seed: bool = True


class CreateEnvRequest(StrictModel):
    identifier: str
    cwd: str | None = None
    prefer_uv: bool = True
    version: str | None = None
    packages: list[str] = Field(default_factory=list)
    seed: bool = True
    linked: bool = False
    native_tls: bool = True
    clear: bool = False


class DeleteEnvRequest(StrictModel):
    identifier: str
    prefer_uv: bool = True
    raise_error: bool = True


class PackageRequest(StrictModel):
    identifier: str | None = "current"
    cwd: str | None = None
    prefer_uv: bool = True
    create_if_missing: bool = False
    version: str | None = None
    packages: list[str] = Field(default_factory=list)
    requirements: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    target: str | None = None
    break_system_packages: bool = False
    seed: bool = True


class ExecuteCodeRequest(StrictModel):
    code: str
    identifier: str | None = "current"
    cwd: str | None = None
    prefer_uv: bool = True
    create_if_missing: bool = False
    version: str | None = None
    packages: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    globs: dict[str, Any] = Field(default_factory=dict)
    stdin: str | None = None
    auto_install: bool = False
    raise_error: bool = True
    seed: bool = True


class ExcelExecuteRequest(StrictModel):
    code: str
    identifier: str | None = "current"
    cwd: str | None = None
    prefer_uv: bool = True
    create_if_missing: bool = False
    version: str | None = None
    packages: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    globs: dict[str, Any] = Field(default_factory=dict)
    stdin: str | None = None
    auto_install: bool = False
    raise_error: bool = True
    seed: bool = True
    df_name: str = "df"
    max_rows: int | None = None
    include_result: bool = False


class EnvResponse(StrictModel):
    env: EnvInfo


class EnvListResponse(StrictModel):
    items: list[EnvInfo]


class DeleteEnvResponse(StrictModel):
    deleted: bool
    identifier: str
    root_path: str


class RequirementsResponse(StrictModel):
    env: EnvInfo
    requirements: list[RequirementItem]


class MutationResponse(StrictModel):
    ok: bool
    env: EnvInfo
    result: CommandResult | None = None


class ExecutionResponse(StrictModel):
    ok: bool
    env: EnvInfo
    result: CommandResult


class ExcelExecuteResponse(StrictModel):
    ok: bool
    env: EnvInfo
    data: DataFramePayload
    result: CommandResult | None = None


class ExcelPrepareRequest(StrictModel):
    code: str
    identifier: str = "default"
    prefer_uv: bool = True
    create_if_missing: bool = False
    packages: list[str] = Field(default_factory=list)
    df_name: str = "df"
    max_rows: int | None = None
    force_refresh: bool = False


class ManifestData(BaseModel):
    df_name: str
    columns: list[str]
    schema_info: list[DataFrameColumn] = Field(default_factory=list, alias="schema")
    row_count: int
    returned_rows: int
    truncated: bool
    result_path: str | None = None

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ExcelPrepareResponse(StrictModel):
    ok: bool
    data: dict[str, Any]

