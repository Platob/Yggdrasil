import inspect
import random
from dataclasses import fields
from typing import Union, Any, Type, Optional, TypeVar

from databricks.sdk.service.sql import (
    EndpointInfo,
    EndpointInfoWarehouseType,
    GetWarehouseResponse, GetWarehouseResponseWarehouseType, WarehousesAPI,
)
from yggdrasil.dataclasses import WaitingConfig

__all__ = [
    "safeGetWarehouseResponse", "safeEndpointInfo",
    "_jitter_sleep_seconds",
    "DEFAULT_ALL_PURPOSE_CLASSIC_NAME", "DEFAULT_ALL_PURPOSE_SERVERLESS_NAME",
    "_EDIT_ARG_NAMES", "_CREATE_ARG_NAMES"
]

DEFAULT_ALL_PURPOSE_CLASSIC_NAME = "Yggdrasil All Purpose"
DEFAULT_ALL_PURPOSE_SERVERLESS_NAME = DEFAULT_ALL_PURPOSE_CLASSIC_NAME + " Serverless"
_CREATE_ARG_NAMES: set[str] = {_ for _ in inspect.signature(WarehousesAPI.create).parameters.keys()}
_EDIT_ARG_NAMES: set[str] = {_ for _ in inspect.signature(WarehousesAPI.edit).parameters.keys()}
T = TypeVar("T")



def safeGetWarehouseResponse(src: Union[GetWarehouseResponse, EndpointInfo]) -> GetWarehouseResponse:
    if isinstance(src, GetWarehouseResponse):
        return src

    payload = _copy_common_fields(src, GetWarehouseResponse, skip={"warehouse_type"})
    payload["warehouse_type"] = _safe_map_enum(
        GetWarehouseResponseWarehouseType,
        getattr(src, "warehouse_type", None),
    )
    return GetWarehouseResponse(**payload)


def safeEndpointInfo(src: Union[GetWarehouseResponse, EndpointInfo]) -> EndpointInfo:
    if isinstance(src, EndpointInfo):
        return src

    payload = _copy_common_fields(src, EndpointInfo, skip={"warehouse_type"})
    payload["warehouse_type"] = _safe_map_enum(
        EndpointInfoWarehouseType,
        getattr(src, "warehouse_type", None),
    )
    return EndpointInfo(**payload)


def _safe_map_enum(dst_enum: Type[T], src_val: Any) -> Optional[T]:
    """Best-effort enum mapping between two SDK enum types."""
    if src_val is None:
        return None
    if isinstance(src_val, dst_enum):
        return src_val
    try:
        return dst_enum(src_val)  # type: ignore[misc]
    except Exception:
        pass
    try:
        return dst_enum(src_val.value)  # type: ignore[misc]
    except Exception:
        pass
    try:
        return dst_enum[src_val.name]  # type: ignore[index]
    except Exception:
        pass
    try:
        return dst_enum(str(src_val))  # type: ignore[misc]
    except Exception:
        return None


def _copy_common_fields(src: Any, dst_cls: Type[T], *, skip: set[str] = frozenset()) -> dict:
    dst_field_names = {f.name for f in fields(dst_cls)}
    return {
        name: getattr(src, name, None)
        for name in dst_field_names
        if name not in skip
    }


def _jitter_sleep_seconds(
    wait: WaitingConfig,
    *,
    iteration: int,
    remaining: Optional[float],
) -> float:
    """Exponential backoff with full jitter, capped by remaining deadline."""
    base = float(wait.interval or 1.0)
    backoff = max(float(wait.backoff or 1.0), 1.0)
    delay = base * (backoff ** iteration)
    if wait.max_interval:
        delay = min(delay, float(wait.max_interval))
    delay = random.uniform(0.0, delay)
    if remaining is not None:
        delay = min(delay, max(0.0, remaining))
    return max(0.0, delay)

