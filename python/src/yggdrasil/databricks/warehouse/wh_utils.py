import inspect
import os
import random
import re
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
    "DEFAULT_ALL_PURPOSE_CLASSIC_NAME",
    "DEFAULT_ALL_PURPOSE_SERVERLESS_NAME",
    "_EDIT_ARG_NAMES", "_CREATE_ARG_NAMES",
    "next_indexed_name",
    "indexed_name_parts", "name_at_index",
    "serverless_sibling_spec"
]

DEFAULT_ALL_PURPOSE_CLASSIC_NAME = os.getenv(
    "DATABRICKS_SQL_WAREHOUSE_NAME",
    "Yggdrasil All Purpose"
).strip()
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


# Matches a trailing " [<int>]" suffix. Captures the base name and the index.
# Anchored at end; tolerates one or more spaces before the bracket.
_INDEXED_NAME_RE = re.compile(r"^(?P<base>.*?)(?:\s+\[(?P<idx>\d+)\])?\s*$")


def next_indexed_name(name: str) -> str:
    """Return ``name`` with its trailing ``[int]`` suffix incremented.

    ``"wh"``        -> ``"wh [2]"``
    ``"wh [2]"``    -> ``"wh [3]"``
    ``"wh [10]"``   -> ``"wh [11]"``
    ``"wh  [2] "``  -> ``"wh [3]"``     (whitespace tolerated)
    ``""``          -> ``" [2]"``        (degenerate; caller should guard)
    """
    m = _INDEXED_NAME_RE.match(name)
    # The regex always matches (everything is optional), so m is never None.
    base = m.group("base").rstrip()
    idx = int(m.group("idx")) if m.group("idx") else 1
    return f"{base} [{idx + 1}]"


def indexed_name_parts(name: str) -> tuple[str, int]:
    """Split ``name`` into ``(base, index)``.

    Unsuffixed names return index ``1`` (the implicit "first" warehouse).

    ``"wh"``      -> ``("wh", 1)``
    ``"wh [2]"``  -> ``("wh", 2)``
    ``"wh [10]"`` -> ``("wh", 10)``
    """
    m = _INDEXED_NAME_RE.match(name or "")
    base = (m.group("base") or "").rstrip()
    idx = int(m.group("idx")) if m.group("idx") else 1
    return base, idx


def name_at_index(name: str, index: int) -> str:
    """Return ``name`` rewritten with ``[index]`` suffix (or no suffix if index==1)."""
    base, _ = indexed_name_parts(name)
    return base if index == 1 else f"{base} [{index}]"


def serverless_sibling_spec(
    details: EndpointInfo,
    *,
    name: Optional[str] = None,
) -> dict:
    """Build a ``create_warehouse`` kwargs dict for a serverless sibling.

    Clones the editable subset of ``details`` (filtered through
    :data:`_EDIT_ARG_NAMES`), forces ``enable_serverless_compute=True``
    for fast cold-start, and assigns ``name`` — defaulting to the next
    ``[idx]``-suffixed sibling of ``details.name``.
    """
    new_name = name or next_indexed_name(getattr(details, "name", "") or "warehouse")

    spec = {
        k: v
        for k, v in details.as_shallow_dict().items()
        if k in _EDIT_ARG_NAMES and v is not None
    }
    spec.pop("id", None)
    spec.pop("name", None)

    if "name" in _CREATE_ARG_NAMES:
        spec["name"] = new_name
    else:
        spec["warehouse_name"] = new_name

    spec["enable_serverless_compute"] = True
    return spec