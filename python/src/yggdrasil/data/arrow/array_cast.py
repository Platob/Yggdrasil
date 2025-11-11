"""Utilities for casting Arrow arrays between compatible schemas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Callable, Dict, Iterable, Optional, Tuple

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - Python < 3.9
    ZoneInfo = None  # type: ignore[misc, assignment]
    ZoneInfoNotFoundError = None  # type: ignore[misc, assignment]

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import types as pa_types


@dataclass(frozen=True)
class ArrowArrayCaster:
    """Cast :class:`pyarrow.Array` instances between two declared fields."""

    source_field: pa.Field
    target_field: pa.Field

    def cast(
        self,
        array: pa.Array,
        target_type: pa.DataType | None = None,
        safe: bool | None = None,
        options: pc.CastOptions | None = None,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.Array:
        """Cast *array* to the target field's type.

        Parameters
        ----------
        memory_pool
        options
        safe
        target_type
        array:
            The Arrow array whose type must match :pyattr:`source_field`.

        Returns
        -------
        :class:`pyarrow.Array`
            The cast array with the target type applied.
        """
        if not array.type.equals(self.source_field.type):
            raise ValueError(
                "Scalar type does not match caster source field: "
                f"{array.type} != {self.source_field.type}"
            )

        resolved_target = target_type or self.target_field.type

        kwargs: Dict[str, object] = {}

        if options is not None:
            kwargs["options"] = options
        else:
            kwargs["target_type"] = resolved_target
            if safe is not None:
                kwargs["safe"] = safe

        if memory_pool is not None:
            kwargs["memory_pool"] = memory_pool

        return array.cast(**kwargs)


class ArrowCastRegistry:
    """Maintain a registry of :class:`ArrowArrayCaster` objects.

    The registry can lazily derive casters for compatible nested schemas to
    minimise upfront configuration.
    """

    def __init__(self) -> None:
        self._casters: Dict[Tuple[str, str], ArrowArrayCaster] = {}

    def register(self, caster: ArrowArrayCaster) -> None:
        """Register an :class:`ArrowArrayCaster` for reuse."""

        self._casters[self._key(caster.source_field, caster.target_field)] = caster

    def get_or_build(
        self,
        source_field: pa.Field | pa.DataType,
        target_field: pa.Field | pa.DataType,
        safe: bool = True
    ) -> ArrowArrayCaster:
        """Return a caster, creating one for compatible nested schemas when needed."""

        source_field = self._ensure_field(source_field)
        target_field = self._ensure_field(target_field)

        key = self._key(source_field, target_field)
        cached = self._casters.get(key)
        if cached is not None:
            return cached

        special = self._build_special_caster(source_field, target_field)
        if special is not None:
            self.register(special)
            return special

        if not self._compatible_nested(source_field, target_field, safe=safe):
            raise KeyError(
                "No caster registered and unable to derive one for the provided fields"
            )

        caster = ArrowArrayCaster(source_field=source_field, target_field=target_field)
        self.register(caster)
        return caster

    def registered_pairs(self) -> Iterable[Tuple[pa.Field, pa.Field]]:
        """Iterate over the currently registered field pairs."""

        for caster in self._casters.values():
            yield caster.source_field, caster.target_field

    def _safe_type(self, obj: pa.DataType | pa.Field):
        if isinstance(obj, pa.DataType):
            return obj
        return obj.type

    def _ensure_field(self, obj: pa.Field | pa.DataType) -> pa.Field:
        if isinstance(obj, pa.Field):
            return obj
        return pa.field("value", obj)

    def _key(self, source_field: pa.Field, target_field: pa.Field) -> Tuple[str, str]:
        return (
            str(self._safe_type(source_field)),
            str(self._safe_type(target_field))
        )

    def _compatible_nested(
        self,
        source_field: pa.Field,
        target_field: pa.Field,
        safe: bool
    ) -> bool:
        """Return ``True`` when the provided fields can be safely cast."""
        source_type = self._safe_type(source_field)
        target_type = self._safe_type(target_field)

        if source_type.equals(target_type):
            return True

        return self._compatible_types(source_type, target_type, safe=safe)

    def _compatible_types(
        self,
        source_type: pa.DataType,
        target_type: pa.DataType,
        safe: bool
    ) -> bool:
        if source_type.equals(target_type):
            return True

        if pa.types.is_string(source_type) and not safe:
            return True

        if pa_types.is_integer(source_type) and pa_types.is_integer(target_type):
            return True
        if pa_types.is_floating(source_type) and pa_types.is_floating(target_type):
            return True
        if pa_types.is_list(source_type) and pa_types.is_list(target_type):
            return self._compatible_types(
                source_type.value_type, target_type.value_type,
                safe=safe
            )
        if pa_types.is_large_list(source_type) and pa_types.is_large_list(target_type):
            return self._compatible_types(
                source_type.value_type, target_type.value_type,
                safe=safe
            )
        if pa_types.is_fixed_size_list(source_type) and pa_types.is_fixed_size_list(target_type):
            return self._compatible_types(
                source_type.value_type, target_type.value_type,
                safe=safe
            )

        if pa_types.is_struct(source_type) and pa_types.is_struct(target_type):
            if source_type.num_fields != target_type.num_fields:
                return False
            source_children = {field.name: field for field in source_type}
            target_children = {field.name: field for field in target_type}

            if source_children.keys() != target_children.keys():
                return False

            return all(
                self._compatible_types(
                    source_children[name].type,
                    target_children[name].type,
                    safe=safe
                )
                for name in source_children
            )

        return False

    def _build_special_caster(
        self, source_field: pa.Field, target_field: pa.Field
    ) -> Optional[ArrowArrayCaster]:
        source_type = self._safe_type(source_field)
        target_type = self._safe_type(target_field)

        if not pa.types.is_string(source_type):
            return None

        if pa.types.is_timestamp(target_type):
            return _StringToTimestampArrayCaster(
                source_field=source_field, target_field=target_field
            )

        if pa.types.is_date32(target_type) or pa.types.is_date64(target_type):
            return _StringToDateArrayCaster(
                source_field=source_field, target_field=target_field
            )

        if pa.types.is_time32(target_type) or pa.types.is_time64(target_type):
            return _StringToTimeArrayCaster(
                source_field=source_field, target_field=target_field
            )

        return None


__all__ = ["ArrowArrayCaster", "ArrowCastRegistry"]


def _parse_string_array(
    array: pa.Array, parser: Callable[[str], object]
) -> list[object | None]:
    values = []
    for value in array.to_pylist():
        if value is None:
            values.append(None)
            continue
        values.append(parser(value))
    return values


def _normalise_timestamp(value: datetime, target_type: pa.TimestampType) -> datetime:
    tzinfo = _resolve_timezone(target_type.tz)
    if tzinfo is not None:
        if value.tzinfo is None:
            return value.replace(tzinfo=tzinfo)
        return value.astimezone(tzinfo)

    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _parse_iso_timestamp(value: str, target_type: pa.TimestampType) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive
        raise pa.ArrowInvalid(f"Invalid ISO 8601 timestamp: {value!r}") from exc
    return _normalise_timestamp(parsed, target_type)


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:  # pragma: no cover - defensive
        raise pa.ArrowInvalid(f"Invalid ISO 8601 date: {value!r}") from exc


def _parse_iso_time(value: str) -> time:
    try:
        return time.fromisoformat(value.strip())
    except ValueError as exc:  # pragma: no cover - defensive
        raise pa.ArrowInvalid(f"Invalid ISO 8601 time: {value!r}") from exc


def _resolve_timezone(name: str | None):
    if name is None:
        return None

    normalised = name.strip()
    if not normalised:
        raise ValueError("Time zone name cannot be empty")

    upper = normalised.upper()
    if upper in {"UTC", "UT", "GMT", "Z"}:
        return timezone.utc

    if ZoneInfo is None:  # pragma: no cover - Python < 3.9
        raise ValueError(
            "Time zone support requires the zoneinfo module to be available"
        )
    try:
        return ZoneInfo(normalised)
    except ZoneInfoNotFoundError as exc:
        if upper == "UTC":
            return timezone.utc
        raise ValueError(f"Unsupported time zone: {name}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported time zone: {name}") from exc


@dataclass(frozen=True)
class _StringToTimestampArrayCaster(ArrowArrayCaster):
    def cast(
        self,
        array: pa.Array,
        target_type: pa.DataType | None = None,
        safe: bool | None = None,
        options: pc.CastOptions | None = None,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.Array:
        if not array.type.equals(self.source_field.type):
            raise ValueError(
                "Scalar type does not match caster source field: "
                f"{array.type} != {self.source_field.type}"
            )

        if options is not None:
            raise TypeError("Custom cast options are not supported for string timestamps")

        resolved_target = target_type or self.target_field.type
        if not pa.types.is_timestamp(resolved_target):
            raise ValueError("Target type must be a timestamp for this caster")

        parsed = _parse_string_array(
            array, lambda value: _parse_iso_timestamp(value, resolved_target)
        )
        return pa.array(parsed, type=resolved_target, memory_pool=memory_pool)


@dataclass(frozen=True)
class _StringToDateArrayCaster(ArrowArrayCaster):
    def cast(
        self,
        array: pa.Array,
        target_type: pa.DataType | None = None,
        safe: bool | None = None,
        options: pc.CastOptions | None = None,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.Array:
        if not array.type.equals(self.source_field.type):
            raise ValueError(
                "Scalar type does not match caster source field: "
                f"{array.type} != {self.source_field.type}"
            )

        if options is not None:
            raise TypeError("Custom cast options are not supported for string dates")

        resolved_target = target_type or self.target_field.type
        if not (pa.types.is_date32(resolved_target) or pa.types.is_date64(resolved_target)):
            raise ValueError("Target type must be a date for this caster")

        parsed = _parse_string_array(array, _parse_iso_date)
        return pa.array(parsed, type=resolved_target, memory_pool=memory_pool)


@dataclass(frozen=True)
class _StringToTimeArrayCaster(ArrowArrayCaster):
    def cast(
        self,
        array: pa.Array,
        target_type: pa.DataType | None = None,
        safe: bool | None = None,
        options: pc.CastOptions | None = None,
        memory_pool: pa.MemoryPool | None = None,
    ) -> pa.Array:
        if not array.type.equals(self.source_field.type):
            raise ValueError(
                "Scalar type does not match caster source field: "
                f"{array.type} != {self.source_field.type}"
            )

        if options is not None:
            raise TypeError("Custom cast options are not supported for string times")

        resolved_target = target_type or self.target_field.type
        if not (pa.types.is_time32(resolved_target) or pa.types.is_time64(resolved_target)):
            raise ValueError("Target type must be a time for this caster")

        parsed = _parse_string_array(array, _parse_iso_time)
        return pa.array(parsed, type=resolved_target, memory_pool=memory_pool)
