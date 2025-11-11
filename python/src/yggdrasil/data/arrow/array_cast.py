"""Utilities for casting Arrow arrays between compatible schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import pyarrow as pa
from pyarrow import types as pa_types
import pyarrow.compute as pc


@dataclass(frozen=True)
class ArrowArrayCaster:
    """Cast :class:`pyarrow.Array` instances between two declared fields."""

    source_field: pa.Field
    target_field: pa.Field
    _cast_impl: Callable[
        [pa.Array, pa.DataType, bool | None, pc.CastOptions | None, pa.MemoryPool | None],
        pa.Array,
    ] | None = field(default=None, repr=False, compare=False)

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
        array:
            The Arrow array whose type must match :pyattr:`source_field`.

        Returns
        -------
        :class:`pyarrow.Array`
            The cast array with the target type applied.
        """

        if not array.type.equals(self.source_field.type):
            raise ValueError(
                "Array type does not match caster source field: "
                f"{array.type} != {self.source_field.type}"
            )
        resolved_target = target_type or self.target_field.type

        if self._cast_impl is not None:
            return self._cast_impl(array, resolved_target, safe, options, memory_pool)

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
        self._builders: List[
            Callable[[pa.Field, pa.Field], ArrowArrayCaster | None]
        ] = []
        self._register_default_string_builders()

    def register(self, caster: ArrowArrayCaster) -> None:
        """Register an :class:`ArrowArrayCaster` for reuse."""

        self._casters[self._key(caster.source_field, caster.target_field)] = caster

    def get_or_build(
        self, source_field: pa.Field | pa.DataType, target_field: pa.Field | pa.DataType
    ) -> ArrowArrayCaster:
        """Return a caster, creating one for compatible nested schemas when needed."""

        source_field = self._ensure_field(source_field, "source")
        target_field = self._ensure_field(target_field, "target")

        key = self._key(source_field, target_field)
        cached = self._casters.get(key)
        if cached is not None:
            return cached

        if self._compatible_nested(source_field, target_field):
            caster = ArrowArrayCaster(
                source_field=source_field, target_field=target_field
            )
            self.register(caster)
            return caster

        for builder in self._builders:
            built = builder(source_field, target_field)
            if built is not None:
                self.register(built)
                return built

        raise KeyError(
            "No caster registered and unable to derive one for the provided fields"
        )

    def registered_pairs(self) -> Iterable[Tuple[pa.Field, pa.Field]]:
        """Iterate over the currently registered field pairs."""

        for caster in self._casters.values():
            yield caster.source_field, caster.target_field

    def _key(self, source_field: pa.Field, target_field: pa.Field) -> Tuple[str, str]:
        return str(source_field.type), str(target_field.type)

    def _compatible_nested(self, source_field: pa.Field, target_field: pa.Field) -> bool:
        """Return ``True`` when the provided fields can be safely cast."""

        if source_field.type.equals(target_field.type):
            return True

        return self._compatible_types(source_field.type, target_field.type)

    def _compatible_types(self, source_type: pa.DataType, target_type: pa.DataType) -> bool:
        if source_type.equals(target_type):
            return True
        if pa_types.is_integer(source_type) and pa_types.is_integer(target_type):
            return True
        if pa_types.is_floating(source_type) and pa_types.is_floating(target_type):
            return True
        if pa_types.is_list(source_type) and pa_types.is_list(target_type):
            return self._compatible_types(source_type.value_type, target_type.value_type)
        if pa_types.is_large_list(source_type) and pa_types.is_large_list(target_type):
            return self._compatible_types(source_type.value_type, target_type.value_type)
        if pa_types.is_fixed_size_list(source_type) and pa_types.is_fixed_size_list(target_type):
            return self._compatible_types(source_type.value_type, target_type.value_type)

        if pa_types.is_struct(source_type) and pa_types.is_struct(target_type):
            if source_type.num_fields != target_type.num_fields:
                return False
            source_children = {field.name: field for field in source_type}
            target_children = {field.name: field for field in target_type}
            if source_children.keys() != target_children.keys():
                return False
            return all(
                self._compatible_types(source_children[name].type, target_children[name].type)
                for name in source_children
            )

        return False

    def _ensure_field(
        self, candidate: pa.Field | pa.DataType, default_name: str
    ) -> pa.Field:
        if isinstance(candidate, pa.Field):
            return candidate
        if isinstance(candidate, pa.DataType):
            return pa.field(default_name, candidate)
        raise TypeError(
            "Fields must be provided as pyarrow.Field or pyarrow.DataType instances"
        )

    def _register_default_string_builders(self) -> None:
        """Register builders that handle flexible string-to-temporal casting."""

        timestamp_formats: Sequence[str] = (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        )
        date_formats: Sequence[str] = ("%Y-%m-%d",)
        time_formats: Sequence[str] = (
            "%H:%M:%S.%f",
            "%H:%M:%S",
        )

        self._builders.append(
            self._string_temporal_builder(timestamp_formats, kind="timestamp")
        )
        self._builders.append(
            self._string_temporal_builder(date_formats, kind="date")
        )
        self._builders.append(
            self._string_temporal_builder(time_formats, kind="time")
        )

    def _string_temporal_builder(
        self, formats: Sequence[str], *, kind: str
    ) -> Callable[[pa.Field, pa.Field], ArrowArrayCaster | None]:
        def builder(source_field: pa.Field, target_field: pa.Field) -> ArrowArrayCaster | None:
            if not pa_types.is_string(source_field.type):
                return None

            target_type = target_field.type

            if kind == "timestamp" and not pa_types.is_timestamp(target_type):
                return None
            if kind == "date" and not (
                pa_types.is_date32(target_type) or pa_types.is_date64(target_type)
            ):
                return None
            if kind == "time" and not (
                pa_types.is_time32(target_type) or pa_types.is_time64(target_type)
            ):
                return None

            def cast_impl(
                array: pa.Array,
                resolved_target: pa.DataType,
                safe: bool | None,
                options: pc.CastOptions | None,
                memory_pool: pa.MemoryPool | None,
            ) -> pa.Array:
                if options is not None:
                    raise TypeError(
                        "String temporal casters do not accept explicit CastOptions"
                    )

                parse_unit = self._strptime_unit(resolved_target)

                last_error: Exception | None = None
                parsed: pa.Array | None = None
                for fmt in formats:
                    try:
                        parsed = pc.strptime(
                            array,
                            options=pc.StrptimeOptions(
                                format=fmt, unit=parse_unit, error_is_null=False
                            ),
                            memory_pool=memory_pool,
                        )
                        break
                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                        last_error = exc
                if parsed is None:
                    try:
                        parsed = self._python_temporal_parse(
                            array, formats, kind=kind, memory_pool=memory_pool
                        )
                    except ValueError as exc:
                        if last_error is not None:
                            raise last_error
                        raise pa.ArrowInvalid("Failed to parse string values for temporal casting") from exc

                if kind == "timestamp":
                    parsed = self._align_timestamp(parsed, resolved_target)
                else:
                    parsed = parsed.cast(resolved_target)

                return parsed

            return ArrowArrayCaster(
                source_field=source_field,
                target_field=target_field,
                _cast_impl=cast_impl,
            )

        return builder

    def _python_temporal_parse(
        self,
        array: pa.Array,
        formats: Sequence[str],
        *,
        kind: str,
        memory_pool: pa.MemoryPool | None,
    ) -> pa.Array:
        values: List[object | None] = []
        last_error: ValueError | None = None
        for raw in array.to_pylist():
            if raw is None:
                values.append(None)
                continue
            for fmt in formats:
                try:
                    parsed = datetime.strptime(raw, fmt)
                except ValueError as exc:
                    last_error = exc
                    continue
                if kind == "date":
                    values.append(parsed.date())
                elif kind == "time":
                    values.append(parsed.time())
                else:
                    values.append(parsed)
                break
            else:
                if last_error is not None:
                    raise last_error
                raise ValueError("Failed to parse value with provided formats")
        return pa.array(values, memory_pool=memory_pool)

    def _strptime_unit(self, target_type: pa.DataType) -> str:
        if pa_types.is_timestamp(target_type):
            return target_type.unit
        if pa_types.is_date32(target_type) or pa_types.is_date64(target_type):
            return "s"
        if pa_types.is_time32(target_type):
            return target_type.unit
        if pa_types.is_time64(target_type):
            return target_type.unit
        raise TypeError("Unsupported temporal target type for string casting")

    def _align_timestamp(
        self, parsed: pa.Array, target_type: pa.DataType
    ) -> pa.Array:
        if not pa_types.is_timestamp(target_type):
            return parsed

        unit = target_type.unit
        target_tz = target_type.tz

        if pa_types.is_timestamp(parsed.type):
            parsed_tz = parsed.type.tz
            if target_tz is None and parsed_tz is not None:
                parsed = parsed.cast(pa.timestamp(parsed.type.unit))
            elif target_tz is not None:
                if parsed_tz is None:
                    parsed = pc.assume_timezone(
                        parsed,
                        timezone=target_tz,
                        ambiguous="raise",
                        nonexistent="raise",
                    )
                elif parsed_tz != target_tz:
                    parsed = parsed.cast(pa.timestamp(parsed.type.unit))
                    parsed = pc.assume_timezone(
                        parsed,
                        timezone=target_tz,
                        ambiguous="raise",
                        nonexistent="raise",
                    )

        desired = pa.timestamp(unit, tz=target_tz)
        if not parsed.type.equals(desired):
            parsed = parsed.cast(desired)

        return parsed


__all__ = ["ArrowArrayCaster", "ArrowCastRegistry"]
