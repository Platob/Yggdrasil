import datetime as dt
import decimal
import json
import sys
from json import JSONDecodeError
from typing import Any, Optional, Iterable, Callable

# Annotated is available in Python 3.9+
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    try:
        from typing_extensions import Annotated
    except ImportError:
        Annotated = None

__all__ = [
    "Annotated",
    "safe_str",
    "safe_bytes",
    "safe_bool",
    "safe_dict",
    "safe_int",
    "merge_dicts",
    "index_of",
    "parse_decimal_metadata",
    "parse_time_metadata",
    "parse_timestamp_metadata"
]

TRUE_STR_VALUES = {"True", "true", "1", "Yes", "yes"}
TRUE_BYTES_VALUES = {_.encode("utf-8") for _ in TRUE_STR_VALUES}

FALSE_STR_VALUES = {"False", "false", "0", "No", "no"}
FALSE_BYTES_VALUES = {_.encode("utf-8") for _ in FALSE_STR_VALUES}


def safe_str(obj: Any, default = None) -> Optional[str]:
    if not obj:
        return default

    if isinstance(obj, str):
        return obj

    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8")

    return str(obj)


def safe_bytes(obj: Any, default = None) -> Optional[bytes]:
    if not obj:
        return default

    if isinstance(obj, bytes):
        return obj

    if isinstance(obj, str):
        return obj.encode("utf-8")

    if isinstance(obj, (bytearray, memoryview)):
        return bytes(obj)

    if is_py_scalar(obj):
        return str(obj).encode("utf-8")
    return json.dumps(obj).encode("utf-8")


def safe_bool(obj: Any, default = None) -> Optional[bool]:
    if obj is None:
        return default

    if isinstance(obj, bool):
        return obj

    if isinstance(obj, str):
        if obj in TRUE_STR_VALUES:
            return True
        elif obj in FALSE_STR_VALUES:
            return False
        else:
            return default

    if isinstance(obj, (bytes, bytearray)):
        if obj in TRUE_BYTES_VALUES:
            return True
        elif obj in FALSE_BYTES_VALUES:
            return False
        else:
            return default

    return bool(obj)


def safe_int(obj: Any, default = None) -> Optional[int]:
    if not obj:
        return default

    if isinstance(obj, int):
        return obj

    if isinstance(obj, bool):
        return 1 if obj else 0

    return int(obj)


def safe_dict(
    obj: Any,
    default: dict = None,
    check_key: Callable = None,
    check_value: Callable = None,
    raise_error: bool = True
) -> Optional[dict]:
    if not obj:
        return default

    if isinstance(obj, dict):
        if check_key or check_value:
            check_key = check_key or (lambda x: x)
            check_value = check_value or (lambda x: x)

            obj = {
                check_key(k): check_value(v)
                for k, v in obj.items()
            }

        if all(bool(_) for _ in obj.keys()):
            return obj

        return {
            k: v
            for k, v in obj.items()
            if k
        }

    if isinstance(obj, (str, bytes, bytearray)):
        try:
            parsed = json.loads(obj)
        except JSONDecodeError as e:
            if raise_error:
                raise JSONDecodeError(
                    f"Cannot parse {type(obj)}({obj}): {e}",
                    doc=e.doc,
                    pos=e.pos
                )
            return default

        return safe_dict(
            parsed, default,
            check_key=check_key,
            check_value=check_value,
            raise_error=raise_error
        )

    check_key = check_key or (lambda x: x)
    check_value = check_value or (lambda x: x)

    if isinstance(obj, tuple) and len(obj) == 2:
        if is_py_scalar(obj[0]) and is_py_scalar(obj[1]):
            return {
                check_key(obj[0]): check_value(obj[1])
            }

    if isinstance(obj, Iterable):
        sane = {}

        for item in obj:
            if item:
                if not isinstance(item, dict):
                    item = safe_dict(
                        item, default=None,
                        check_key=check_key, check_value=check_value,
                        raise_error=raise_error
                    )

                if item:
                    sane.update(item)

        return safe_dict(
            sane, default=default,
            check_key=check_key, check_value=check_value,
            raise_error=raise_error
        )

    raise TypeError(f"Cannot convert {obj} to dict")


def merge_dicts(dicts: Iterable, default = None) -> Optional[dict]:
    if not dicts:
        return default

    merged = {}

    for d in dicts:
        d = safe_dict(d, default=None)

        if d:
            merged.update(d)

    return merged or default


def index_of(
    collection: list[str],
    value: str,
    strict_names: bool | None = None,
    raise_error: bool = True
) -> int:
    try:
        return collection.index(value)
    except ValueError:
        if strict_names:
            if raise_error:
                raise ValueError(f"Cannot find '{value}' in {collection}")
            return -1

        idx = 0

        for item in collection:
            if safe_str(value).casefold() == safe_str(item).casefold():
                return idx
            idx += 1

        if raise_error:
            raise ValueError(f"Cannot find '{value}' in {collection}")

        return -1


def is_py_scalar(obj: Any) -> bool:
    return isinstance(obj, (
        str, bytes, bytearray,
        int, float, decimal.Decimal,
        dt.datetime, dt.date, dt.time, dt.timedelta, dt.timezone
    ))


def parse_decimal_metadata(metadata: dict) -> tuple[int, int]:
    """
    Parse precision and scale for a decimal type from metadata.

    Args:
        metadata: Dictionary containing metadata with precision and scale information

    Returns:
        A tuple of (precision, scale) for decimal types

    Notes:
        - Looks for keys 'precision' and 'scale' in the metadata
        - Both string and bytes keys are supported
        - Default precision is 38 and scale is 18 if not found
        - Values can be provided as strings or integers
    """
    precision = 38  # Default precision
    scale = 18      # Default scale

    # Check for string keys using get() to handle None gracefully
    precision_val = metadata.get('precision')
    if precision_val is not None:
        precision = safe_int(precision_val, default=precision)

    scale_val = metadata.get('scale')
    if scale_val is not None:
        scale = safe_int(scale_val, default=scale)

    # Check for bytes keys using get() (PyArrow stores metadata keys as bytes)
    precision_val = metadata.get(b'precision')
    if precision_val is not None:
        if isinstance(precision_val, bytes):
            precision_val = precision_val.decode('utf-8')
        precision = safe_int(precision_val, default=precision)

    scale_val = metadata.get(b'scale')
    if scale_val is not None:
        if isinstance(scale_val, bytes):
            scale_val = scale_val.decode('utf-8')
        scale = safe_int(scale_val, default=scale)

    # Ensure valid precision and scale
    if precision <= 0:
        precision = 38

    if scale < 0 or scale > precision:
        scale = min(18, precision)

    return precision, scale


def parse_time_metadata(metadata: dict) -> str:
    """
    Parse time resolution unit from metadata.

    Args:
        metadata: Dictionary containing metadata with time unit information

    Returns:
        Time unit string ('s', 'ms', 'us', or 'ns')

    Notes:
        - Looks for key 'unit' in the metadata
        - Both string and bytes keys are supported
        - Default unit is 'us' (microseconds) if not found
        - Valid values are: 's', 'ms', 'us', 'ns'
    """
    default_unit = 'us'  # Default unit is microseconds
    valid_units = ['s', 'ms', 'us', 'ns']

    # Check for string keys using get() to handle None gracefully
    unit_val = metadata.get('unit')
    if unit_val is not None:
        if isinstance(unit_val, bytes):
            unit_val = unit_val.decode('utf-8')
        if unit_val in valid_units:
            return unit_val

    # Check for bytes keys using get() (PyArrow stores metadata keys as bytes)
    unit_val = metadata.get(b'unit')
    if unit_val is not None:
        if isinstance(unit_val, bytes):
            unit_val = unit_val.decode('utf-8')
        if unit_val in valid_units:
            return unit_val

    return default_unit


def parse_timestamp_metadata(metadata: dict) -> tuple[str, str | None]:
    """
    Parse timestamp unit and timezone from metadata.

    Args:
        metadata: Dictionary containing metadata with timestamp information

    Returns:
        A tuple of (unit, timezone) where:
        - unit is one of: 's', 'ms', 'us', 'ns'
        - timezone is a valid timezone string or None

    Notes:
        - Looks for keys 'unit' and 'tz' in the metadata
        - Both string and bytes keys are supported
        - Default unit is 'us' (microseconds) if not found
        - Default timezone is None (timezone-naive) if not found
        - If timezone is specified but invalid, None will be returned for timezone
    """
    # First, get the time unit using the existing function
    unit = parse_time_metadata(metadata)

    # Now handle timezone
    timezone = None

    # Check for string keys using get() to handle None gracefully
    tz_val = metadata.get('tz')
    if tz_val is not None:
        if isinstance(tz_val, bytes):
            tz_val = tz_val.decode('utf-8')
        timezone = safe_str(tz_val)

    # Check for bytes keys using get() (PyArrow stores metadata keys as bytes)
    tz_val = metadata.get(b'tz')
    if tz_val is not None:
        if isinstance(tz_val, bytes):
            tz_val = tz_val.decode('utf-8')
        timezone = safe_str(tz_val)

    # Empty string timezone should be treated as None
    if timezone == '':
        timezone = None

    return unit, timezone