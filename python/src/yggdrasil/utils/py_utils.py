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
    "merge_dicts"
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


def is_py_scalar(obj: Any) -> bool:
    return isinstance(obj, (
        str, bytes, bytearray,
        int, float, decimal.Decimal,
        dt.datetime, dt.date, dt.time, dt.timedelta, dt.timezone
    ))