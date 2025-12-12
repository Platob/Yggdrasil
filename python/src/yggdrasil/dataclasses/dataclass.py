import dataclasses
from inspect import isclass
from typing import Any, Iterable, Mapping, Tuple

import pyarrow as pa

__all__ = [
    "yggdataclass",
    "is_yggdataclass",
    "get_dataclass_arrow_field"
]

DATACLASS_ARROW_FIELD_CACHE: dict[type, pa.Field] = {}


def is_yggdataclass(cls_or_instance: Any) -> bool:
    """Check if a class or instance is a yggdrasil dataclass.

    Args:
        cls_or_instance: The class or instance to check.
    Returns:
        True if the class or instance
        is a yggdrasil dataclass, False otherwise.
    """
    return hasattr(cls_or_instance, "__arrow_field__")


def get_dataclass_arrow_field(cls_or_instance: Any) -> pa.Field:
    if is_yggdataclass(cls_or_instance):
        return cls_or_instance.__arrow_field__()

    if dataclasses.is_dataclass(cls_or_instance):
        cls = cls_or_instance
        if not isclass(cls_or_instance):
            cls = cls_or_instance.__class__

        existing = DATACLASS_ARROW_FIELD_CACHE.get(cls, None)
        if existing is not None:
            return existing

        from yggdrasil.types.python_arrow import arrow_field_from_hint

        built = arrow_field_from_hint(cls)
        DATACLASS_ARROW_FIELD_CACHE[cls] = built
        return built

    raise ValueError(f"{cls_or_instance!r} is not a dataclass or yggdrasil dataclass")


def yggdataclass(
    cls=None, /,
    *,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False, frozen=False, match_args=True,
    kw_only=False, slots=False,
    weakref_slot=False
):
    """Add dunder methods based on the fields defined in the class.

    Examines PEP 526 __annotations__ to determine fields.

    If init is true, an __init__() method is added to the class. If repr
    is true, a __repr__() method is added. If order is true, rich
    comparison dunder methods are added. If unsafe_hash is true, a
    __hash__() method is added. If frozen is true, fields may not be
    assigned to after instance creation. If match_args is true, the
    __match_args__ tuple is added. If kw_only is true, then by default
    all fields are keyword-only. If slots is true, a new class with a
    __slots__ attribute is returned.
    """

    def wrap(c):
        def _init_public_fields(cls):
            return [
                field
                for field in dataclasses.fields(cls)
                if field.init and not field.name.startswith("_")
            ]

        if not hasattr(c, "to_dict"):
            def to_dict(self) -> Mapping[str, Any]:
                return dataclasses.asdict(self)

            c.to_dict = to_dict

        if not hasattr(c, "from_dict"):
            @classmethod
            def from_dict(cls, values: Mapping[str, Any], *, safe: bool = True):
                defaults = cls.default_instance()
                if not safe:
                    return dataclasses.replace(defaults, **values)

                from yggdrasil.types.cast import convert

                fields = {field.name: field for field in _init_public_fields(cls)}
                converted = {}

                for name, value in values.items():
                    if name not in fields:
                        raise TypeError(f"{name!r} is an invalid field for {cls.__name__}")

                    field = fields[name]
                    default_value = getattr(defaults, name, None)
                    converted[name] = convert(
                        value,
                        field.type,
                    )

                return dataclasses.replace(defaults, **converted)

            c.from_dict = from_dict

        if not hasattr(c, "to_tuple"):
            def to_tuple(self) -> Tuple[Any, ...]:
                return dataclasses.astuple(self)

            c.to_tuple = to_tuple

        if not hasattr(c, "from_tuple"):
            @classmethod
            def from_tuple(cls, values: Iterable[Any], *, safe: bool = True):
                items = tuple(values)
                fields = _init_public_fields(cls)

                if len(items) != len(fields):
                    raise TypeError(
                        f"Expected {len(fields)} values but received {len(items)}"
                    )

                if not safe:
                    kwargs = {field.name: value for field, value in zip(fields, items)}
                    return cls(**kwargs)

                from yggdrasil.types.cast import convert

                defaults = cls.default_instance()
                kwargs = {}

                for field, value in zip(fields, items):
                    if value is None:
                        kwargs[field.name] = getattr(defaults, field.name, None)
                    else:
                        kwargs[field.name] = convert(
                            value,
                            field.type,
                        )

                return cls(**kwargs)

            c.from_tuple = from_tuple

        if not hasattr(c, "default_instance"):
            @classmethod
            def default_instance(cls):
                from yggdrasil.types import default_scalar

                if not hasattr(cls, "__default_instance__"):
                    cls.__default_instance__ = default_scalar(cls)

                return dataclasses.replace(cls.__default_instance__)

            c.default_instance = default_instance

        if not hasattr(c, "__safe_init__"):
            @classmethod
            def __safe_init__(cls, *args, **kwargs):
                """Safely initialize a dataclass using type conversion and defaults."""

                fields = _init_public_fields(cls)
                field_names = [field.name for field in fields]

                if len(args) > len(field_names):
                    raise TypeError(
                        f"Expected at most {len(field_names)} positional arguments, got {len(args)}"
                    )

                provided = {name: value for name, value in zip(field_names, args)}

                for key, value in kwargs.items():
                    if key in provided:
                        raise TypeError(f"Got multiple values for argument '{key}'")
                    if key not in field_names:
                        raise TypeError(
                            f"{key!r} is an invalid field for {cls.__name__}"
                        )

                    provided[key] = value

                from yggdrasil.types.cast import convert

                defaults = cls.default_instance()
                init_kwargs = {}

                for field in fields:
                    if field.name in provided:
                        init_kwargs[field.name] = convert(provided[field.name], field.type)
                    else:
                        init_kwargs[field.name] = getattr(defaults, field.name, None)

                return cls(**init_kwargs)

            c.__safe_init__ = __safe_init__

        if not hasattr(c, "__arrow_field__"):
            @classmethod
            def __arrow_field__(cls, name: str | None = None):
                from yggdrasil.types.python_arrow import arrow_field_from_hint

                return arrow_field_from_hint(cls, name=name)

            c.__arrow_field__ = __arrow_field__

        base = dataclasses.dataclass(
            c,
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            match_args=match_args,
            kw_only=kw_only,
            slots=slots,
        )

        return base

    # See if we're being called as @dataclass or @dataclass().
    if cls is None:
        # We're called with parens.
        return wrap

    # We're called as @dataclass without parens.
    return wrap(cls)
