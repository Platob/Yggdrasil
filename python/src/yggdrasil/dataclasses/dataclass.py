import dataclasses
from typing import Any, Iterable, Mapping, Tuple

__all__ = [
    "dataclass"
]

_builtin_dataclass = dataclasses.dataclass


def dataclass(
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

                fields = {field.name: field for field in dataclasses.fields(cls)}
                converted = {}

                for name, value in values.items():
                    if name not in fields:
                        raise TypeError(f"{name!r} is an invalid field for {cls.__name__}")

                    field = fields[name]
                    default_value = getattr(defaults, name, None)
                    converted[name] = convert(
                        value,
                        field.type,
                        default_value=default_value,
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
                fields = dataclasses.fields(cls)

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
                    default_value = getattr(defaults, field.name, None)
                    kwargs[field.name] = convert(
                        value,
                        field.type,
                        default_value=default_value,
                    )

                return cls(**kwargs)

            c.from_tuple = from_tuple

        if not hasattr(c, "default_instance"):
            @classmethod
            def default_instance(cls):
                from yggdrasil.types import default_from_hint

                if not hasattr(cls, "__default_instance__"):
                    cls.__default_instance__ = default_from_hint(cls)

                return dataclasses.replace(cls.__default_instance__)

            c.default_instance = default_instance

        if not hasattr(c, "copy"):
            @classmethod
            def copy(cls, *args, **kwargs):
                """Return a new instance using defaults merged with overrides.

                Positional arguments override fields in definition order, while
                keyword arguments override matching field names. Both sets of
                overrides are applied on top of the cached default instance,
                mirroring the class constructor's positional ordering.
                """

                fields = dataclasses.fields(cls)
                init_fields = [field.name for field in fields if field.init]

                if len(args) > len(init_fields):
                    raise TypeError(
                        f"Expected at most {len(init_fields)} positional arguments, "
                        f"got {len(args)}"
                    )

                positional_overrides = {
                    name: value for name, value in zip(init_fields, args)
                }

                overrides = {**positional_overrides, **kwargs}

                return dataclasses.replace(cls.default_instance(), **overrides)

            c.copy = copy

        if not hasattr(c, "arrow_field"):
            @classmethod
            def arrow_field(cls, name: str | None = None):
                from yggdrasil.types import arrow_field_from_hint

                return arrow_field_from_hint(cls, name=name)

            c.arrow_field = arrow_field

        base = _builtin_dataclass(
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


dataclasses.dataclass = dataclass
