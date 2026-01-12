"""dataclasses.dataclass module documentation."""

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
    """
    get_dataclass_arrow_field documentation.
    
    Args:
        cls_or_instance: Parameter.
    
    Returns:
        The result.
    """

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
        """
        wrap documentation.
        
        Args:
            c: Parameter.
        
        Returns:
            The result.
        """

        def _init_public_fields(cls):
            """
            _init_public_fields documentation.
            
            Args:
                None.
            
            Returns:
                The result.
            """

            return [
                field
                for field in dataclasses.fields(cls)
                if field.init and not field.name.startswith("_")
            ]

        if not hasattr(c, "default_instance"):
            @classmethod
            def default_instance(cls):
                """
                default_instance documentation.
                
                Args:
                    None.
                
                Returns:
                    The result.
                """

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
                """
                __arrow_field__ documentation.
                
                Args:
                    name: Parameter.
                
                Returns:
                    The result.
                """

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
