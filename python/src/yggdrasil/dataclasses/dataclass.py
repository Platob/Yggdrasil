"""Dataclass helpers that integrate with Arrow schemas and safe casting."""
from __future__ import annotations

from dataclasses import MISSING, Field, fields, is_dataclass
from inspect import Parameter, isclass, signature
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Optional, Callable, TypeVar, get_type_hints

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "DATACLASS_ARROW_FIELD_CACHE",
    "dataclass_to_arrow_field",
    "get_from_dict",
    "serialize_dataclass_state",
    "restore_dataclass_state",
    "default_value",
    "lazy_property",
    "YggDataclass",
]

DATACLASS_ARROW_FIELD_CACHE: dict[type, "pa.Field"] = {}
S = TypeVar("S")
T = TypeVar("T")


def dataclass_to_arrow_field(cls_or_instance: Any) -> "pa.Field":
    if not isinstance(cls_or_instance, type):
        cls = cls_or_instance.__class__
    else:
        cls = cls_or_instance

    existing = DATACLASS_ARROW_FIELD_CACHE.get(cls)
    if existing is not None:
        return existing

    from yggdrasil.data.data_field import Field
    built = Field.from_dataclass(cls).to_arrow_field()
    DATACLASS_ARROW_FIELD_CACHE[cls] = built
    return built


def get_from_dict(
    obj: Mapping[str, Any],
    keys: Sequence[str],
    prefix: Optional[str],
) -> Any:
    """Best-effort field lookup with optional prefix support.

    Lookup order for each key:
      1) obj[key]
      2) obj[prefix + key]

    Returns:
      - first non-MISSING value found
      - MISSING if nothing matched
    """
    for key in keys:
        found = obj.get(key, MISSING)
        if found is not MISSING:
            return found

        if prefix:
            found = obj.get(prefix + key, MISSING)
            if found is not MISSING:
                return found

    return MISSING


def default_value(f: Field[Any], with_factory: bool = True) -> Any:
    """Return the effective default value for a dataclass field.

    Returns:
        - f.default when present
        - f.default_factory() when present
        - MISSING otherwise
    """
    if f.default is not MISSING:
        return f.default

    if with_factory and f.default_factory is not MISSING:  # type: ignore[attr-defined]
        return f.default_factory()  # type: ignore[misc]

    return MISSING


def serialize_dataclass_state(obj: Any) -> dict[str, Any]:
    """Serialize constructor state for a dataclass instance.

    Rules:
      - only init=True fields are considered
      - private fields (name starts with "_") are skipped
      - None values are skipped
      - values equal to their effective default are skipped
      - output is a raw payload dict with no version envelope
    """
    payload: dict[str, Any] = {}

    for f in fields(obj):
        if not f.init or f.name.startswith("_"):
            continue

        value = getattr(obj, f.name)

        if value is None:
            continue

        default = default_value(f, with_factory=True)
        if default is not MISSING and value == default:
            continue

        payload[f.name] = value

    return payload


def restore_dataclass_state(obj: Any, state: Any) -> None:
    """Restore dataclass state from a raw payload dict.

    Rules:
      - None is treated as {}
      - unknown keys are ignored
      - missing init=True fields are filled from effective defaults
      - missing required init=True fields raise TypeError
      - non-init fields are reset to their effective defaults when available

    Raises:
        TypeError: If state is not a dict or a required field is missing.
    """
    if state is None:
        payload: dict[str, Any] = {}
    elif isinstance(state, dict):
        payload = state
    else:
        raise TypeError(f"Invalid pickle state for {type(obj).__name__}: {type(state)!r}")

    known_fields = {f.name: f for f in fields(obj)}

    for name, f in known_fields.items():
        if not f.init:
            continue

        if name in payload:
            value = payload[name]
        else:
            value = default_value(f)
            if value is MISSING:
                raise TypeError(
                    f"Cannot restore {type(obj).__name__}: missing required field {name!r}"
                )

        object.__setattr__(obj, name, value)

    for name, f in known_fields.items():
        if f.init:
            continue

        value = default_value(f)
        if value is not MISSING:
            object.__setattr__(obj, name, value)


def lazy_property(
    self: S,
    *,
    cache_attr: str,
    factory: Callable[[S], T],
    use_cache: bool,
) -> T:
    if use_cache:
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached

        created = factory(self)
        object.__setattr__(self, cache_attr, created)
        return created

    return factory(self)


def _stateful_fields(obj_or_cls: Any) -> list[Field[Any]]:
    """Fields that participate in the default getstate/setstate contract.

    Rules:
      - every init=True field
      - every init=False field whose name does not start with "_"

    Private (leading underscore) non-init fields are treated as internal cache
    state and skipped on both serialize and restore.
    """
    return [
        f for f in fields(obj_or_cls)
        if f.init or not f.name.startswith("_")
    ]


def _withable_field(obj_or_cls: Any, name: str) -> Optional[Field[Any]]:
    """Return the dataclass field that backs ``with_<name>``, or None."""
    for f in fields(obj_or_cls):
        if f.name != name:
            continue
        if f.init or not f.name.startswith("_"):
            return f
        return None
    return None


def _init_param_names(cls: type) -> list[str]:
    """Names of regular (non-vararg) ``__init__`` parameters in declaration order.

    Returns an empty list when the class has no inspectable ``__init__`` —
    e.g. C-extension types or classes that inherit ``object.__init__`` only.
    """
    init = getattr(cls, "__init__", None)
    if init is None or init is object.__init__:
        return []
    try:
        sig = signature(init)
    except (TypeError, ValueError):
        return []

    names: list[str] = []
    for i, (name, param) in enumerate(sig.parameters.items()):
        if i == 0:  # skip self
            continue
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        names.append(name)
    return names


def _public_attr_names(obj: Any) -> list[str]:
    """Public instance attribute names for non-dataclass objects.

    Walks ``__dict__`` plus every non-private ``__slots__`` entry up the MRO,
    skipping anything that starts with ``"_"``. Order: ``__dict__`` first, then
    slots in MRO order — gives a stable, predictable serialization order.
    """
    names: list[str] = []
    seen: set[str] = set()

    inst_dict = getattr(obj, "__dict__", None)
    if inst_dict is not None:
        for key in inst_dict:
            if not key.startswith("_") and key not in seen:
                names.append(key)
                seen.add(key)

    for klass in type(obj).__mro__:
        slots = klass.__dict__.get("__slots__")
        if not slots:
            continue
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot.startswith("_") or slot in seen:
                continue
            try:
                getattr(obj, slot)
            except AttributeError:
                continue
            names.append(slot)
            seen.add(slot)

    return names


class YggDataclass:
    """Mixin adding ``copy``, ``with_<field>`` and pickle defaults to dataclasses.

    Apply ``@dataclass`` (frozen or not, slots or not) to a subclass and you get:

    - ``copy(*args, **kwargs)`` — build a sibling instance, replacing init fields
      by position and/or keyword. Conflicting positional + keyword for the same
      field raises ``TypeError`` instead of silently picking a side.
    - ``with_<field>(value, *, inplace=False)`` — single-field updater. Defaults
      to returning a fresh copy; ``inplace=True`` mutates the current instance
      (uses ``object.__setattr__`` so it works on frozen dataclasses too — handy
      for one-shot rehydration but skip it if you rely on hash stability).
    - ``__getstate__`` / ``__setstate__`` — pickle/copy support that round-trips
      every init field plus any non-init field whose name does not start with
      ``"_"``. Init=False fields starting with ``"_"`` are treated as derived
      cache state and rebuilt by ``__post_init__`` (or simply left unset).

    Example::

        @dataclass(frozen=True)
        class Endpoint(YggDataclass):
            host: str
            port: int = 443
            scheme: str = "https"

        ep = Endpoint("api.example.com")
        ep.with_port(8443)                  # → Endpoint('api.example.com', 8443, 'https')
        ep.copy(scheme="http", port=80)     # → Endpoint('api.example.com', 80, 'http')
        ep.copy("other.example.com")        # positional copy, replaces 'host'
    """

    # Empty slots so the mixin stays compatible with @dataclass(slots=True)
    # subclasses — adding a __dict__ here would silently re-enable instance
    # dicts on slotted children.
    __slots__ = ()

    # ── construction helpers ────────────────────────────────────

    def copy(self, *args: Any, **kwargs: Any) -> "YggDataclass":
        """Return a new instance with selected init fields replaced.

        Positional ``args`` map to init fields in declaration order, exactly
        like calling the dataclass constructor.  Keyword ``kwargs`` override
        by name.  Any field not mentioned is carried over from ``self``.

        Works on non-dataclass subclasses too: init fields are derived from
        ``__init__``'s signature and any public non-init attributes are
        carried across via direct assignment after construction.
        """
        cls = type(self)
        is_dc = is_dataclass(cls)

        if is_dc:
            init_fields = [f for f in fields(cls) if f.init]
            init_names = [f.name for f in init_fields]
        else:
            init_names = _init_param_names(cls)

        if len(args) > len(init_names):
            kind = "init fields" if is_dc else "init params"
            raise TypeError(
                f"{cls.__name__}.copy() takes up to {len(init_names)} positional "
                f"args, got {len(args)}. {kind.capitalize()} in order: {init_names}."
            )

        # Start with current values so callers only need to mention what changes.
        init_kwargs: dict[str, Any] = {}
        for name in init_names:
            try:
                init_kwargs[name] = getattr(self, name)
            except AttributeError:
                # Non-dataclass __init__ may take a param the instance never
                # exposes as an attribute (e.g. it gets transformed before
                # being stored). Leave it out and let __init__ fill its own
                # default — error surfaces clearly if there is no default.
                pass

        for name, value in zip(init_names, args):
            if name in kwargs:
                raise TypeError(
                    f"{cls.__name__}.copy(): field {name!r} given both positionally "
                    f"and by keyword. Pick one."
                )
            init_kwargs[name] = value

        if is_dc:
            for key, value in kwargs.items():
                if key not in init_kwargs and key not in init_names:
                    raise TypeError(
                        f"{cls.__name__}.copy(): unknown init field {key!r}. "
                        f"Valid init fields: {init_names}."
                    )
                init_kwargs[key] = value
            return cls(**init_kwargs)

        # Non-dataclass: anything not in the init signature is applied as a
        # plain attribute override after construction. Lets users update
        # post-init state without forcing them to know which fields are
        # constructor params.
        extra_overrides: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in init_names:
                init_kwargs[key] = value
            else:
                extra_overrides[key] = value

        try:
            new = cls(**init_kwargs)
        except TypeError as exc:
            raise TypeError(
                f"{cls.__name__}.copy(): could not reconstruct via {cls.__name__}(...) "
                f"with kwargs {sorted(init_kwargs)}. Original error: {exc}. "
                f"If __init__ takes required arguments not exposed as attributes, "
                f"pass them explicitly to copy()."
            ) from exc

        # Carry public non-init state across so the copy actually mirrors self.
        for name in _public_attr_names(self):
            if name in init_names or name in extra_overrides:
                continue
            try:
                object.__setattr__(new, name, getattr(self, name))
            except AttributeError:
                pass

        for name, value in extra_overrides.items():
            object.__setattr__(new, name, value)

        return new

    # ── dynamic with_<field> ────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal lookup misses, so this stays off
        # the hot path for real attribute access.
        if not name.startswith("with_") or len(name) <= 5 or name.startswith("__"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )

        field_name = name[5:]

        if is_dataclass(self):
            target = _withable_field(self, field_name)
            if target is None:
                available = [
                    f.name for f in fields(self)
                    if f.init or not f.name.startswith("_")
                ]
                raise AttributeError(
                    f"{type(self).__name__!r} has no with_-able field {field_name!r}. "
                    f"Available fields: {available}."
                )
            is_init = target.init
        else:
            # Non-dataclass: every public attribute that exists on self or is
            # accepted by __init__ is with_-able. Private (leading underscore)
            # names stay off-limits for symmetry with the dataclass path.
            if field_name.startswith("_"):
                raise AttributeError(
                    f"{type(self).__name__!r} has no with_-able field {field_name!r}: "
                    f"private (leading underscore) names are not exposed."
                )
            init_names = _init_param_names(type(self))
            exists = field_name in init_names
            if not exists:
                # hasattr would recurse through __getattr__; only check the
                # instance dict / non-private slots we already enumerate.
                exists = field_name in _public_attr_names(self)
            if not exists:
                available = sorted(set(init_names) | set(_public_attr_names(self)))
                hint = f" Known fields: {available}." if available else ""
                raise AttributeError(
                    f"{type(self).__name__!r} has no with_-able field {field_name!r}.{hint}"
                )
            is_init = field_name in init_names

        def setter(value: Any, *, inplace: bool = False) -> Any:
            if inplace:
                object.__setattr__(self, field_name, value)
                return self
            if is_init:
                return self.copy(**{field_name: value})
            new = self.copy()
            object.__setattr__(new, field_name, value)
            return new

        setter.__name__ = name
        setter.__qualname__ = f"{type(self).__name__}.{name}"
        setter.__doc__ = (
            f"Return a copy of this {type(self).__name__} with {field_name!r} "
            f"set to *value*. Pass ``inplace=True`` to mutate this instance instead."
        )
        return setter

    # ── pickle / copy.deepcopy support ──────────────────────────

    def __getstate__(self) -> dict[str, Any]:
        if is_dataclass(self):
            return {f.name: getattr(self, f.name) for f in _stateful_fields(self)}

        # Non-dataclass: init params (when present on self) come first so the
        # serialized order roughly matches the constructor; public non-init
        # attributes follow.
        state: dict[str, Any] = {}
        for name in _init_param_names(type(self)):
            try:
                state[name] = getattr(self, name)
            except AttributeError:
                continue
        for name in _public_attr_names(self):
            if name not in state:
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state: Any) -> None:
        if state is None:
            payload: dict[str, Any] = {}
        elif isinstance(state, Mapping):
            payload = dict(state)
        else:
            raise TypeError(
                f"Cannot restore {type(self).__name__}: expected a dict-like state, "
                f"got {type(state).__name__}."
            )

        if is_dataclass(self):
            cls_fields = _stateful_fields(self)
            for f in cls_fields:
                if f.name in payload:
                    object.__setattr__(self, f.name, payload[f.name])
                    continue
                default = default_value(f)
                if default is not MISSING:
                    object.__setattr__(self, f.name, default)
                elif f.init:
                    raise TypeError(
                        f"Cannot restore {type(self).__name__}: missing required field "
                        f"{f.name!r}. Got keys: {sorted(payload)}."
                    )
            return

        # Non-dataclass: the state dict is authoritative. Skip private
        # (leading underscore) keys so a tampered payload can't smuggle
        # internal fields back in.
        for key, value in payload.items():
            if key.startswith("_"):
                continue
            object.__setattr__(self, key, value)
