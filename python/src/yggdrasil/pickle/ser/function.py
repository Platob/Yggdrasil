from __future__ import annotations

import functools
import marshal
import types
from dataclasses import dataclass
from typing import Any, ClassVar

from yggdrasil.io import BytesIO

from .registry import REGISTRY
from .serialized import ArraySerialized, Serialized, _COMPRESS_THRESHOLD
from .tags import SerdeTags

__all__ = ["FunctionSerialized"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _referenced_globals(fn: types.FunctionType) -> dict[str, Any]:
    """Return only the global names actually referenced by *fn*'s code."""
    code = fn.__code__
    names: set[str] = set()

    def _collect(co: types.CodeType) -> None:
        names.update(co.co_names)
        for const in co.co_consts:
            if isinstance(const, types.CodeType):
                _collect(const)

    _collect(code)
    return {k: v for k, v in fn.__globals__.items() if k in names}


def _closure_cells(fn: types.FunctionType) -> list[Any]:
    """Extract cell contents from ``__closure__`` (free variables)."""
    if fn.__closure__ is None:
        return []
    return [cell.cell_contents for cell in fn.__closure__]


def _make_cell(value: Any) -> types.CellType:
    """Create a cell object holding *value*."""
    def _factory(v):
        def inner():
            return v
        return inner.__closure__[0]  # type: ignore[index]
    return _factory(value)


def _unwrap_fn(value: Any) -> types.FunctionType:
    """Unwrap decorated callables to the underlying ``FunctionType``.

    Follows the ``__wrapped__`` chain set by :func:`functools.wraps`.
    If the outermost object is already a ``FunctionType`` it is returned as-is.
    """
    seen: set[int] = set()
    obj = value
    while not isinstance(obj, types.FunctionType):
        wrapped = getattr(obj, "__wrapped__", None)
        if wrapped is None:
            raise TypeError(
                f"{type(value)!r} is not a function and has no __wrapped__ attribute; "
                "use ObjectSerialized for arbitrary callables."
            )
        if id(obj) in seen:
            raise ValueError("Circular __wrapped__ chain detected")
        seen.add(id(obj))
        obj = wrapped
    return obj


def _wrapper_attrs(fn: types.FunctionType) -> dict[str, Any]:
    """Return ``__dict__`` attributes copied by :func:`functools.wraps`."""
    return dict(fn.__dict__) if fn.__dict__ else {}


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FunctionSerialized(ArraySerialized):
    """Serialize a Python function, capturing optimized locals & globals.

    Wire layout (array of ``Serialized`` items in order):

    1. ``code_bytes``   – ``BytesSerialized`` of ``marshal.dumps(fn.__code__)``
    2. ``globals_map``  – ``DictSerialized`` (optimized) of referenced globals
    3. ``defaults``     – ``ListSerialized`` of positional defaults
    4. ``kwdefaults``   – ``DictSerialized`` of keyword-only defaults
    5. ``closure``      – ``ListSerialized`` of free-variable cell contents
    6. ``wrapper_dict`` – ``DictSerialized`` of ``fn.__dict__`` (set by
                          ``functools.wraps`` / ``update_wrapper``)

    Metadata keys
    ~~~~~~~~~~~~~
    ``name``        – ``fn.__name__``
    ``qualname``    – ``fn.__qualname__``
    ``module``      – ``fn.__module__``
    ``wrapped``     – ``b"1"`` when the input had a ``__wrapped__`` chain
    ``wrapped_name``– ``__name__`` of the innermost wrapped function
    """

    TAG: ClassVar[int] = SerdeTags.FUNCTION

    # ---- read ----

    @property
    def value(self) -> types.FunctionType:
        items = list(self.iter_())
        if len(items) != 6:
            raise ValueError(
                f"FunctionSerialized expected 6 items, got {len(items)}"
            )
        code_ser, globals_ser, defaults_ser, kwdefaults_ser, closure_ser, wdict_ser = items

        code: types.CodeType = marshal.loads(code_ser.value)

        globs: dict[str, Any] = {}
        if globals_ser.metadata.get(b"optimized") == b"1" and isinstance(
            globals_ser, ArraySerialized
        ):
            it = globals_ser.iter_()
            for key_ser, val_ser in zip(it, it):
                globs[key_ser.value] = val_ser.value
        else:
            globs = globals_ser.value  # type: ignore[assignment]

        defaults: list[Any] = defaults_ser.value
        kwdefaults: dict[str, Any] = kwdefaults_ser.value
        closure_vals: list[Any] = closure_ser.value
        wrapper_dict: dict[str, Any] = wdict_ser.value

        cells = tuple(_make_cell(v) for v in closure_vals) or None  # type: ignore[arg-type]

        fn = types.FunctionType(
            code,
            globs,
            name=self.metadata.get(b"name", b"").decode("utf-8") or code.co_name,
            argdefs=tuple(defaults) if defaults else None,
            closure=cells,
        )
        if kwdefaults:
            fn.__kwdefaults__ = kwdefaults
        fn.__qualname__ = self.metadata.get(b"qualname", b"").decode("utf-8") or fn.__qualname__
        fn.__module__ = self.metadata.get(b"module", b"").decode("utf-8")
        if wrapper_dict:
            fn.__dict__.update(wrapper_dict)

        return fn

    # ---- write ----

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "FunctionSerialized":
        # Accept plain FunctionType or any decorated callable with __wrapped__
        had_wrapped = not isinstance(value, types.FunctionType) or hasattr(value, "__wrapped__")
        fn: types.FunctionType = (
            value if isinstance(value, types.FunctionType) else _unwrap_fn(value)
        )

        buf = BytesIO() if payload is None else payload
        start_index = buf.tell()

        from .scalars import BytesSerialized, StringSerialized
        from .arrays import ListSerialized
        from .maps import DictSerialized

        # 1. code bytes
        BytesSerialized.from_value(marshal.dumps(fn.__code__)).bwrite(buf)

        # 2. referenced globals – native Serialized types (modules → ModuleSerialized)
        ref_globals = _referenced_globals(fn)
        globals_buf = BytesIO()
        globals_start = globals_buf.tell()
        for k, v in ref_globals.items():
            StringSerialized.from_value(k, byte_limit=byte_limit).bwrite(globals_buf)
            Serialized.from_python(v, byte_limit=byte_limit).bwrite(globals_buf)
        globals_size = globals_buf.tell() - globals_start
        DictSerialized(
            metadata={b"optimized": b"1"},
            data=globals_buf,
            size=globals_size,
            start_index=globals_start,
        ).bwrite(buf)

        # 3. positional defaults
        defaults = list(fn.__defaults__) if fn.__defaults__ else []
        ListSerialized.from_value(defaults, byte_limit=byte_limit).bwrite(buf)

        # 4. keyword-only defaults
        kwdefaults = dict(fn.__kwdefaults__) if fn.__kwdefaults__ else {}
        DictSerialized.from_value(kwdefaults, byte_limit=byte_limit).bwrite(buf)

        # 5. closure cell contents
        ListSerialized.from_value(_closure_cells(fn), byte_limit=byte_limit).bwrite(buf)

        # 6. __dict__ (wrapper attributes from functools.update_wrapper / @wraps)
        DictSerialized.from_value(_wrapper_attrs(fn), byte_limit=byte_limit).bwrite(buf)

        size = buf.tell() - start_index
        data, size, start_index, codec = cls._maybe_compress(
            buf, size, start_index, byte_limit=byte_limit,
        )

        md = {} if metadata is None else dict(metadata)
        md[b"name"] = fn.__name__.encode("utf-8")
        md[b"qualname"] = fn.__qualname__.encode("utf-8")
        md[b"module"] = (fn.__module__ or "").encode("utf-8")
        if had_wrapped:
            md[b"wrapped"] = b"1"

        return cls(
            metadata=md,
            data=data,
            size=size,
            start_index=start_index,
            codec=codec,
        )


REGISTRY.register_python_type(types.FunctionType, FunctionSerialized)

