from __future__ import annotations

import ast
import hashlib
import inspect
import marshal
import sys
import textwrap
from dataclasses import dataclass, field as dataclass_field, fields, is_dataclass, make_dataclass
from types import CodeType, FunctionType, MethodType, ModuleType
from typing import Callable, ClassVar, Generic, Mapping

from yggdrasil.environ import runtime_import_module
from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.constants import FORMAT_VERSION
from yggdrasil.pickle.ser.serialized import Serialized, T
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "ComplexSerialized",
    "ModuleSerialized",
    "ClassSerialized",
    "FunctionSerialized",
    "BaseExceptionSerialized",
    "DataclassSerialized",
]

_BUILTINS_KEY = "__builtins__"
_FORMAT_VERSION = FORMAT_VERSION
_PYTHON_VERSION = tuple(sys.version_info[:3])
_MODULE_CACHE: dict[str, ModuleType] = {}
_CLASS_CACHE: dict[tuple[str, str], type[object]] = {}
_FUNCTION_CACHE: dict[tuple[object, ...], Callable[..., object]] = {}
_REFERENCE_FUNCTION_CACHE: dict[tuple[str, str], Callable[..., object]] = {}
_LOCAL_DATACLASS_CACHE: dict[tuple[object, ...], type[object]] = {}


# ---------------------------------------------------------------------------
# basic validators
# ---------------------------------------------------------------------------


def _require_dict(obj: object, *, name: str) -> dict[object, object]:
    if not isinstance(obj, dict):
        raise TypeError(f"{name} must be dict")
    return obj


def _require_tuple(obj: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(obj, tuple):
        raise TypeError(f"{name} must be tuple")
    return obj


def _require_str(obj: object, *, name: str) -> str:
    if not isinstance(obj, str):
        raise TypeError(f"{name} must be str")
    return obj


def _require_bytes(obj: object, *, name: str) -> bytes:
    if not isinstance(obj, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes")
    return bytes(obj)


def _require_list(obj: object, *, name: str) -> list[object]:
    if not isinstance(obj, list):
        raise TypeError(f"{name} must be list")
    return obj


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def _resolve_qualname(root: object, qualname: str) -> object:
    obj = root
    for part in qualname.split("."):
        if part == "<locals>":
            raise AttributeError("Cannot resolve local qualname segment '<locals>'")
        obj = getattr(obj, part)
    return obj


def _make_cell(value: object):
    return (lambda x: lambda: x)(value).__closure__[0]


def _serialize_nested(obj: object) -> bytes:
    return Serialized.from_python_object(obj).write_to().to_bytes()


def _deserialize_nested(blob: bytes) -> object:
    return Serialized.read_from(BytesIO(blob), pos=0).as_python()


def _iter_slots(cls: type[object]) -> tuple[str, ...]:
    out: list[str] = []
    for base in reversed(cls.__mro__):
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in ("__dict__", "__weakref__"):
                out.append(name)
    return tuple(dict.fromkeys(out))


def _extract_object_state(obj: object) -> dict[str, object]:
    state: dict[str, object] = {}

    if hasattr(obj, "__dict__"):
        state.update(getattr(obj, "__dict__", {}))

    for name in _iter_slots(type(obj)):
        if name in state:
            continue
        try:
            state[name] = getattr(obj, name)
        except AttributeError:
            continue

    return state


def _get_declared_attr(cls: type[object], name: str) -> object | None:
    for base in cls.__mro__:
        if name in base.__dict__:
            return base.__dict__[name]
    return None


def _has_meaningful_custom_getstate(obj: object) -> bool:
    attr = _get_declared_attr(type(obj), "__getstate__")
    if attr is None:
        return False
    object_attr = getattr(object, "__getstate__", None)
    return attr is not object_attr


def _has_meaningful_custom_setstate(obj: object) -> bool:
    attr = _get_declared_attr(type(obj), "__setstate__")
    if attr is None:
        return False
    object_attr = getattr(object, "__setstate__", None)
    return attr is not object_attr


def _dump_object_state(obj: object) -> dict[str, object]:
    if _has_meaningful_custom_getstate(obj):
        return {
            "kind": "custom",
            "value": obj.__getstate__(),
        }

    return {
        "kind": "default",
        "value": _extract_object_state(obj),
    }


def _restore_object_state(obj: object, payload: object) -> None:
    data = _require_dict(payload, name="Object state payload")
    kind = _require_str(data["kind"], name="Object state payload kind")
    value = data.get("value")

    if kind == "custom":
        if _has_meaningful_custom_setstate(obj):
            obj.__setstate__(value)
            return

        if isinstance(value, dict):
            for name, item in value.items():
                try:
                    object.__setattr__(obj, name, item)
                except Exception:
                    if hasattr(obj, "__dict__"):
                        obj.__dict__[name] = item
            return

        raise TypeError("Custom object state requires __setstate__ or dict-compatible state")

    if kind == "default":
        state_obj = _require_dict(value, name="Default object state value")
        for name, item in state_obj.items():
            try:
                object.__setattr__(obj, name, item)
            except Exception:
                if hasattr(obj, "__dict__"):
                    obj.__dict__[name] = item
        return

    raise ValueError(f"Unsupported object state payload kind: {kind!r}")


# ---------------------------------------------------------------------------
# module / class / callable reference policy
# ---------------------------------------------------------------------------

def _module_file_contains_site_packages(module_name: str | None) -> bool:
    if not module_name:
        return False

    try:
        module = _module_cache_get_or_load(module_name)
    except Exception:
        return False

    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False

    path = module_file if isinstance(module_file, str) else str(module_file)
    path = path.replace("\\", "/")
    return "site-packages" in path or "site-packages" in path.lower()


def _should_reference_only_module(module_name: str | None) -> bool:
    return _module_file_contains_site_packages(module_name)


def _is_importable_class(cls: type[object]) -> bool:
    return "<locals>" not in cls.__qualname__


def _should_use_reference_only_for_class(cls: type[object]) -> bool:
    return _is_importable_class(cls) and _should_reference_only_module(getattr(cls, "__module__", None))


def _unwrap_method_or_function(obj: object) -> Callable[..., object] | None:
    if isinstance(obj, FunctionType):
        return obj
    if isinstance(obj, MethodType):
        return obj.__func__
    if callable(obj):
        call = getattr(obj, "__call__", None)
        if isinstance(call, MethodType):
            return call.__func__
        if isinstance(call, FunctionType):
            return call
    return None


def _should_use_reference_only_for_callable(obj: object) -> bool:
    fn = _unwrap_method_or_function(obj)
    if fn is None:
        return False

    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)

    if not module_name or not qualname:
        return False
    if "<locals>" in qualname:
        return False

    return _should_reference_only_module(module_name)


# ---------------------------------------------------------------------------
# module / class caches
# ---------------------------------------------------------------------------


def _module_cache_get_or_load(module_name: str) -> ModuleType:
    cached = _MODULE_CACHE.get(module_name)
    if cached is not None:
        return cached

    module = runtime_import_module(module_name)
    _MODULE_CACHE[module_name] = module
    return module


def _class_cache_get_or_load(module_name: str, qualname: str) -> type[object]:
    key = (module_name, qualname)
    cached = _CLASS_CACHE.get(key)
    if cached is not None:
        return cached

    module = _module_cache_get_or_load(module_name)
    obj = _resolve_qualname(module, qualname)

    if not isinstance(obj, type):
        raise TypeError(f"Resolved object is not a class: {module_name}.{qualname}")

    _CLASS_CACHE[key] = obj
    return obj


# ---------------------------------------------------------------------------
# hashing helpers
# ---------------------------------------------------------------------------


def _hash_bytes(data: bytes | None) -> str | None:
    if data is None:
        return None
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _hash_text(data: str | None) -> str | None:
    if data is None:
        return None
    return hashlib.blake2b(data.encode("utf-8"), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# function AST / code helpers
# ---------------------------------------------------------------------------


def _dump_function_code_payload(fn: Callable[..., object]) -> dict[str, object]:
    marshal_code: bytes | None = None
    source_code: str | None = None

    try:
        marshal_code = marshal.dumps(fn.__code__)
    except Exception:
        marshal_code = None

    try:
        source_code = textwrap.dedent(inspect.getsource(fn))
    except Exception:
        source_code = None

    if marshal_code is None and source_code is None:
        raise TypeError(f"Unable to serialize function code for {fn!r}")

    return {
        "python_version": _PYTHON_VERSION,
        "marshal_code": marshal_code,
        "source_code": source_code,
    }


def _find_function_node(
    source: str,
    *,
    fn_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            return node
    return None


def _strip_function_decorators_from_source(source: str, *, fn_name: str) -> str:
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            node.decorator_list = []
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)

    return source


def _root_name(node: ast.AST) -> str | None:
    cur = node

    while True:
        if isinstance(cur, ast.Name):
            return cur.id
        if isinstance(cur, ast.Attribute):
            cur = cur.value
            continue
        if isinstance(cur, ast.Call):
            cur = cur.func
            continue
        if isinstance(cur, ast.Subscript):
            cur = cur.value
            continue
        return None


def _iter_annotation_nodes(fn_node: ast.FunctionDef | ast.AsyncFunctionDef):
    for arg in (
        list(fn_node.args.posonlyargs)
        + list(fn_node.args.args)
        + list(fn_node.args.kwonlyargs)
    ):
        if arg.annotation is not None:
            yield arg.annotation

    if fn_node.args.vararg and fn_node.args.vararg.annotation is not None:
        yield fn_node.args.vararg.annotation

    if fn_node.args.kwarg and fn_node.args.kwarg.annotation is not None:
        yield fn_node.args.kwarg.annotation

    if fn_node.returns is not None:
        yield fn_node.returns


def _iter_default_nodes(fn_node: ast.FunctionDef | ast.AsyncFunctionDef):
    for node in fn_node.args.defaults:
        if node is not None:
            yield node
    for node in fn_node.args.kw_defaults:
        if node is not None:
            yield node


def _collect_load_names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            out.add(child.id)
    return out


def _extract_definition_global_names(source: str, *, fn_name: str) -> set[str]:
    target = _find_function_node(source, fn_name=fn_name)
    if target is None:
        return set()

    names: set[str] = set()

    for dec in target.decorator_list:
        root = _root_name(dec)
        if root:
            names.add(root)
        names.update(_collect_load_names(dec))

    for ann in _iter_annotation_nodes(target):
        names.update(_collect_load_names(ann))

    for default in _iter_default_nodes(target):
        names.update(_collect_load_names(default))

    names.discard(target.name)
    return names


def _drop_function_self_refs(
    mapping: Mapping[object, object],
    *,
    fn: Callable[..., object],
) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        if value is fn:
            continue
        out[key] = value
    return out


def _function_cache_key(
    *,
    module_name: str | None,
    qualname: str,
    python_version: tuple[int, int, int] | None,
    marshal_code: bytes | None,
    source_code: str | None,
    defaults,
    kwdefaults,
    annotations,
    globals_obj: dict[object, object],
    nonlocals_obj: dict[object, object],
) -> tuple[object, ...]:
    marshal_hash = _hash_bytes(bytes(marshal_code) if isinstance(marshal_code, (bytes, bytearray)) else None)
    source_hash = _hash_text(source_code if isinstance(source_code, str) else None)

    return (
        module_name,
        qualname,
        python_version,
        marshal_hash,
        source_hash,
        repr(defaults),
        repr(kwdefaults),
        repr(annotations),
        repr(sorted((k, repr(v)) for k, v in globals_obj.items() if isinstance(k, str))),
        repr(sorted((k, repr(v)) for k, v in nonlocals_obj.items() if isinstance(k, str))),
    )


def _load_function_from_source(
    *,
    source: str,
    globals_dict: dict[str, object],
    name: str,
) -> Callable[..., object]:
    ns = dict(globals_dict)
    stripped_source = _strip_function_decorators_from_source(source, fn_name=name)

    try:
        exec(stripped_source, ns, ns)
    except Exception as e:
        src_msg = stripped_source if len(stripped_source) < 1000 else stripped_source[:1000] + "..."
        raise RuntimeError(f"Failed to exec function source code:\n{src_msg}") from e

    obj = ns.get(name)
    if not callable(obj):
        raise ValueError(f"Failed to rebuild function from source: {name!r}")
    return obj


def _build_function_from_code(
    *,
    code: CodeType,
    globals_dict: dict[str, object],
    module_name: str | None,
    name: str,
    qualname: str,
    defaults,
    kwdefaults,
    annotations,
    closure,
) -> Callable[..., object]:
    fn = FunctionType(
        code,
        globals_dict,
        name=name,
        argdefs=defaults,
        closure=closure,
    )
    fn.__kwdefaults__ = kwdefaults
    fn.__annotations__ = annotations
    fn.__qualname__ = qualname
    if module_name is not None:
        fn.__module__ = module_name
    return fn


def _load_function_code_payload(
    python_version: tuple[int, int, int] | None,
    marshal_code: bytes | None,
    source_code: str | None,
    *,
    globals_dict: dict[str, object],
    module_name: str | None,
    name: str,
    qualname: str,
    defaults,
    kwdefaults,
    annotations,
    closure,
) -> Callable[..., object]:
    same_version = python_version == _PYTHON_VERSION
    errors: list[Exception] = []

    def _try_marshal() -> Callable[..., object]:
        blob = _require_bytes(marshal_code, name="marshal_code")
        code = marshal.loads(blob)
        if not isinstance(code, CodeType):
            raise TypeError("Decoded marshal payload is not a code object")
        return _build_function_from_code(
            code=code,
            globals_dict=globals_dict,
            module_name=module_name,
            name=name,
            qualname=qualname,
            defaults=defaults,
            kwdefaults=kwdefaults,
            annotations=annotations,
            closure=closure,
        )

    def _try_source() -> Callable[..., object]:
        source = _require_str(source_code, name="source_code")
        fn = _load_function_from_source(
            source=source,
            globals_dict=globals_dict,
            name=name,
        )
        fn.__defaults__ = defaults
        fn.__kwdefaults__ = kwdefaults
        fn.__annotations__ = annotations
        fn.__qualname__ = qualname
        if module_name is not None:
            fn.__module__ = module_name
        return fn

    preferred = (_try_marshal, _try_source) if same_version else (_try_source, _try_marshal)

    for builder in preferred:
        try:
            return builder()
        except Exception as exc:
            errors.append(exc)

    raise RuntimeError(
        "Failed to reconstruct function from both marshal and source fallbacks"
    ) from errors[-1]


# ---------------------------------------------------------------------------
# class reference helpers
# ---------------------------------------------------------------------------


def _dump_class_ref(cls: type[object]) -> bytes:
    if "<locals>" in cls.__qualname__:
        raise TypeError(f"Cannot serialize non-importable local class reference: {cls!r}")

    return _serialize_nested(
        {
            "module": cls.__module__,
            "qualname": cls.__qualname__,
        }
    )


def _load_class_ref(data: bytes) -> type[object]:
    payload = _require_dict(_deserialize_nested(data), name="Class payload")
    module_name = _require_str(payload["module"], name="Class payload module")
    qualname = _require_str(payload["qualname"], name="Class payload qualname")
    return _class_cache_get_or_load(module_name, qualname)


# ---------------------------------------------------------------------------
# callable reference-only helpers
# ---------------------------------------------------------------------------


def _dump_reference_function_payload(fn: Callable[..., object]) -> bytes:
    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)

    if not module_name or not qualname:
        raise TypeError(f"Callable does not have stable module/qualname reference: {fn!r}")
    if "<locals>" in qualname:
        raise TypeError(f"Callable is local and cannot be reference-serialized: {fn!r}")

    return _serialize_nested(
        {
            "version": _FORMAT_VERSION,
            "kind": "ref",
            "module": module_name,
            "qualname": qualname,
        }
    )


def _load_reference_function_payload(data: bytes) -> Callable[..., object]:
    payload = _require_dict(_deserialize_nested(data), name="Reference function payload")

    version = payload.get("version")
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported reference function payload version: {version!r}")

    kind = _require_str(payload["kind"], name="Reference function payload kind")
    if kind != "ref":
        raise ValueError(f"Unsupported reference function payload kind: {kind!r}")

    module_name = _require_str(payload["module"], name="Reference function payload module")
    qualname = _require_str(payload["qualname"], name="Reference function payload qualname")

    key = (module_name, qualname)
    cached = _REFERENCE_FUNCTION_CACHE.get(key)
    if cached is not None:
        return cached

    module = _module_cache_get_or_load(module_name)
    obj = _resolve_qualname(module, qualname)
    if not callable(obj):
        raise TypeError(f"Resolved reference is not callable: {module_name}.{qualname}")

    _REFERENCE_FUNCTION_CACHE[key] = obj
    return obj


# ---------------------------------------------------------------------------
# full function payload helpers
# ---------------------------------------------------------------------------


def _dump_function_payload(fn: Callable[..., object]) -> bytes:
    if not isinstance(fn, FunctionType):
        raise TypeError(
            f"FunctionSerialized only supports Python functions, got {type(fn)!r}"
        )

    closurevars = inspect.getclosurevars(fn)
    code_payload = _dump_function_code_payload(fn)

    definition_globals: dict[str, object] = {}
    source_code_obj = code_payload.get("source_code")

    if isinstance(source_code_obj, str):
        try:
            needed_names = _extract_definition_global_names(
                source_code_obj,
                fn_name=fn.__name__,
            )
        except Exception:
            needed_names = set()

        for name in needed_names:
            if name in closurevars.globals or name in closurevars.nonlocals:
                continue
            if name in fn.__globals__:
                definition_globals[name] = fn.__globals__[name]

    payload = {
        "version": _FORMAT_VERSION,
        "kind": "full",
        "module": fn.__module__,
        "name": fn.__name__,
        "qualname": fn.__qualname__,
        "defaults": fn.__defaults__,
        "kwdefaults": fn.__kwdefaults__,
        "annotations": fn.__annotations__,
        "globals": _drop_function_self_refs(closurevars.globals, fn=fn),
        "definition_globals": _drop_function_self_refs(definition_globals, fn=fn),
        "nonlocals": _drop_function_self_refs(closurevars.nonlocals, fn=fn),
        **code_payload,
    }

    return _serialize_nested(payload)


def _load_function_payload(data: bytes) -> Callable[..., object]:
    payload = _require_dict(_deserialize_nested(data), name="Function payload")

    version = payload.get("version")
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported function payload version: {version!r}")

    kind = _require_str(payload.get("kind", "full"), name="Function payload kind")
    if kind == "ref":
        return _load_reference_function_payload(data)
    if kind != "full":
        raise ValueError(f"Unsupported function payload kind: {kind!r}")

    module_name = payload["module"]
    name = _require_str(payload["name"], name="Function payload name")
    qualname = _require_str(payload["qualname"], name="Function payload qualname")

    if not isinstance(module_name, (str, type(None))):
        raise TypeError("Function payload module must be str | None")

    defaults = payload["defaults"]
    kwdefaults = payload["kwdefaults"]
    annotations = payload["annotations"]

    globals_obj = _require_dict(payload["globals"], name="Function payload globals")
    definition_globals_obj = _require_dict(
        payload.get("definition_globals", {}),
        name="Function payload definition_globals",
    )
    nonlocals_obj = _require_dict(payload["nonlocals"], name="Function payload nonlocals")

    cache_key = _function_cache_key(
        module_name=module_name,
        qualname=qualname,
        python_version=payload.get("python_version"),
        marshal_code=payload.get("marshal_code"),
        source_code=payload.get("source_code"),
        defaults=defaults,
        kwdefaults=kwdefaults,
        annotations=annotations,
        globals_obj={**globals_obj, **definition_globals_obj},
        nonlocals_obj=nonlocals_obj,
    )
    cached = _FUNCTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    glb: dict[str, object] = {
        _BUILTINS_KEY: __builtins__,
        "__name__": module_name or "__main__",
    }

    for source in (definition_globals_obj, globals_obj):
        for key, value in source.items():
            if not isinstance(key, str):
                raise TypeError("Global name must be str")
            glb[key] = value

    for key, value in nonlocals_obj.items():
        if not isinstance(key, str):
            raise TypeError("Nonlocal name must be str")
        glb[key] = value

    closure = None
    marshal_code = payload.get("marshal_code")
    if isinstance(marshal_code, (bytes, bytearray)):
        try:
            code_obj = marshal.loads(bytes(marshal_code))
            if isinstance(code_obj, CodeType) and code_obj.co_freevars:
                cells = []
                for freevar_name in code_obj.co_freevars:
                    if freevar_name not in nonlocals_obj:
                        raise ValueError(
                            f"Missing nonlocal capture for freevar {freevar_name!r}"
                        )
                    cells.append(_make_cell(nonlocals_obj[freevar_name]))
                closure = tuple(cells)
        except Exception:
            closure = None

    fn = _load_function_code_payload(
        python_version=payload.get("python_version"),
        marshal_code=payload.get("marshal_code"),
        source_code=payload.get("source_code"),
        globals_dict=glb,
        module_name=module_name,
        name=name,
        qualname=qualname,
        defaults=defaults,
        kwdefaults=kwdefaults,
        annotations=annotations,
        closure=closure,
    )

    fn.__globals__[name] = fn
    _FUNCTION_CACHE[cache_key] = fn
    return fn


# ---------------------------------------------------------------------------
# exception helpers
# ---------------------------------------------------------------------------


def _dump_exception_payload(exc: BaseException) -> bytes:
    return _serialize_nested(
        {
            "version": _FORMAT_VERSION,
            "class": type(exc),
            "args": exc.args,
            "state": _dump_object_state(exc),
        }
    )


def _load_exception_payload(data: bytes) -> BaseException:
    payload = _require_dict(_deserialize_nested(data), name="Exception payload")

    version = payload.get("version")
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported exception payload version: {version!r}")

    exc_cls_obj = payload["class"]
    if not isinstance(exc_cls_obj, type) or not issubclass(exc_cls_obj, BaseException):
        raise TypeError("Decoded exception class is not a BaseException subclass")

    args_obj = _require_tuple(payload["args"], name="Decoded exception args")
    state_payload = payload["state"]

    try:
        exc = exc_cls_obj(*args_obj)
    except Exception:
        exc = BaseException.__new__(exc_cls_obj)
        exc.args = args_obj

    _restore_object_state(exc, state_payload)
    return exc


# ---------------------------------------------------------------------------
# dataclass helpers
# ---------------------------------------------------------------------------


def _safe_dump_annotation(annotation: object) -> dict[str, object]:
    try:
        return {
            "kind": "value",
            "value": _serialize_nested(annotation),
        }
    except Exception:
        return {
            "kind": "repr",
            "value": repr(annotation),
        }


def _safe_load_annotation(payload: object) -> object:
    data = _require_dict(payload, name="Annotation payload")
    kind = _require_str(data["kind"], name="Annotation payload kind")

    if kind == "value":
        blob = _require_bytes(data["value"], name="Annotation payload value")
        return _deserialize_nested(blob)

    if kind == "repr":
        return _require_str(data["value"], name="Annotation payload value")

    raise ValueError(f"Unsupported annotation payload kind: {kind!r}")


def _dump_dataclass_class_payload(cls: type[object]) -> dict[str, object]:
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type, got {cls!r}")

    if _is_importable_class(cls):
        return {
            "kind": "ref",
            "module": cls.__module__,
            "qualname": cls.__qualname__,
        }

    params = cls.__dataclass_params__
    field_payloads: list[dict[str, object]] = []

    for f in fields(cls):
        field_payloads.append(
            {
                "name": f.name,
                "annotation": _safe_dump_annotation(f.type),
                "init": f.init,
                "repr": f.repr,
                "hash": f.hash,
                "compare": f.compare,
                "kw_only": getattr(f, "kw_only", False),
                "metadata": dict(f.metadata) if f.metadata else {},
            }
        )

    return {
        "kind": "local",
        "name": cls.__name__,
        "qualname": cls.__qualname__,
        "module": cls.__module__,
        "params": {
            "repr": params.repr,
            "eq": params.eq,
            "order": params.order,
            "unsafe_hash": params.unsafe_hash,
            "frozen": params.frozen,
            "slots": "__slots__" in cls.__dict__,
        },
        "fields": field_payloads,
    }


def _load_dataclass_class_payload(payload: object) -> type[object]:
    data = _require_dict(payload, name="Dataclass class payload")
    kind = _require_str(data["kind"], name="Dataclass class payload kind")

    if kind == "ref":
        module_name = _require_str(data["module"], name="Dataclass class payload module")
        qualname = _require_str(data["qualname"], name="Dataclass class payload qualname")
        cls = _class_cache_get_or_load(module_name, qualname)
        if not is_dataclass(cls):
            raise TypeError(f"Resolved class is not a dataclass: {module_name}.{qualname}")
        return cls

    if kind != "local":
        raise ValueError(f"Unsupported dataclass class payload kind: {kind!r}")

    name = _require_str(data["name"], name="Dataclass local payload name")
    qualname = _require_str(data["qualname"], name="Dataclass local payload qualname")
    module_name = _require_str(data["module"], name="Dataclass local payload module")
    params = _require_dict(data["params"], name="Dataclass local payload params")
    fields_payload = _require_list(data["fields"], name="Dataclass local payload fields")

    cache_key = (
        module_name,
        qualname,
        repr(params),
        repr(fields_payload),
    )
    cached = _LOCAL_DATACLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    spec = []
    for item in fields_payload:
        field_data = _require_dict(item, name="Dataclass field payload")

        field_name = _require_str(field_data["name"], name="Dataclass field name")
        annotation = _safe_load_annotation(field_data["annotation"])
        init = bool(field_data["init"])
        repr_flag = bool(field_data["repr"])
        hash_flag = field_data["hash"]
        compare_flag = bool(field_data["compare"])
        kw_only = bool(field_data.get("kw_only", False))
        metadata = _require_dict(field_data.get("metadata", {}), name="Dataclass field metadata")

        fld = dataclass_field(
            init=init,
            repr=repr_flag,
            hash=hash_flag,
            compare=compare_flag,
            kw_only=kw_only,
            metadata=metadata,
        )
        spec.append((field_name, annotation, fld))

    namespace = {
        "__module__": module_name,
        "__qualname__": qualname,
    }

    cls = make_dataclass(
        cls_name=name,
        fields=spec,
        namespace=namespace,
        repr=bool(params.get("repr", True)),
        eq=bool(params.get("eq", True)),
        order=bool(params.get("order", False)),
        unsafe_hash=bool(params.get("unsafe_hash", False)),
        frozen=bool(params.get("frozen", False)),
        slots=bool(params.get("slots", False)),
    )

    _LOCAL_DATACLASS_CACHE[cache_key] = cls
    return cls


def _dump_dataclass_payload(obj: object) -> bytes:
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"DataclassSerialized requires a dataclass instance, got {type(obj)!r}")

    cls = type(obj)
    init_values: dict[str, object] = {}
    non_init_values: dict[str, object] = {}

    for f in fields(obj):
        value = getattr(obj, f.name)
        if f.init:
            init_values[f.name] = value
        else:
            non_init_values[f.name] = value

    field_names = {f.name for f in fields(obj)}
    raw_state = _dump_object_state(obj)

    if raw_state["kind"] == "default":
        default_state = _require_dict(raw_state["value"], name="Dataclass default state")
        extra_state_payload: dict[str, object] = {
            "kind": "default",
            "value": {
                key: value
                for key, value in default_state.items()
                if key not in field_names
            },
        }
    else:
        extra_state_payload = raw_state

    payload = {
        "version": _FORMAT_VERSION,
        "class_payload": _dump_dataclass_class_payload(cls),
        "init_values": init_values,
        "non_init_values": non_init_values,
        "extra_state": extra_state_payload,
    }
    return _serialize_nested(payload)


def _load_dataclass_payload(data: bytes) -> object:
    payload = _require_dict(_deserialize_nested(data), name="Dataclass payload")

    version = payload.get("version")
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported dataclass payload version: {version!r}")

    class_payload = payload["class_payload"]
    cls_obj = _load_dataclass_class_payload(class_payload)
    if not isinstance(cls_obj, type) or not is_dataclass(cls_obj):
        raise TypeError("Decoded dataclass class is not a dataclass type")

    init_values = _require_dict(payload["init_values"], name="Dataclass init_values")
    non_init_values = _require_dict(payload["non_init_values"], name="Dataclass non_init_values")
    extra_state_payload = payload.get("extra_state", {"kind": "default", "value": {}})

    obj = cls_obj(**init_values)

    for name, value in non_init_values.items():
        object.__setattr__(obj, name, value)

    _restore_object_state(obj, extra_state_payload)
    return obj


# ---------------------------------------------------------------------------
# serialized wrappers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComplexSerialized(Serialized[T], Generic[T]):
    TAG: ClassVar[int]

    @property
    def value(self) -> T:
        raise NotImplementedError

    def as_python(self) -> T:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if isinstance(obj, (FunctionType, MethodType)):
            return FunctionSerialized.build_function(obj, codec=codec)

        if is_dataclass(obj) and not isinstance(obj, type):
            return DataclassSerialized.build_dataclass(obj, codec=codec)

        if isinstance(obj, BaseException):
            return BaseExceptionSerialized.build_exception(obj, codec=codec)

        if isinstance(obj, type):
            return ClassSerialized.build_class(obj, codec=codec)

        if isinstance(obj, ModuleType):
            return ModuleSerialized.build_module(obj, codec=codec)

        return None


@dataclass(frozen=True, slots=True)
class ModuleSerialized(ComplexSerialized[ModuleType]):
    TAG: ClassVar[int] = Tags.MODULE

    @property
    def value(self) -> ModuleType:
        module_name = self.decode().decode("utf-8")
        return _module_cache_get_or_load(module_name)

    @classmethod
    def build_module(
        cls,
        module: ModuleType,
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        return cls.build(
            tag=cls.TAG,
            data=module.__name__.encode("utf-8"),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ClassSerialized(ComplexSerialized[type[object]]):
    TAG: ClassVar[int] = Tags.CLASS

    @property
    def value(self) -> type[object]:
        return _load_class_ref(self.decode())

    @classmethod
    def build_class(
        cls,
        klass: type[object],
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        return cls.build(
            tag=cls.TAG,
            data=_dump_class_ref(klass),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class FunctionSerialized(ComplexSerialized[Callable[..., object]]):
    TAG: ClassVar[int] = Tags.FUNCTION

    @property
    def value(self) -> Callable[..., object]:
        return _load_function_payload(self.decode())

    @classmethod
    def build_function(
        cls,
        fn: Callable[..., object] | MethodType,
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        callable_obj = _unwrap_method_or_function(fn)
        if callable_obj is None:
            raise TypeError(f"FunctionSerialized requires a Python function or method, got {type(fn)!r}")

        if _should_use_reference_only_for_callable(callable_obj):
            payload = _dump_reference_function_payload(callable_obj)
        else:
            payload = _dump_function_payload(callable_obj)

        return cls.build(
            tag=cls.TAG,
            data=payload,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class BaseExceptionSerialized(ComplexSerialized[BaseException]):
    TAG: ClassVar[int] = Tags.BASE_EXCEPTION

    @property
    def value(self) -> BaseException:
        return _load_exception_payload(self.decode())

    @classmethod
    def build_exception(
        cls,
        exc: BaseException,
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        return cls.build(
            tag=cls.TAG,
            data=_dump_exception_payload(exc),
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class DataclassSerialized(ComplexSerialized[object]):
    TAG: ClassVar[int] = Tags.DATACLASS

    @property
    def value(self) -> object:
        return _load_dataclass_payload(self.decode())

    @classmethod
    def build_dataclass(
        cls,
        obj: object,
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        return cls.build(
            tag=cls.TAG,
            data=_dump_dataclass_payload(obj),
            codec=codec,
        )


for cls in ComplexSerialized.__subclasses__():
    Tags.register_class(cls)
