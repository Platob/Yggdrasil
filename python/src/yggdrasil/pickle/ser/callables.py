"""
Callable serialization: functions and bound methods.

This module provides:
- All AST / symtable / source-inspection helpers for functions
- _dump_function_payload / _load_function_payload
- _dump_method_payload  / _load_method_payload
- FunctionSerialized, MethodSerialized

All shared utilities (caches, hashing, object-state, …) live in libs.py.
complexs.py re-exports everything for backward compatibility.
"""

from __future__ import annotations

import ast
import inspect
import marshal
import symtable
import textwrap
from dataclasses import dataclass
from types import CodeType, FunctionType, MethodType
from typing import Callable, ClassVar, Mapping

from yggdrasil.environ import PyEnv
from yggdrasil.pickle.ser.libs import (
    _BUILTINS_KEY,
    _FORMAT_VERSION,
    _PYTHON_VERSION,
    _deserialize_nested,
    _hash_bytes,
    _hash_text,
    _make_cell,
    _module_cache_get_or_load,
    _require_bytes,
    _require_dict,
    _require_str,
    _require_tuple_len,
    _resolve_qualname,
    _serialize_nested,
    _should_reference_only_module,
    ComplexSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

def _dump_callable_annotations_payload(annotations: object) -> tuple[int, dict[str, tuple[str, object]]]:
    from yggdrasil.pickle.ser.annotations import dump_function_annotations

    return dump_function_annotations(annotations if isinstance(annotations, Mapping) else None)


def _load_callable_annotations_payload(payload: object) -> dict[str, object]:
    from yggdrasil.pickle.ser.annotations import load_function_annotations

    return load_function_annotations(payload)


__all__ = [
    "FunctionSerialized",
    "MethodSerialized",
]

# ---------------------------------------------------------------------------
# callable-specific caches
# ---------------------------------------------------------------------------

_REFERENCE_FUNCTION_CACHE: dict[tuple[str, str], Callable[..., object]] = {}
# Key: (module_name, qualname, marshal_hash, source_hash) — payload hashes
# prevent a cached result from masking a different payload (e.g. corrupted
# marshal during source-fallback tests).
_FULL_FUNCTION_CACHE: dict[
    tuple[str | None, str, str | None, str | None], Callable[..., object]
] = {}
_MODULE_IMPORT_BINDINGS_CACHE: dict[str, frozenset[str]] = {}
_IN_DATABRICKS = PyEnv.in_databricks()

# ---------------------------------------------------------------------------
# function payload format constants
# ---------------------------------------------------------------------------

_FN_REF = 0
_FN_FULL = 1

_FN_REF_VERSION = 0
_FN_REF_KIND = 1
_FN_REF_MODULE = 2
_FN_REF_QUALNAME = 3

_FN_FULL_VERSION = 0
_FN_FULL_KIND = 1
_FN_FULL_MODULE = 2
_FN_FULL_NAME = 3
_FN_FULL_QUALNAME = 4
_FN_FULL_DEFAULTS = 5
_FN_FULL_KWDEFAULTS = 6
_FN_FULL_ANNOTATIONS = 7
_FN_FULL_GLOBALS = 8
_FN_FULL_DEFINITION_GLOBALS = 9
_FN_FULL_NONLOCALS = 10
_FN_FULL_PY_VERSION = 11
_FN_FULL_MARSHAL = 12
_FN_FULL_SOURCE = 13

_METHOD_VERSION = 0
_METHOD_FUNCTION = 1
_METHOD_SELF = 2


# ---------------------------------------------------------------------------
# unwrap / reference-only detection
# ---------------------------------------------------------------------------

def _unwrap_py_function(fn: Callable[..., object]) -> FunctionType:
    unwrapped = inspect.unwrap(fn)
    if not isinstance(unwrapped, FunctionType):
        raise TypeError(f"Expected Python function after unwrap, got {type(unwrapped)!r}")
    return unwrapped


def _unwrap_method_or_function(obj: object) -> Callable[..., object] | None:
    if isinstance(obj, FunctionType):
        return _unwrap_py_function(obj)
    if isinstance(obj, MethodType):
        fn = obj.__func__
        return _unwrap_py_function(fn) if isinstance(fn, FunctionType) else None
    if callable(obj):
        call = getattr(obj, "__call__", None)
        if isinstance(call, MethodType):
            fn = call.__func__
            return _unwrap_py_function(fn) if isinstance(fn, FunctionType) else None
        if isinstance(call, FunctionType):
            return _unwrap_py_function(call)
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
# code-payload dump
# ---------------------------------------------------------------------------

def _dump_function_code_payload(fn: Callable[..., object]) -> tuple[
    tuple[int, int, int],
    bytes | None,
    str | None,
]:
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

    return _PYTHON_VERSION, marshal_code, source_code


# ---------------------------------------------------------------------------
# AST helpers for source manipulation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# module source / import-binding helpers
# ---------------------------------------------------------------------------

def _load_module_source_text(module_name: str | None) -> str | None:
    if not module_name:
        return None

    try:
        module = _module_cache_get_or_load(module_name)
    except Exception:
        return None

    try:
        return inspect.getsource(module)
    except Exception:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return None

        path = module_file if isinstance(module_file, str) else str(module_file)
        if path.endswith((".pyc", ".pyo")):
            path = path[:-1]

        try:
            with open(path, encoding="utf-8") as fh:
                return fh.read()
        except Exception:
            return None


class _ModuleImportBindingCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            self.names.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None


def _module_import_bindings_get_or_load(module_name: str | None) -> frozenset[str]:
    if not module_name:
        return frozenset()

    cached = _MODULE_IMPORT_BINDINGS_CACHE.get(module_name)
    if cached is not None:
        return cached

    source = _load_module_source_text(module_name)
    if not isinstance(source, str):
        bindings = frozenset()
    else:
        try:
            tree = ast.parse(source)
        except Exception:
            bindings = frozenset()
        else:
            collector = _ModuleImportBindingCollector()
            collector.visit(tree)
            bindings = frozenset(collector.names)

    _MODULE_IMPORT_BINDINGS_CACHE[module_name] = bindings
    return bindings


# ---------------------------------------------------------------------------
# symbol-table helpers for global-name extraction
# ---------------------------------------------------------------------------

def _find_symbol_table_by_name(
    table: symtable.SymbolTable,
    *,
    name: str,
) -> symtable.SymbolTable | None:
    for child in table.get_children():
        if child.get_name() == name:
            return child
    return None


def _collect_referenced_global_names_from_symbol_table(
    table: symtable.SymbolTable,
) -> set[str]:
    names: set[str] = set()

    for identifier in table.get_identifiers():
        symbol = table.lookup(identifier)
        if symbol.is_referenced() and symbol.is_global():
            names.add(identifier)

    for child in table.get_children():
        names.update(_collect_referenced_global_names_from_symbol_table(child))

    return names


def _extract_used_names_from_function_source(source: str, *, fn_name: str) -> set[str]:
    module_table = symtable.symtable(source, "<function-source>", "exec")
    fn_table = _find_symbol_table_by_name(module_table, name=fn_name)
    if fn_table is None:
        return set()

    return _collect_referenced_global_names_from_symbol_table(fn_table)


# ---------------------------------------------------------------------------
# globals-context extraction
# ---------------------------------------------------------------------------

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


def _safe_getclosurevars(fn: Callable[..., object]):
    try:
        return inspect.getclosurevars(fn)
    except Exception:
        class _EmptyClosureVars:
            globals = {}
            nonlocals = {}
            builtins = {}
            unbound = set()
        return _EmptyClosureVars()


def _infer_imported_globals_from_source_module(
    fn: FunctionType,
    *,
    source_code: str | None,
) -> dict[str, object]:
    if not isinstance(source_code, str):
        return {}

    try:
        used_names = _extract_used_names_from_function_source(source_code, fn_name=fn.__name__)
    except Exception:
        return {}

    if not used_names:
        return {}

    import_bindings = _module_import_bindings_get_or_load(getattr(fn, "__module__", None))
    if not import_bindings:
        return {}

    inferred: dict[str, object] = {}
    for name in used_names:
        if name not in import_bindings or name not in fn.__globals__:
            continue
        value = fn.__globals__[name]
        if value is fn:
            continue
        inferred[name] = value

    return _drop_function_self_refs(inferred, fn=fn)


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


def _collect_outer_function_runtime_context(
    fn: FunctionType,
) -> tuple[dict[str, object], dict[str, object]]:
    closurevars = _safe_getclosurevars(fn)
    runtime_globals = _drop_function_self_refs(closurevars.globals, fn=fn)
    nonlocals_dict = _drop_function_self_refs(closurevars.nonlocals, fn=fn)
    return runtime_globals, nonlocals_dict


def _collect_inner_function_definition_globals(
    outer_fn: FunctionType,
    inner_fn: FunctionType,
    *,
    source_code: str | None,
) -> dict[str, object]:
    definition_globals: dict[str, object] = {}

    if not isinstance(source_code, str):
        return definition_globals

    try:
        needed_names = _extract_definition_global_names(
            source_code,
            fn_name=inner_fn.__name__,
        )
    except Exception:
        needed_names = set()

    inner_closurevars = _safe_getclosurevars(inner_fn)

    for name in needed_names:
        if name in inner_closurevars.globals or name in inner_closurevars.nonlocals:
            continue

        if name in inner_fn.__globals__:
            value = inner_fn.__globals__[name]
        elif name in outer_fn.__globals__:
            value = outer_fn.__globals__[name]
        else:
            continue

        if value is inner_fn or value is outer_fn:
            continue

        definition_globals[name] = value

    return _drop_function_self_refs(definition_globals, fn=inner_fn)


# ---------------------------------------------------------------------------
# function code load helpers
# ---------------------------------------------------------------------------

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
        src_msg = (
            stripped_source
            if len(stripped_source) < 1000
            else stripped_source[:1000] + "..."
        )
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
# reference-function payload
# ---------------------------------------------------------------------------

def _dump_reference_function_payload(fn: Callable[..., object]) -> bytes:
    if not isinstance(fn, FunctionType):
        raise TypeError(f"Expected FunctionType, got {type(fn)!r}")

    inner_fn = _unwrap_py_function(fn)

    module_name = getattr(inner_fn, "__module__", None)
    qualname = getattr(inner_fn, "__qualname__", None)

    if not module_name or not qualname:
        raise TypeError(f"Callable does not have stable module/qualname reference: {fn!r}")
    if "<locals>" in qualname:
        raise TypeError(f"Callable is local and cannot be reference-serialized: {fn!r}")

    return _serialize_nested((_FORMAT_VERSION, _FN_REF, module_name, qualname))


def _load_reference_function_payload(data: bytes) -> Callable[..., object]:
    payload = _require_tuple_len(
        _deserialize_nested(data),
        name="Reference function payload",
        expected=4,
    )

    version = payload[_FN_REF_VERSION]
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported reference function payload version: {version!r}")

    kind = payload[_FN_REF_KIND]
    if kind != _FN_REF:
        raise ValueError(f"Unsupported reference function payload kind: {kind!r}")

    module_name = _require_str(payload[_FN_REF_MODULE], name="Reference function payload module")
    qualname = _require_str(payload[_FN_REF_QUALNAME], name="Reference function payload qualname")

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
# full function payload
# ---------------------------------------------------------------------------

def _dump_function_payload(fn: Callable[..., object]) -> bytes:
    if not isinstance(fn, FunctionType):
        raise TypeError(
            f"FunctionSerialized only supports Python functions, got {type(fn)!r}"
        )

    outer_fn = fn
    inner_fn = _unwrap_py_function(fn)

    python_version, marshal_code, outer_source_code = _dump_function_code_payload(outer_fn)

    try:
        _inner_pyver, _inner_marshal, inner_source_code = _dump_function_code_payload(inner_fn)
        _ = _inner_pyver, _inner_marshal
    except Exception:
        inner_source_code = None

    outer_runtime_globals, nonlocals_dict = _collect_outer_function_runtime_context(outer_fn)
    inner_closurevars = _safe_getclosurevars(inner_fn)

    runtime_globals = dict(outer_runtime_globals)
    runtime_globals.update(_drop_function_self_refs(inner_closurevars.globals, fn=inner_fn))

    source_globals_fn = inner_fn if isinstance(inner_source_code, str) else outer_fn
    source_globals_code = (
        inner_source_code if isinstance(inner_source_code, str) else outer_source_code
    )
    runtime_globals.update(
        _infer_imported_globals_from_source_module(
            source_globals_fn,
            source_code=source_globals_code,
        )
    )

    definition_globals = _collect_inner_function_definition_globals(
        outer_fn,
        inner_fn,
        source_code=inner_source_code,
    )

    # Use inner_fn's signature metadata (defaults, kwdefaults, annotations) so
    # that cross-version source-fallback reconstruction gets the correct defaults.
    #
    # Background: @wraps does NOT copy __defaults__ / __kwdefaults__ from the
    # wrapped function to the wrapper, so outer_fn.__defaults__ is typically
    # None for decorated functions.  The source stored in this payload is
    # inner_fn's source, so during source-fallback the exec'd function already
    # has the right defaults — but then "_try_source" overwrites them with
    # `fn.__defaults__ = defaults`.  Storing inner_fn's values here keeps that
    # override correct.  When inner_fn is outer_fn (no decoration) the values
    # are identical, so behaviour is unchanged for plain functions.
    _sig_fn = inner_fn  # prefer inner (real) function's signature metadata
    safe_annotations = _dump_callable_annotations_payload(
        getattr(_sig_fn, "__annotations__", None)
    )

    payload = (
        _FORMAT_VERSION,
        _FN_FULL,
        outer_fn.__module__,
        outer_fn.__name__,
        outer_fn.__qualname__,
        _sig_fn.__defaults__,
        _sig_fn.__kwdefaults__,
        safe_annotations,
        runtime_globals,
        definition_globals,
        nonlocals_dict,
        python_version,
        marshal_code,
        inner_source_code,
    )

    return _serialize_nested(payload)


def _load_function_payload(data: bytes) -> Callable[..., object]:
    payload_obj = _deserialize_nested(data)

    if not isinstance(payload_obj, tuple):
        raise TypeError("Function payload must be tuple")

    if len(payload_obj) == 4:
        return _load_reference_function_payload(data)

    payload = _require_tuple_len(payload_obj, name="Function payload", expected=14)

    version = payload[_FN_FULL_VERSION]
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported function payload version: {version!r}")

    kind = payload[_FN_FULL_KIND]
    if kind == _FN_REF:
        return _load_reference_function_payload(data)
    if kind != _FN_FULL:
        raise ValueError(f"Unsupported function payload kind: {kind!r}")

    module_name = payload[_FN_FULL_MODULE]
    name = _require_str(payload[_FN_FULL_NAME], name="Function payload name")
    qualname = _require_str(payload[_FN_FULL_QUALNAME], name="Function payload qualname")

    if not isinstance(module_name, (str, type(None))):
        raise TypeError("Function payload module must be str | None")

    # Extract payload data early so we can include it in the cache key.
    # This prevents a cached result from a prior load (e.g. with valid marshal)
    # from masking a deliberately different payload (e.g. corrupted marshal for
    # source-fallback tests, or both-corrupt payloads that must raise).
    marshal_code = payload[_FN_FULL_MARSHAL]
    source_code = payload[_FN_FULL_SOURCE]

    _marshal_bytes: bytes | None = (
        marshal_code if isinstance(marshal_code, (bytes, bytearray)) else None
    )
    _source_str: str | None = source_code if isinstance(source_code, str) else None

    # Cache hit: non-local (importable) functions are stable — return early.
    # Key includes payload hashes so different marshal/source data never alias.
    _cache_key = (
        module_name,
        qualname,
        _hash_bytes(_marshal_bytes),
        _hash_text(_source_str),
    )
    if "<locals>" not in qualname:
        _cached_fn = _FULL_FUNCTION_CACHE.get(_cache_key)
        if _cached_fn is not None:
            return _cached_fn

    defaults = payload[_FN_FULL_DEFAULTS]
    kwdefaults = payload[_FN_FULL_KWDEFAULTS]
    annotations = _load_callable_annotations_payload(payload[_FN_FULL_ANNOTATIONS])

    globals_obj = _require_dict(payload[_FN_FULL_GLOBALS], name="Function payload globals")
    definition_globals_obj = _require_dict(
        payload[_FN_FULL_DEFINITION_GLOBALS],
        name="Function payload definition_globals",
    )
    nonlocals_obj = _require_dict(payload[_FN_FULL_NONLOCALS], name="Function payload nonlocals")

    python_version = payload[_FN_FULL_PY_VERSION]
    if not isinstance(python_version, (tuple, type(None))):
        raise TypeError("Function payload python_version must be tuple | None")

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
        python_version=python_version,
        marshal_code=marshal_code if isinstance(marshal_code, (bytes, bytearray)) else None,
        source_code=source_code if isinstance(source_code, str) else None,
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

    # Cache non-local (importable) functions so subsequent loads skip reconstruction
    if "<locals>" not in qualname:
        _FULL_FUNCTION_CACHE[_cache_key] = fn

    return fn


# ---------------------------------------------------------------------------
# method payload
# ---------------------------------------------------------------------------

def _dump_method_payload(method: MethodType) -> bytes:
    if not isinstance(method, MethodType):
        raise TypeError(f"MethodSerialized requires MethodType, got {type(method)!r}")

    raw_fn = method.__func__
    if not isinstance(raw_fn, FunctionType):
        raise TypeError(f"Method __func__ must be FunctionType, got {type(raw_fn)!r}")

    self_obj = method.__self__

    if _should_use_reference_only_for_callable(raw_fn):
        fn_payload = _dump_reference_function_payload(raw_fn)
    else:
        fn_payload = _dump_function_payload(raw_fn)

    self_payload = _serialize_nested(self_obj)

    return _serialize_nested(
        (
            _FORMAT_VERSION,
            fn_payload,
            self_payload,
        )
    )


def _load_method_payload(data: bytes) -> MethodType:
    payload = _require_tuple_len(
        _deserialize_nested(data), name="Method payload", expected=3
    )

    version = payload[_METHOD_VERSION]
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported method payload version: {version!r}")

    fn_blob = _require_bytes(payload[_METHOD_FUNCTION], name="Method payload function")
    self_blob = _require_bytes(payload[_METHOD_SELF], name="Method payload self")

    fn = _load_function_payload(fn_blob)
    if not isinstance(fn, FunctionType):
        raise TypeError(f"Decoded method function must be FunctionType, got {type(fn)!r}")

    self_obj = _deserialize_nested(self_blob)

    return MethodType(fn, self_obj)


# ---------------------------------------------------------------------------
# serializer classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FunctionSerialized(ComplexSerialized[Callable[..., object]]):
    TAG: ClassVar[int] = Tags.FUNCTION

    @property
    def value(self) -> Callable[..., object]:
        return _load_function_payload(self.decode())

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Make the serialized function directly callable.

        Delegates to the deserialized function so that code receiving a
        ``FunctionSerialized`` instead of a plain function can still invoke
        it transparently (e.g. ``_f(*_a, **_k)`` in remote execution
        snippets).
        """
        return self.as_cache_python()(*args, **kwargs)

    @classmethod
    def build_function(
        cls,
        fn: Callable[..., object],
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        if not isinstance(fn, FunctionType):
            raise TypeError(f"FunctionSerialized requires FunctionType, got {type(fn)!r}")

        if _should_use_reference_only_for_callable(fn):
            payload = _dump_reference_function_payload(fn)
        else:
            payload = _dump_function_payload(fn)

        return cls.build(
            tag=cls.TAG,
            data=payload,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class MethodSerialized(FunctionSerialized):
    TAG: ClassVar[int] = Tags.METHOD

    @property
    def value(self) -> MethodType:
        return _load_method_payload(self.decode())

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Make the serialized method directly callable.

        Delegates to the deserialized bound method so that code receiving a
        ``MethodSerialized`` instead of a plain bound method can still invoke
        it transparently.
        """
        return self.as_cache_python()(*args, **kwargs)

    @classmethod
    def build_method(
        cls,
        method: MethodType,
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        if not isinstance(method, MethodType):
            raise TypeError(f"MethodSerialized requires MethodType, got {type(method)!r}")

        return cls.build(
            tag=cls.TAG,
            data=_dump_method_payload(method),
            codec=codec,
        )


for _cls in (FunctionSerialized, MethodSerialized):
    Tags.register_class(_cls, tag=_cls.TAG)

FunctionSerialized = Tags.get_class(Tags.FUNCTION) or FunctionSerialized
MethodSerialized = Tags.get_class(Tags.METHOD) or MethodSerialized

