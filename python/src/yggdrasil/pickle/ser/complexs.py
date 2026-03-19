from __future__ import annotations

import ast
import fnmatch
import hashlib
import importlib
import importlib.util
import inspect
import io
import logging
import marshal
import sys
import tempfile
import textwrap
import zipfile
from dataclasses import MISSING, dataclass, field as dataclass_field, fields, is_dataclass, make_dataclass
from pathlib import Path
from types import CodeType, FunctionType, MethodType, ModuleType
from typing import Callable, ClassVar, Generic, Mapping

from yggdrasil.environ import runtime_import_module
from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.serialized import Serialized, T
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "ComplexSerialized",
    "ModuleSerialized",
    "ClassSerialized",
    "FunctionSerialized",
    "MethodSerialized",
    "BaseExceptionSerialized",
    "DataclassSerialized",
]


# ============================================================================
# Runtime constants / caches
# ============================================================================

_BUILTINS_KEY = "__builtins__"
_FORMAT_VERSION = 1
_PYTHON_VERSION = tuple(sys.version_info[:3])

_LOGGER = logging.getLogger(__name__)

# Maximum total (filtered) directory size for inline module serialisation.
_MAX_MODULE_INLINE_BYTES: int = 1024  # 1 Kb

# ---------------------------------------------------------------------------
# Module whitelist – modules that should always be serialised as a name-only
# reference because they are part of the stdlib, too large, or expected to
# already be installed on the target environment.
# ---------------------------------------------------------------------------

_MODULE_WHITELIST_NAMES: frozenset[str] = frozenset(
    getattr(sys, "stdlib_module_names", set())
) | frozenset({
    # stdlib extras / aliases sometimes not in stdlib_module_names
    "builtins", "importlib", "encodings", "codecs", "_thread",
    # well-known third-party packages that should be imported by name
    "numpy", "np",
    "pandas", "pd",
    "polars", "pl",
    "pyarrow", "pa",
    "pyspark",
    "scipy",
    "sklearn", "scikit_learn",
    "torch", "torchvision", "torchaudio",
    "tensorflow", "tf",
    "jax", "jaxlib",
    "matplotlib", "mpl_toolkits",
    "seaborn",
    "plotly",
    "bokeh",
    "PIL", "pillow",
    "cv2", "opencv",
    "requests",
    "httpx",
    "aiohttp",
    "flask",
    "fastapi",
    "django",
    "celery",
    "sqlalchemy",
    "alembic",
    "boto3", "botocore",
    "google", "googleapis",
    "azure",
    "databricks",
    "setuptools", "pip", "wheel", "pkg_resources",
    "pytest", "unittest", "nose",
    "cryptography", "jwt", "certifi",
    "yaml", "pyyaml",
    "toml", "tomli", "tomllib",
    "click", "typer", "argparse",
    "attrs", "pydantic",
    "grpc", "grpcio", "protobuf",
    "cython", "cffi", "ctypes",
    "msgpack",
    "orjson", "ujson", "simplejson",
    "dask",
    "ray",
    "mlflow",
    "wandb",
    "transformers", "tokenizers", "datasets", "huggingface_hub",
    "langchain", "openai", "anthropic",
    "IPython", "ipykernel", "ipywidgets", "notebook", "nbformat",
    "mongoengine", "pymongo", "uv", "yggdrasil"
})

# ---------------------------------------------------------------------------
# Module zip cache
# ---------------------------------------------------------------------------
# Keyed by (module_name, root_path_str) → zip bytes.
# Avoids re-zipping the same module directory across repeated serialisations.
_MODULE_ZIP_CACHE: dict[tuple[str, str], bytes] = {}

# ---------------------------------------------------------------------------
# Module zip exclusion rules
# ---------------------------------------------------------------------------

_MODULE_EXCLUDE_DIRS: frozenset[str] = frozenset({
    "__pycache__",
    "__pypackages__",
    ".git", ".svn", ".hg",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis",
    ".tox", ".nox",
    ".venv", "venv", "env", ".env",
    "node_modules",
    ".idea", ".vscode", ".vs",
    "build", "dist",
    ".eggs",
    "htmlcov", "coverage",
    ".ipynb_checkpoints",
    "tmp", "temp",
})

_MODULE_EXCLUDE_PATTERNS: frozenset[str] = frozenset({
    "*.egg-info",
    "*.dist-info",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.o",
    "*.obj",
    "*.a",
    "*.lib",
    "*.class",
})


def _is_whitelisted_module(module_name: str) -> bool:
    """Return True if *module_name* (root package) should stay name-only."""
    root = module_name.split(".", 1)[0]
    return root in _MODULE_WHITELIST_NAMES


def _should_exclude_module_path(parts: tuple[str, ...]) -> bool:
    """Return True when a relative path inside a module tree should be skipped."""
    for part in parts:
        if part in _MODULE_EXCLUDE_DIRS:
            return True
        if any(fnmatch.fnmatch(part, p) for p in _MODULE_EXCLUDE_PATTERNS):
            return True
    return False


def _get_module_root_path(module: ModuleType) -> Path | None:
    """Return the filesystem root directory of a module/package, or None."""
    name = module.__name__.split(".", 1)[0]
    try:
        spec = importlib.util.find_spec(name)
    except (ValueError, ModuleNotFoundError):
        spec = None

    if spec is not None and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()

    mod_file = getattr(module, "__file__", None)
    if mod_file:
        p = Path(mod_file).resolve()
        # single-file module → return its parent so we can grab that one file
        return p.parent if p.is_file() else p

    return None


def _module_dir_filtered_bytes(root: Path) -> int:
    """Total bytes of files that would be included in a module zip."""
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel_parts = p.relative_to(root).parts
        except ValueError:
            continue
        if _should_exclude_module_path(rel_parts):
            continue
        total += p.stat().st_size
    return total


def _zip_module_to_bytes(root: Path) -> bytes:
    """Zip a module directory, excluding junk / system files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if not p.exists():
                continue
            arcname = p.relative_to(root)
            if _should_exclude_module_path(arcname.parts):
                continue
            if p.is_dir():
                if arcname.parts:
                    zf.writestr(f"{arcname.as_posix().rstrip('/')}/", b"")
            elif p.is_file():
                zf.write(filename=p, arcname=arcname.as_posix())
    return buf.getvalue()


def _get_or_zip_module(module_name: str, root: Path) -> bytes:
    """Return cached module zip bytes, or build and cache them."""
    key = (module_name, str(root))
    cached = _MODULE_ZIP_CACHE.get(key)
    if cached is not None:
        return cached
    data = _zip_module_to_bytes(root)
    _MODULE_ZIP_CACHE[key] = data
    return data


def _extract_module_zip(data: bytes, module_name: str) -> Path:
    """Extract module zip to a temp directory and add it to sys.path."""
    root_name = module_name.split(".", 1)[0]
    tmp_root = Path(tempfile.mkdtemp(prefix=f"ygg_mod_{root_name}_"))
    pkg_dir = tmp_root / root_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data), mode="r") as zf:
        zf.extractall(pkg_dir)
    # Add the parent of the package dir so ``import root_name`` works
    str_path = str(tmp_root)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
    return pkg_dir

# ============================================================================
# Payload tags / compact schemas
# ============================================================================

# ---------------------------------------------------------------------------
# Generic small tags
# ---------------------------------------------------------------------------

# object state payload: (tag, value)
# - ("d", mapping) => default extracted object state
# - ("c", value)   => custom __getstate__() result
_STATE_DEFAULT = "d"
_STATE_CUSTOM = "c"

# annotation payload: (tag, value)
# - ("v", bytes) => nested serialized annotation object
# - ("r", str)   => repr(annotation)
_ANN_VALUE = "v"
_ANN_REPR = "r"

# function payload kind
_FN_REF = 0
_FN_FULL = 1

# dataclass class payload kind
_DC_CLASS_REF = 0
_DC_CLASS_LOCAL = 1


# ---------------------------------------------------------------------------
# Dataclass flags
# ---------------------------------------------------------------------------

# Dataclass class flags
_DC_REPR = 1 << 0
_DC_EQ = 1 << 1
_DC_ORDER = 1 << 2
_DC_UNSAFE_HASH = 1 << 3
_DC_FROZEN = 1 << 4
_DC_SLOTS = 1 << 5

# Dataclass field flags
_DCF_INIT = 1 << 0
_DCF_REPR = 1 << 1
_DCF_COMPARE = 1 << 2
_DCF_KW_ONLY = 1 << 3


# ---------------------------------------------------------------------------
# Tuple index helpers
# ---------------------------------------------------------------------------

# class ref payload = (module_name, qualname)
_CLASS_REF_MODULE = 0
_CLASS_REF_QUALNAME = 1

# reference function payload = (version, kind, module_name, qualname)
_FN_REF_VERSION = 0
_FN_REF_KIND = 1
_FN_REF_MODULE = 2
_FN_REF_QUALNAME = 3

# full function payload =
# (
#   version,
#   kind,
#   module_name,
#   name,
#   qualname,
#   defaults,
#   kwdefaults,
#   annotations,
#   globals_dict,
#   definition_globals_dict,
#   nonlocals_dict,
#   python_version,
#   marshal_code,
#   source_code,
# )
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

# method payload = (version, function_payload_bytes, self_payload_bytes)
_METHOD_VERSION = 0
_METHOD_FUNCTION = 1
_METHOD_SELF = 2

# exception payload = (version, exc_cls, args_tuple, state_payload)
_EXC_VERSION = 0
_EXC_CLASS = 1
_EXC_ARGS = 2
_EXC_STATE = 3

# dataclass class ref payload = (kind, module_name, qualname)
_DC_REF_KIND = 0
_DC_REF_MODULE = 1
_DC_REF_QUALNAME = 2

# dataclass local class payload =
# (
#   kind,
#   cls_name,
#   qualname,
#   module_name,
#   flags,
#   field_records,
# )
_DC_LOCAL_KIND = 0
_DC_LOCAL_NAME = 1
_DC_LOCAL_QUALNAME = 2
_DC_LOCAL_MODULE = 3
_DC_LOCAL_FLAGS = 4
_DC_LOCAL_FIELDS = 5

# dataclass field record =
# (
#   field_name,
#   annotation_payload,
#   field_flags,
#   hash_value,
#   metadata_mapping_or_none,
# )
_DCF_NAME = 0
_DCF_ANNOTATION = 1
_DCF_FLAGS = 2
_DCF_HASH = 3
_DCF_METADATA = 4

# dataclass instance payload =
# (
#   version,
#   class_payload,
#   init_values,
#   non_init_values,
#   extra_state,
# )
_DC_PAYLOAD_VERSION = 0
_DC_PAYLOAD_CLASS = 1
_DC_PAYLOAD_INIT_VALUES = 2
_DC_PAYLOAD_NON_INIT_VALUES = 3
_DC_PAYLOAD_EXTRA_STATE = 4


# ============================================================================
# Validators
# ============================================================================

def _require_dict(obj: object, *, name: str) -> dict[object, object]:
    if not isinstance(obj, dict):
        raise TypeError(f"{name} must be dict")
    return obj


def _require_tuple(obj: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(obj, tuple):
        raise TypeError(f"{name} must be tuple")
    return obj


def _require_list(obj: object, *, name: str) -> list[object]:
    if not isinstance(obj, list):
        raise TypeError(f"{name} must be list")
    return obj


def _require_str(obj: object, *, name: str) -> str:
    if not isinstance(obj, str):
        raise TypeError(f"{name} must be str")
    return obj


def _require_bytes(obj: object, *, name: str) -> bytes:
    if not isinstance(obj, (bytes, bytearray)):
        raise TypeError(f"{name} must be bytes")
    return bytes(obj)


def _require_tuple_len(
    obj: object,
    *,
    name: str,
    expected: int,
) -> tuple[object, ...]:
    out = _require_tuple(obj, name=name)
    if len(out) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(out)}")
    return out


# ============================================================================
# Generic helpers
# ============================================================================

def _field_has_explicit_default(f) -> bool:
    return getattr(f, "default", MISSING) is not MISSING


def _field_value_equals_default(f, value: object) -> bool:
    """
    Only compare against explicit default.

    We intentionally do NOT compare against default_factory, because:
    - calling factories during serialization is gross
    - comparing factory-produced mutables is ambiguous
    - it can introduce side effects or fake equality wins
    """
    if not _field_has_explicit_default(f):
        return False

    try:
        return value == f.default
    except Exception:
        return False


def _resolve_qualname(root: object, qualname: str) -> object:
    """
    Resolve a dotted __qualname__ path from a root object.

    This intentionally rejects '<locals>' because local objects are not
    import-resolvable.
    """
    obj = root
    for part in qualname.split("."):
        if part == "<locals>":
            raise AttributeError("Cannot resolve local qualname segment '<locals>'")
        obj = getattr(obj, part)
    return obj


def _make_cell(value: object):
    """
    Build a closure cell containing value.

    Python does not expose a public cell constructor, so this is the classic
    tiny cursed trick.
    """
    return (lambda x: lambda: x)(value).__closure__[0]


def _serialize_nested(obj: object) -> bytes:
    """
    Serialize a nested object through the existing Serialized framework.

    This keeps nested payloads compatible with the rest of the serializer stack.
    """
    try:
        return Serialized.from_python_object(obj).write_to().to_bytes()
    except AttributeError:
        raise TypeError(
            f"Object of type {type(obj).__name__} is not serializable as a nested payload"
        )


def _deserialize_nested(blob: bytes) -> object:
    return Serialized.read_from(BytesIO(blob), pos=0).as_python()


def _iter_slots(cls: type[object]) -> tuple[str, ...]:
    """
    Yield effective instance slot names across the MRO, excluding __dict__ and
    __weakref__.
    """
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
    """
    Extract default instance state:

    - items in __dict__
    - slot values that are present
    """
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
    """
    Return the first directly-declared attribute in the MRO, or None.
    """
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


# ============================================================================
# Object state payloads
# ============================================================================

def _dump_object_state(obj: object) -> tuple[str, object]:
    """
    Compact object-state payload.

    Schema:
        (_STATE_CUSTOM, custom_state)
        (_STATE_DEFAULT, extracted_state_mapping)
    """
    if _has_meaningful_custom_getstate(obj):
        return (_STATE_CUSTOM, obj.__getstate__())

    return (_STATE_DEFAULT, _extract_object_state(obj))


def _restore_object_state(obj: object, payload: object) -> None:
    """
    Restore a state payload created by _dump_object_state().

    Rules:
    - custom state prefers __setstate__ when available
    - otherwise, dict-compatible state is applied field-by-field
    - default state always applies field-by-field
    """
    tag, value = _require_tuple_len(payload, name="Object state payload", expected=2)

    kind = _require_str(tag, name="Object state payload kind")

    if kind == _STATE_CUSTOM:
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

    if kind == _STATE_DEFAULT:
        state_obj = _require_dict(value, name="Default object state value")
        for name, item in state_obj.items():
            try:
                object.__setattr__(obj, name, item)
            except Exception:
                if hasattr(obj, "__dict__"):
                    obj.__dict__[name] = item
        return

    raise ValueError(f"Unsupported object state payload kind: {kind!r}")


# ============================================================================
# Reference policy
# ============================================================================

def _module_file_contains_site_packages(module_name: str | None) -> bool:
    """
    Heuristic: treat modules coming from site-packages as stable import references.
    """
    if not module_name:
        return False

    try:
        module = runtime_import_module(module_name, install=False)
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
    """
    Normalize:
    - function -> function
    - bound method -> underlying function
    - callable object -> __call__ function if that is a plain Python function/method
    """
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

# ============================================================================
# Caches
# ============================================================================

def _class_cache_get_or_load(module_name: str, qualname: str) -> type[object]:
    module = runtime_import_module(module_name, install=False)
    obj = _resolve_qualname(module, qualname)

    if not isinstance(obj, type):
        raise TypeError(f"Resolved object is not a class: {module_name}.{qualname}")

    return obj


# ============================================================================
# Hash helpers
# ============================================================================

def _hash_bytes(data: bytes | None) -> str | None:
    if data is None:
        return None
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _hash_text(data: str | None) -> str | None:
    if data is None:
        return None
    return hashlib.blake2b(data.encode("utf-8"), digest_size=16).hexdigest()


# ============================================================================
# Function AST / code helpers
# ============================================================================

def _dump_function_code_payload(fn: Callable[..., object]) -> tuple[
    tuple[int, int, int],
    bytes | None,
    str | None,
]:
    """
    Return the compact function code payload:

        (python_version, marshal_code, source_code)

    We try to store both:
    - marshal code: fastest path when Python version matches
    - source code: fallback path across versions / environments
    """
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

    return (_PYTHON_VERSION, marshal_code, source_code)


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
    """
    Remove decorators before exec().

    This avoids replaying import-time side effects or requiring decorator
    symbols to exist during function reconstruction.
    """
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
    """
    Extract names required to *define* a function from source, not names used
    inside the function body at runtime.

    We intentionally scan:
    - decorators
    - annotations
    - default expressions

    These must exist when exec(source) runs.
    """
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
    """
    Drop:
    - non-string keys
    - direct self-references to the function itself

    This avoids recursive payload nonsense.
    """
    out: dict[str, object] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        if value is fn:
            continue
        out[key] = value
    return out


def _load_function_from_source(
    *,
    source: str,
    globals_dict: dict[str, object],
    name: str,
) -> Callable[..., object]:
    """
    Rebuild a function by exec() of stripped source in a controlled namespace.
    """
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
    """
    Rebuild a Python function from a CodeType plus execution context.
    """
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
    """
    Reconstruct a function using two strategies:

    1. marshal => preferred when Python version matches
    2. source  => safer cross-version fallback

    If one fails, the other gets a shot. Tiny resurrection machine.
    """
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


# ============================================================================
# Class reference payload
# ============================================================================

def _dump_class_ref(cls: type[object]) -> bytes:
    """
    Class reference payload schema:
        (module_name, qualname)
    """
    if "<locals>" in cls.__qualname__:
        raise TypeError(f"Cannot serialize non-importable local class reference: {cls!r}")

    return _serialize_nested((cls.__module__, cls.__qualname__))


def _load_class_ref(data: bytes) -> type[object]:
    payload = _require_tuple_len(_deserialize_nested(data), name="Class payload", expected=2)
    module_name = _require_str(payload[_CLASS_REF_MODULE], name="Class payload module")
    qualname = _require_str(payload[_CLASS_REF_QUALNAME], name="Class payload qualname")
    return _class_cache_get_or_load(module_name, qualname)


# ============================================================================
# Function reference-only payload
# ============================================================================

def _dump_reference_function_payload(fn: Callable[..., object]) -> bytes:
    """
    Reference-only function payload schema:
        (version, _FN_REF, module_name, qualname)
    """
    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)

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

    module = runtime_import_module(module_name, install=False)
    obj = _resolve_qualname(module, qualname)
    if not callable(obj):
        raise TypeError(f"Resolved reference is not callable: {module_name}.{qualname}")

    return obj


# ============================================================================
# Full function payload
# ============================================================================

def _dump_function_payload(fn: Callable[..., object]) -> bytes:
    """
    Full function payload schema:

        (
            version,
            _FN_FULL,
            module_name,
            name,
            qualname,
            defaults,
            kwdefaults,
            annotations,
            globals_dict,
            definition_globals_dict,
            nonlocals_dict,
            python_version,
            marshal_code,
            source_code,
        )

    Why keep both globals and definition_globals?
    - globals: names captured by inspect.getclosurevars()
    - definition_globals: names only needed to exec source (decorators/defaults/annotations)
    """
    if not isinstance(fn, FunctionType):
        raise TypeError(
            f"FunctionSerialized only supports Python functions, got {type(fn)!r}"
        )

    closurevars = inspect.getclosurevars(fn)
    python_version, marshal_code, source_code = _dump_function_code_payload(fn)

    definition_globals: dict[str, object] = {}

    if isinstance(source_code, str):
        try:
            needed_names = _extract_definition_global_names(
                source_code,
                fn_name=fn.__name__,
            )
        except Exception:
            needed_names = set()

        for name in needed_names:
            if name in closurevars.globals or name in closurevars.nonlocals:
                continue
            if name in fn.__globals__:
                definition_globals[name] = fn.__globals__[name]

    payload = (
        _FORMAT_VERSION,
        _FN_FULL,
        fn.__module__,
        fn.__name__,
        fn.__qualname__,
        fn.__defaults__,
        fn.__kwdefaults__,
        fn.__annotations__,
        _drop_function_self_refs(closurevars.globals, fn=fn),
        _drop_function_self_refs(definition_globals, fn=fn),
        _drop_function_self_refs(closurevars.nonlocals, fn=fn),
        python_version,
        marshal_code,
        source_code,
    )

    return _serialize_nested(payload)


def _load_function_payload(data: bytes) -> Callable[..., object]:
    """
    Load either a reference-only or full function payload.
    """
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

    defaults = payload[_FN_FULL_DEFAULTS]
    kwdefaults = payload[_FN_FULL_KWDEFAULTS]
    annotations = payload[_FN_FULL_ANNOTATIONS]

    globals_obj = _require_dict(payload[_FN_FULL_GLOBALS], name="Function payload globals")
    definition_globals_obj = _require_dict(
        payload[_FN_FULL_DEFINITION_GLOBALS],
        name="Function payload definition_globals",
    )
    nonlocals_obj = _require_dict(payload[_FN_FULL_NONLOCALS], name="Function payload nonlocals")

    python_version = payload[_FN_FULL_PY_VERSION]
    if not isinstance(python_version, (tuple, type(None))):
        raise TypeError("Function payload python_version must be tuple | None")

    marshal_code = payload[_FN_FULL_MARSHAL]
    source_code = payload[_FN_FULL_SOURCE]

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

    # Restore self-reference in globals for recursive functions.
    fn.__globals__[name] = fn
    return fn


# ============================================================================
# Method payload
# ============================================================================

def _dump_method_payload(method: MethodType) -> bytes:
    """
    Method payload schema:
        (version, function_payload_bytes, self_payload_bytes)

    Notes:
    - function payload is the same payload FunctionSerialized uses
    - self payload is the bound receiver (instance or class)
    """
    if not isinstance(method, MethodType):
        raise TypeError(f"MethodSerialized requires MethodType, got {type(method)!r}")

    fn = method.__func__
    self_obj = method.__self__

    if _should_use_reference_only_for_callable(fn):
        fn_payload = _dump_reference_function_payload(fn)
    else:
        fn_payload = _dump_function_payload(fn)

    self_payload = _serialize_nested(self_obj)

    return _serialize_nested(
        (
            _FORMAT_VERSION,
            fn_payload,
            self_payload,
        )
    )


def _load_method_payload(data: bytes) -> MethodType:
    payload = _require_tuple_len(_deserialize_nested(data), name="Method payload", expected=3)

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


# ============================================================================
# Exception payload
# ============================================================================

def _dump_exception_payload(exc: BaseException) -> bytes:
    """
    Exception payload schema:
        (version, exc_cls, args_tuple, state_payload)
    """
    return _serialize_nested(
        (
            _FORMAT_VERSION,
            type(exc),
            exc.args,
            _dump_object_state(exc),
        )
    )


def _load_exception_payload(data: bytes) -> BaseException:
    payload = _require_tuple_len(_deserialize_nested(data), name="Exception payload", expected=4)

    version = payload[_EXC_VERSION]
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported exception payload version: {version!r}")

    exc_cls_obj = payload[_EXC_CLASS]
    if not isinstance(exc_cls_obj, type) or not issubclass(exc_cls_obj, BaseException):
        raise TypeError("Decoded exception class is not a BaseException subclass")

    args_obj = _require_tuple(payload[_EXC_ARGS], name="Decoded exception args")
    state_payload = payload[_EXC_STATE]

    try:
        exc = exc_cls_obj(*args_obj)
    except Exception:
        exc = BaseException.__new__(exc_cls_obj)
        exc.args = args_obj

    _restore_object_state(exc, state_payload)
    return exc


# ============================================================================
# Dataclass annotation payload
# ============================================================================

def _safe_dump_annotation(annotation: object) -> tuple[str, object]:
    """
    Compact annotation payload.

    Schema:
        (_ANN_VALUE, nested_serialized_bytes)
        (_ANN_REPR, repr_string)
    """
    try:
        return (_ANN_VALUE, _serialize_nested(annotation))
    except Exception:
        return (_ANN_REPR, repr(annotation))


def _safe_load_annotation(payload: object) -> object:
    tag, value = _require_tuple_len(payload, name="Annotation payload", expected=2)
    kind = _require_str(tag, name="Annotation payload kind")

    if kind == _ANN_VALUE:
        blob = _require_bytes(value, name="Annotation payload value")
        return _deserialize_nested(blob)

    if kind == _ANN_REPR:
        return _require_str(value, name="Annotation payload value")

    raise ValueError(f"Unsupported annotation payload kind: {kind!r}")


# ============================================================================
# Dataclass class payload
# ============================================================================

def _dataclass_param_flags(cls: type[object]) -> int:
    """
    Pack dataclass class parameters into a single int.
    """
    p = cls.__dataclass_params__
    flags = 0

    if p.repr:
        flags |= _DC_REPR
    if p.eq:
        flags |= _DC_EQ
    if p.order:
        flags |= _DC_ORDER
    if p.unsafe_hash:
        flags |= _DC_UNSAFE_HASH
    if p.frozen:
        flags |= _DC_FROZEN
    if "__slots__" in cls.__dict__:
        flags |= _DC_SLOTS

    return flags


def _field_flags(f) -> int:
    """
    Pack dataclass field booleans into a single int.
    """
    flags = 0
    if f.init:
        flags |= _DCF_INIT
    if f.repr:
        flags |= _DCF_REPR
    if f.compare:
        flags |= _DCF_COMPARE
    if getattr(f, "kw_only", False):
        flags |= _DCF_KW_ONLY
    return flags


def _flag_on(flags: int, bit: int) -> bool:
    return bool(flags & bit)


def _dump_dataclass_class_payload(cls: type[object]) -> tuple[object, ...]:
    """
    Dataclass class payload schemas:

    Importable dataclass:
        (_DC_CLASS_REF, module_name, qualname)

    Local/non-importable dataclass:
        (
            _DC_CLASS_LOCAL,
            cls_name,
            qualname,
            module_name,
            class_flags,
            [
                {
                    "name": field_name,
                    "annotation": annotation_payload,
                    "flags": field_flags,
                    "hash": hash_value,
                    "metadata": metadata_or_none,
                },
                ...
            ],
        )
    """
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type, got {cls!r}")

    if _is_importable_class(cls):
        return (
            _DC_CLASS_REF,
            cls.__module__,
            cls.__qualname__,
        )

    field_payloads: list[dict[str, object]] = []

    for f in fields(cls):
        field_payloads.append(
            {
                "name": f.name,
                "annotation": _safe_dump_annotation(f.type),
                "flags": _field_flags(f),
                "hash": f.hash,
                "metadata": dict(f.metadata) if f.metadata else None,
            }
        )

    return (
        _DC_CLASS_LOCAL,
        cls.__name__,
        cls.__qualname__,
        cls.__module__,
        _dataclass_param_flags(cls),
        field_payloads,
    )


def _load_dataclass_class_payload(payload: object) -> type[object]:
    data = _require_tuple(payload, name="Dataclass class payload")
    if not data:
        raise ValueError("Dataclass class payload must not be empty")

    kind = data[0]

    if kind == _DC_CLASS_REF:
        data = _require_tuple_len(data, name="Dataclass class ref payload", expected=3)
        module_name = _require_str(data[_DC_REF_MODULE], name="Dataclass class payload module")
        qualname = _require_str(data[_DC_REF_QUALNAME], name="Dataclass class payload qualname")
        cls = _class_cache_get_or_load(module_name, qualname)
        if not is_dataclass(cls):
            raise TypeError(f"Resolved class is not a dataclass: {module_name}.{qualname}")
        return cls

    if kind != _DC_CLASS_LOCAL:
        raise ValueError(f"Unsupported dataclass class payload kind: {kind!r}")

    data = _require_tuple_len(data, name="Dataclass local class payload", expected=6)

    name = _require_str(data[_DC_LOCAL_NAME], name="Dataclass local payload name")
    qualname = _require_str(data[_DC_LOCAL_QUALNAME], name="Dataclass local payload qualname")
    module_name = _require_str(data[_DC_LOCAL_MODULE], name="Dataclass local payload module")
    flags = data[_DC_LOCAL_FLAGS]
    if not isinstance(flags, int):
        raise TypeError("Dataclass local payload flags must be int")

    fields_payload = _require_list(data[_DC_LOCAL_FIELDS], name="Dataclass local payload fields")

    spec = []
    for item in fields_payload:
        if isinstance(item, dict):
            field_name = _require_str(item.get("name"), name="Dataclass field name")
            annotation = _safe_load_annotation(item.get("annotation"))
            field_flags_value = item.get("flags")
            hash_flag = item.get("hash")
            metadata_obj = item.get("metadata")
        else:
            field_data = _require_tuple_len(item, name="Dataclass field payload", expected=5)
            field_name = _require_str(field_data[_DCF_NAME], name="Dataclass field name")
            annotation = _safe_load_annotation(field_data[_DCF_ANNOTATION])
            field_flags_value = field_data[_DCF_FLAGS]
            hash_flag = field_data[_DCF_HASH]
            metadata_obj = field_data[_DCF_METADATA]

        if not isinstance(field_flags_value, int):
            raise TypeError("Dataclass field flags must be int")

        if metadata_obj is None:
            metadata = {}
        else:
            metadata = _require_dict(metadata_obj, name="Dataclass field metadata")

        fld = dataclass_field(
            init=_flag_on(field_flags_value, _DCF_INIT),
            repr=_flag_on(field_flags_value, _DCF_REPR),
            hash=hash_flag,
            compare=_flag_on(field_flags_value, _DCF_COMPARE),
            kw_only=_flag_on(field_flags_value, _DCF_KW_ONLY),
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
        repr=_flag_on(flags, _DC_REPR),
        eq=_flag_on(flags, _DC_EQ),
        order=_flag_on(flags, _DC_ORDER),
        unsafe_hash=_flag_on(flags, _DC_UNSAFE_HASH),
        frozen=_flag_on(flags, _DC_FROZEN),
        slots=_flag_on(flags, _DC_SLOTS),
    )

    return cls


# ============================================================================
# Dataclass instance payload
# ============================================================================

def _dump_dataclass_payload(obj: object) -> bytes:
    """
    Dataclass instance payload schema:
        (
            version,
            class_payload,
            init_values_dict,
            non_init_values_dict,
            extra_state_payload,
        )

    Notes:
    - field values equal to explicit default are omitted
    - default_factory is intentionally ignored for filtering
    """
    try:
        obj_fields = fields(obj)
    except Exception:
        if not is_dataclass(obj) or isinstance(obj, type):
            raise TypeError(f"DataclassSerialized requires a dataclass instance, got {type(obj)!r}")
        raise

    cls = type(obj)
    init_values: dict[str, object] = {}
    non_init_values: dict[str, object] = {}

    field_names = {f.name for f in obj_fields}

    for f in obj_fields:
        value = getattr(obj, f.name)

        # Only filter against explicit default; skip default_factory checks.
        if _field_value_equals_default(f, value):
            continue
        elif not f.init:
            continue

        if f.init:
            init_values[f.name] = value
        else:
            non_init_values[f.name] = value

    raw_state = _dump_object_state(obj)

    if raw_state[0] == _STATE_DEFAULT:
        default_state = _require_dict(raw_state[1], name="Dataclass default state")
        extra_state_payload: tuple[str, object] = (
            _STATE_DEFAULT,
            {
                key: value
                for key, value in default_state.items()
                if key not in field_names
            },
        )
    else:
        extra_state_payload = raw_state

    payload = (
        _FORMAT_VERSION,
        _dump_dataclass_class_payload(cls),
        init_values,
        non_init_values,
        extra_state_payload,
    )
    return _serialize_nested(payload)


def _load_dataclass_payload(data: bytes) -> object:
    payload = _require_tuple_len(_deserialize_nested(data), name="Dataclass payload", expected=5)

    version = payload[_DC_PAYLOAD_VERSION]
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported dataclass payload version: {version!r}")

    class_payload = payload[_DC_PAYLOAD_CLASS]
    cls_obj = _load_dataclass_class_payload(class_payload)
    if not isinstance(cls_obj, type) or not is_dataclass(cls_obj):
        raise TypeError("Decoded dataclass class is not a dataclass type")

    init_values = _require_dict(payload[_DC_PAYLOAD_INIT_VALUES], name="Dataclass init_values")
    non_init_values = _require_dict(payload[_DC_PAYLOAD_NON_INIT_VALUES], name="Dataclass non_init_values")
    extra_state_payload = payload[_DC_PAYLOAD_EXTRA_STATE]

    try:
        obj = cls_obj(**init_values)
    except Exception:
        # resolve non-init fields first to improve chances of successful construction
        init_values = {
            key: value
            for key, value in init_values.items()
            if key in cls_obj.__dataclass_fields__ and cls_obj.__dataclass_fields__[key].init
        }
        obj = cls_obj(**init_values)

    for name, value in non_init_values.items():
        object.__setattr__(obj, name, value)

    _restore_object_state(obj, extra_state_payload)
    return obj


# ============================================================================
# Serialized wrappers
# ============================================================================

@dataclass(frozen=True, slots=True)
class ComplexSerialized(Serialized[T], Generic[T]):
    """
    Base wrapper for "complex" Python objects that need special handling beyond
    primitive / logical / collection payloads.

    Subclasses only need to implement value.
    """
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
        """
        Try to serialize supported complex object kinds.

        Order matters a bit:
        - methods
        - functions
        - dataclass instances
        - exceptions
        - classes
        - modules
        """
        if isinstance(obj, MethodType):
            return MethodSerialized.build_method(obj, codec=codec)

        if isinstance(obj, FunctionType):
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
    """
    Module serializer that handles both name-only and full-zip payloads
    using the same tag (MODULE).  The variant is distinguished by a
    metadata flag.

    Name-only payload (no metadata / ``M_MODULE_MODE`` absent):
        utf-8 encoded module name.

    Full payload (``M_MODULE_MODE == b"full"``):
        nested serialized tuple (version, root_name, module_name, zip_bytes).
    """
    TAG: ClassVar[int] = Tags.MODULE

    # Metadata key that marks a full-zip module payload.
    M_MODULE_MODE: ClassVar[bytes] = b"module_mode"
    _MODE_FULL: ClassVar[bytes] = b"full"

    @property
    def value(self) -> ModuleType:
        meta = self.metadata
        if meta and meta.get(self.M_MODULE_MODE) == self._MODE_FULL:
            return self._load_full()
        return self._load_name_only()

    def _load_name_only(self) -> ModuleType:
        module_name = self.decode().decode("utf-8")
        # Whitelisted modules are expected to be available (or installable) on
        # the target runtime.  Non-whitelisted modules serialised as name-only
        # fell back from the full-zip path, so we don't attempt an install.
        install = _is_whitelisted_module(module_name)
        return runtime_import_module(module_name, install=install)

    def _load_full(self) -> ModuleType:
        payload = _require_tuple_len(
            _deserialize_nested(self.decode()),
            name="Full module payload",
            expected=4,
        )
        version = payload[0]
        if version != _FORMAT_VERSION:
            raise ValueError(f"Unsupported full module payload version: {version!r}")

        root_name = _require_str(payload[1], name="Full module payload root_name")
        module_name = _require_str(payload[2], name="Full module payload module_name")
        zip_data = _require_bytes(payload[3], name="Full module payload zip_bytes")

        # Fast path: if the current runtime already has it, use it directly.
        try:
            return importlib.import_module(module_name)
        except ImportError:
            pass

        # Extract and install into sys.path, then import.
        _extract_module_zip(zip_data, root_name)
        importlib.invalidate_caches()
        return importlib.import_module(module_name)

    @classmethod
    def build_module(
        cls,
        module: ModuleType,
        *,
        codec: int | None = None,
    ) -> Serialized[object]:
        module_name = module.__name__
        root_name = module_name.split(".", 1)[0]

        # Whitelisted modules (stdlib, well-known packages) → name-only reference
        if _is_whitelisted_module(root_name):
            return cls.build(
                tag=cls.TAG,
                data=module_name.encode("utf-8"),
                codec=codec,
            )

        # Try full-module serialisation for non-whitelisted modules
        root_path = _get_module_root_path(module)
        if root_path is not None and root_path.exists():
            try:
                total = _module_dir_filtered_bytes(root_path)
            except Exception:
                total = _MAX_MODULE_INLINE_BYTES + 1

            if total <= _MAX_MODULE_INLINE_BYTES:
                try:
                    zip_bytes = _get_or_zip_module(root_name, root_path)
                    payload = _serialize_nested((
                        _FORMAT_VERSION,
                        root_name,
                        module_name,
                        zip_bytes,
                    ))
                    return cls.build(
                        tag=cls.TAG,
                        data=payload,
                        metadata={cls.M_MODULE_MODE: cls._MODE_FULL},
                        codec=codec,
                    )
                except Exception:
                    _LOGGER.debug(
                        "Failed to zip-serialize module '%s'; falling back to name-only",
                        module_name,
                        exc_info=True,
                    )

        # Fallback: name-only reference
        return cls.build(
            tag=cls.TAG,
            data=module_name.encode("utf-8"),
            codec=codec,
        )



@dataclass(frozen=True, slots=True)
class ClassSerialized(ComplexSerialized[type[object]]):
    """
    Class payload is a nested compact class reference tuple.
    """
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
    """
    Function payload uses either:
    - reference-only tuple
    - full reconstruction tuple

    This serializer is for Python functions only.
    Bound methods use MethodSerialized.
    """
    TAG: ClassVar[int] = Tags.FUNCTION

    @property
    def value(self) -> Callable[..., object]:
        return _load_function_payload(self.decode())

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
    """
    Method payload stores:
    - serialized function payload
    - serialized bound receiver (__self__)

    This preserves bound instance/class semantics explicitly.
    """
    TAG: ClassVar[int] = Tags.METHOD

    @property
    def value(self) -> MethodType:
        return _load_method_payload(self.decode())

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


@dataclass(frozen=True, slots=True)
class BaseExceptionSerialized(ComplexSerialized[BaseException]):
    """
    Exception payload stores:
    - class
    - args
    - instance state
    """
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
    """
    Dataclass payload stores:
    - class schema/reference
    - init values
    - non-init values
    - extra state
    """
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


# Register all complex serializer subclasses with Tags
for cls in ComplexSerialized.__subclasses__():
    Tags.register_class(cls)


for t, cls in (
    (ModuleType, ModuleSerialized),
    (FunctionType, FunctionSerialized),
    (MethodType, MethodSerialized),
    (BaseException, BaseExceptionSerialized),
):
    Tags.register_class(cls, pytype=t)