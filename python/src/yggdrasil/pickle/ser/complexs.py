"""
Backward-compatible re-export façade for the complex serialization layer.

The implementation has been split into three focused modules:
    libs.py        — shared utilities and base serializer classes
    callables.py   — FunctionSerialized / MethodSerialized + helpers
    dataclasses.py — DataclassSerialized + helpers

Everything that was previously in this file is re-exported here so that all
existing import paths (``from yggdrasil.pickle.ser.complexs import …``) keep
working without modification.

``import inspect`` is kept at module level so that test code that does::

    import yggdrasil.pickle.ser.complexs as complexs_module
    monkeypatch.setattr(complexs_module.inspect, "getclosurevars", …)

continues to work — patching ``inspect.getclosurevars`` through this module
attribute affects the shared ``inspect`` module object used by callables.py.
"""

from __future__ import annotations

import inspect  # noqa: F401 — kept for test monkeypatching; see module docstring
from types import FunctionType, MethodType, ModuleType

# ---------------------------------------------------------------------------
# re-exports from libs.py
# ---------------------------------------------------------------------------

from yggdrasil.pickle.ser.libs import (
    # public
    BaseExceptionSerialized,
    ClassSerialized,
    ComplexSerialized,
    ModuleSerialized,
    # private constants
    _ANN_REPR,
    _ANN_VALUE,
    _BUILTINS_KEY,
    _CLASS_CACHE,
    _CLASS_REF_MODULE,
    _CLASS_REF_QUALNAME,
    _EXC_ARGS,
    _EXC_CLASS,
    _EXC_STATE,
    _EXC_VERSION,
    _FORMAT_VERSION,
    _MODULE_CACHE,
    _PYTHON_VERSION,
    _STATE_CUSTOM,
    _STATE_DEFAULT,
    # private helpers
    _class_cache_get_or_load,
    _deserialize_nested,
    _dump_class_ref,
    _dump_exception_payload,
    _dump_object_state,
    _extract_object_state,
    _get_declared_attr,
    _has_meaningful_custom_getstate,
    _has_meaningful_custom_setstate,
    _hash_bytes,
    _hash_text,
    _is_importable_class,
    _iter_slots,
    _load_class_ref,
    _load_exception_payload,
    _make_cell,
    _module_cache_get_or_load,
    _module_file_contains_site_packages,
    _require_bytes,
    _require_dict,
    _require_list,
    _require_str,
    _require_tuple,
    _require_tuple_len,
    _resolve_qualname,
    _restore_object_state,
    _safe_dump_annotation,
    _safe_load_annotation,
    _serialize_nested,
    _should_reference_only_module,
    _should_use_reference_only_for_class,
)

# ---------------------------------------------------------------------------
# re-exports from callables.py
# ---------------------------------------------------------------------------

from yggdrasil.pickle.ser.callables import (
    # public
    FunctionSerialized,
    MethodSerialized,
    # private caches
    _FULL_FUNCTION_CACHE,
    _MODULE_IMPORT_BINDINGS_CACHE,
    _REFERENCE_FUNCTION_CACHE,
    # private constants
    _FN_FULL,
    _FN_FULL_ANNOTATIONS,
    _FN_FULL_DEFAULTS,
    _FN_FULL_DEFINITION_GLOBALS,
    _FN_FULL_GLOBALS,
    _FN_FULL_KIND,
    _FN_FULL_KWDEFAULTS,
    _FN_FULL_MARSHAL,
    _FN_FULL_MODULE,
    _FN_FULL_NAME,
    _FN_FULL_NONLOCALS,
    _FN_FULL_PY_VERSION,
    _FN_FULL_QUALNAME,
    _FN_FULL_SOURCE,
    _FN_FULL_VERSION,
    _FN_REF,
    _FN_REF_KIND,
    _FN_REF_MODULE,
    _FN_REF_QUALNAME,
    _FN_REF_VERSION,
    _METHOD_FUNCTION,
    _METHOD_SELF,
    _METHOD_VERSION,
    # private helpers
    _ModuleImportBindingCollector,
    _build_function_from_code,
    _collect_inner_function_definition_globals,
    _collect_load_names,
    _collect_outer_function_runtime_context,
    _collect_referenced_global_names_from_symbol_table,
    _drop_function_self_refs,
    _dump_function_code_payload,
    _dump_function_payload,
    _dump_method_payload,
    _dump_reference_function_payload,
    _extract_definition_global_names,
    _extract_used_names_from_function_source,
    _find_function_node,
    _find_symbol_table_by_name,
    _infer_imported_globals_from_source_module,
    _iter_annotation_nodes,
    _iter_default_nodes,
    _load_function_code_payload,
    _load_function_from_source,
    _load_function_payload,
    _load_method_payload,
    _load_module_source_text,
    _load_reference_function_payload,
    _module_import_bindings_get_or_load,
    _root_name,
    _safe_getclosurevars,
    _should_use_reference_only_for_callable,
    _strip_function_decorators_from_source,
    _unwrap_method_or_function,
    _unwrap_py_function,
)

# ---------------------------------------------------------------------------
# re-exports from dataclasses.py
# ---------------------------------------------------------------------------

from yggdrasil.pickle.ser.dataclasses import (
    # public
    DataclassSerialized,
    # private cache
    _LOCAL_DATACLASS_CACHE,
    # private constants
    _DC_CLASS_LOCAL,
    _DC_CLASS_REF,
    _DC_EQ,
    _DC_FROZEN,
    _DC_LOCAL_FIELDS,
    _DC_LOCAL_FLAGS,
    _DC_LOCAL_KIND,
    _DC_LOCAL_MODULE,
    _DC_LOCAL_NAME,
    _DC_LOCAL_QUALNAME,
    _DC_ORDER,
    _DC_PAYLOAD_CLASS,
    _DC_PAYLOAD_EXTRA_STATE,
    _DC_PAYLOAD_INIT_VALUES,
    _DC_PAYLOAD_NON_INIT_VALUES,
    _DC_PAYLOAD_VERSION,
    _DC_REF_KIND,
    _DC_REF_MODULE,
    _DC_REF_QUALNAME,
    _DC_REPR,
    _DC_SLOTS,
    _DC_UNSAFE_HASH,
    _DCF_COMPARE,
    _DCF_HASH,
    _DCF_INIT,
    _DCF_KW_ONLY,
    _DCF_METADATA,
    _DCF_NAME,
    _DCF_REPR,
    _DCF_ANNOTATION,
    _DCF_FLAGS,
    # private helpers
    _dataclass_param_flags,
    _dump_dataclass_class_payload,
    _dump_dataclass_payload,
    _field_flags,
    _field_has_explicit_default,
    _field_value_equals_default,
    _flag_on,
    _load_dataclass_class_payload,
    _load_dataclass_payload,
)

# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

__all__ = [
    "ComplexSerialized",
    "ModuleSerialized",
    "ClassSerialized",
    "FunctionSerialized",
    "MethodSerialized",
    "BaseExceptionSerialized",
    "DataclassSerialized",
]

# ---------------------------------------------------------------------------
# Tags registration
# ---------------------------------------------------------------------------

from yggdrasil.pickle.ser.tags import Tags  # noqa: E402 — must come after class definitions

for _cls in ComplexSerialized.__subclasses__():
    Tags.register_class(_cls)

for _pytype, _cls in (
    (ModuleType, ModuleSerialized),
    (FunctionType, FunctionSerialized),
    (MethodType, MethodSerialized),
    (BaseException, BaseExceptionSerialized),
):
    Tags.register_class(_cls, pytype=_pytype)

del _cls, _pytype

