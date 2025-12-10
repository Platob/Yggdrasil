# yggdrasil.ser

Serialization helpers and dependency inspection utilities.

## `DependencyInfo` / `DependencyCheckResult`
Dataclasses representing a dependency (root module, submodule, detected root path) and the outcome of an importability check.

## Helpers
- `_find_package_root_from_file(module_file)` — walk upward from a module `__file__` to locate the top-level package directory.
- `_extract_function_source(raw_src, qualname, func_name)` — best-effort extraction of a specific function's source (supports nested functions) from inspected source text.
- `_dedent_if_needed(src)` — normalize indentation for serialized snippets.

These utilities support reflecting over functions, validating optional dependencies, and generating reproducible serialized code segments.

## Navigation
- [Module overview](../../modules.md)
- [Dataclasses](../dataclasses/README.md)
- [Libs](../libs/README.md)
- [Requests](../requests/README.md)
- [Types](../types/README.md)
- [Databricks](../databricks/README.md)
- [Pyutils](../pyutils/README.md)
- [Ser](./README.md)
