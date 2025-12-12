# yggdrasil.ser

Serialization helpers and dependency inspection utilities.

## When to use
- You need to introspect optional dependencies and communicate missing modules clearly.
- You want to extract and serialize function source code (including nested functions) for reproducibility.

## Core types
- `DependencyInfo` – captures a dependency's root module, submodule, and detected root path.
- `DependencyCheckResult` – outcome of an importability check for a dependency.

## Helper functions
- `_find_package_root_from_file(module_file)` — walk upward from a module `__file__` to locate the top-level package directory.
- `_extract_function_source(raw_src, qualname, func_name)` — best-effort extraction of a specific function's source (supports nested functions) from inspected source text.
- `_dedent_if_needed(src)` — normalize indentation for serialized snippets.

## Notes
- Designed to support reflection features used by other modules (e.g., dependency guards).
- When extracting sources, ensure the original objects are defined in importable modules; dynamically generated functions may not round-trip well.

## Related modules
- [yggdrasil.libs](../libs/README.md) for dependency guards and auto-install utilities.
