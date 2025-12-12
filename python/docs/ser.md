# yggdrasil.ser

Serialization helpers and dependency inspection utilities.

## Core data structures
- `DependencyInfo` – captures a dependency's root module, submodule, and detected root path.
- `DependencyCheckResult` – outcome of an importability check for a dependency.

## Helper functions
- `_find_package_root_from_file(module_file)` — walk upward from a module `__file__` to locate the top-level package directory.
- `_extract_function_source(raw_src, qualname, func_name)` — best-effort extraction of a specific function's source (supports nested functions) from inspected source text.
- `_dedent_if_needed(src)` — normalize indentation for serialized snippets.

Use these utilities to reflect over functions, validate optional dependencies, and generate reproducible serialized code segments.
