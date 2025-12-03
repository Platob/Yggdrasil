from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List, Mapping


# ---------- small data structs -------------------------------------------------


@dataclass(frozen=True)
class DependencyInfo:
    """
    Single dependency entry.

    Example:
        DependencyInfo(
            root_module="databricks",
            submodule="databricks.sdk",
            root_path="/path/to/databricks"
        )
    """

    root_module: str
    submodule: str
    root_path: Optional[str]


@dataclass(frozen=True)
class DependencyCheckResult:
    """
    Result of checking whether a dependency is importable in the current env.
    """

    root_module: str
    root_path: Optional[str]
    importable: bool
    runtime_file: Optional[str]
    error: Optional[str]


# ---------- helpers ------------------------------------------------------------


def _find_package_root_from_file(module_file: str) -> str:
    """
    Given a module __file__, find the top-most package directory.

    Walk upwards from the directory containing the file as long
    as there is an __init__.py present. Return that top-most dir.
    """
    path = os.path.abspath(module_file)
    current_dir = os.path.dirname(path)

    last_pkg_dir: Optional[str] = None

    # If the current dir itself is a package, mark it
    if os.path.isfile(os.path.join(current_dir, "__init__.py")):
        last_pkg_dir = current_dir

    # Walk upwards while parent is also a package
    while True:
        parent = os.path.dirname(current_dir)
        if parent == current_dir:  # FS root
            break

        init_path = os.path.join(parent, "__init__.py")
        if os.path.isfile(init_path):
            last_pkg_dir = parent
            current_dir = parent
        else:
            break

    return last_pkg_dir or os.path.dirname(path)


def _extract_function_source(
    raw_src: str,
    qualname: str,
    func_name: str,
) -> str:
    """
    Given raw source returned by inspect.getsource(func), try to extract just the
    function definition for the target function.

    Supports:
      - top-level functions: qualname == name
      - nested/local functions: qualname like "outer.<locals>.inner"

    For nested funcs, we *try* to walk the AST and use ast.get_source_segment()
    to pull the exact code for the target node. If that fails (e.g. because
    raw_src is just an indented inner function and ast.parse blows up with
    IndentationError), we fall back to simply dedenting the whole block.
    """
    # strip out "<locals>" segments from qualname
    parts = [p for p in qualname.split(".") if p != "<locals>"]

    # Simple case: top-level function
    if len(parts) == 1 and parts[0] == func_name:
        return textwrap.dedent(raw_src)

    # Nested/local function: try AST-based extraction
    try:
        tree = ast.parse(raw_src)
    except (SyntaxError, IndentationError):
        # Most likely case: raw_src is just an indented inner def, like:
        # "        def inner(x):\n            return x + 1\n"
        # Dedent and treat that as the function source.
        return textwrap.dedent(raw_src)

    target_path = parts  # e.g. ["outer", "inner"]

    def _walk(node: ast.AST, idx: int) -> Optional[ast.AST]:
        body = getattr(node, "body", [])
        for child in body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.name == target_path[idx]:
                    if idx == len(target_path) - 1:
                        return child
                    return _walk(child, idx + 1)
        return None

    found_node = _walk(tree, 0)

    if not found_node:
        # fallback: give the whole block; better than blowing up
        return textwrap.dedent(raw_src)

    segment = ast.get_source_segment(raw_src, found_node)
    if segment is None:
        return textwrap.dedent(raw_src)

    return textwrap.dedent(segment)


def _infer_raw_dependencies_and_aliases(
    func: Callable[..., Any]
) -> tuple[Dict[str, Optional[str]], Dict[str, str]]:
    """
    Infer:

      - raw_deps: { full_module_name -> original_module_file_or_None }
      - aliases:  { global_name_used_in_func -> full_module_name }

    by inspecting closure globals/nonlocals for module objects.
    Example alias:
        MATH_MOD -> "math"
        JSON_MOD -> "json"
    """
    raw_deps: Dict[str, Optional[str]] = {}
    aliases: Dict[str, str] = {}

    closure = inspect.getclosurevars(func)

    def _collect(mapping: Mapping[str, Any]) -> None:
        for name, val in mapping.items():
            if inspect.ismodule(val):
                mod_name = getattr(val, "__name__", None)
                if not mod_name:
                    continue
                mod_file = getattr(val, "__file__", None)
                if mod_name not in raw_deps or (not raw_deps[mod_name] and mod_file):
                    raw_deps[mod_name] = mod_file
                # record alias (e.g. MATH_MOD -> "math")
                aliases.setdefault(name, mod_name)

    _collect(closure.globals)
    _collect(closure.nonlocals)

    return raw_deps, aliases


def _build_dependency_infos(
    raw_deps: Mapping[str, Optional[str]],
) -> List[DependencyInfo]:
    """
    Turn raw deps ({full_module_name -> file}) into DependencyInfo objects:

        "databricks.sdk.service.compute" ->
            DependencyInfo(
                root_module="databricks",
                submodule="databricks.sdk",
                root_path="/path/to/databricks"
            )
    """
    infos: List[DependencyInfo] = []
    seen: set[tuple[str, str]] = set()

    for full_name in raw_deps.keys():
        parts = full_name.split(".")
        root = parts[0]

        if len(parts) > 1:
            sub = ".".join(parts[:2])
        else:
            sub = root

        key = (root, sub)
        if key in seen:
            continue
        seen.add(key)

        root_path: Optional[str] = None
        try:
            root_mod = importlib.import_module(root)
            root_file = getattr(root_mod, "__file__", None)
            if root_file:
                root_path = os.path.dirname(os.path.abspath(root_file))
        except Exception:
            root_path = None

        infos.append(DependencyInfo(root_module=root, submodule=sub, root_path=root_path))

    return infos


# ---------- main class --------------------------------------------------------


@dataclass
class EmbeddedFunction:
    """
    Function + its source code embedded as text.

    - Can be serialized (pickle/dill/msgpack/etc.)
    - Can be rebuilt on another Python process via exec()
    - Is directly callable: embedded_fn(*args, **kwargs)
    - Tracks dependency infos:
        DependencyInfo(root_module, submodule, root_path)
    - Captures the package root of the defining module (if resolvable)
    - Supports top-level and local/nested functions, as long as they do NOT
      capture nonlocal variables in their closure.
    """

    name: str
    source: str
    source_file_content: Optional[str]

    # List of DependencyInfo entries
    dependencies_map: List[DependencyInfo] = field(default_factory=list)

    # Global-name -> full module name alias mapping (e.g. "MATH_MOD" -> "math")
    aliases: Dict[str, str] = field(default_factory=dict)

    # Filesystem path to the top-most package containing the defining module
    package_root: Optional[str] = None

    # internal cache for the rebuilt function
    _compiled: Callable[..., Any] | None = field(
        default=None, init=False, repr=False
    )

    # ---------- constructors ----------
    @classmethod
    def _infer_imported_modules_from_file(
        cls,
        module_file: str,
        src: Optional[str] = None,
        package_root: Optional[str] = None,
    ) -> List[DependencyInfo]:
        """
        Given a Python source file, parse its imports and return DependencyInfo
        objects for modules that look relevant.

        Returns:
            List[DependencyInfo]
        """
        if not src:
            with open(module_file, "r", encoding="utf-8") as f:
                src = f.read()

        tree = ast.parse(src, filename=module_file)

        imported_module_names: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # import x, import x.y.z
                for alias in node.names:
                    imported_module_names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                # from x import y; from x.y import z; from .x import y
                if node.module is None:
                    # "from . import foo" -> skip unless you want to get fancy
                    continue
                full = ("." * node.level + node.module) if node.level else node.module
                imported_module_names.add(full)

        # Resolve names → real modules, filter by package_root if given
        resolved_names: set[str] = set()

        for name in imported_module_names:
            try:
                # Handle relative names relative to this module's package
                if name.startswith("."):
                    if not package_root:
                        continue
                    rel_path = os.path.relpath(module_file, package_root)
                    # e.g. my_pkg/mod.py -> my_pkg.mod
                    pkg = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
                    pkg = pkg.rsplit(".", 1)[0]  # strip final segment
                    full_name = pkg + name  # let importlib normalize it
                    spec = importlib.util.find_spec(full_name)
                else:
                    spec = importlib.util.find_spec(name)

                if spec is None or spec.origin is None:
                    continue

                resolved_names.add(spec.name)
            except Exception:
                # noisy/edge imports shouldn't break everything
                continue

        # Now build DependencyInfo list reusing your existing helper
        raw_deps = {full_name: None for full_name in resolved_names}
        return _build_dependency_infos(raw_deps)

    @classmethod
    def from_callable(cls, func: Callable[..., Any]) -> "EmbeddedFunction":
        """
        Build an EmbeddedFunction from a function.

        Supports:
          - top-level functions
          - local/nested functions (qualname may contain '<locals>')

        Rejects:
          - bound methods (inspect.ismethod)
          - functions that capture nonlocal variables in their closure
        """
        # 1) Reject bound methods
        if inspect.ismethod(func):
            raise ValueError("EmbeddedFunction does not support bound methods")

        # 2) Qualname is how we identify nested/local defs
        qualname = getattr(func, "__qualname__", None)
        if not qualname:
            raise ValueError(f"Cannot derive qualname for {func!r}")

        # 3) Reject functions that close over nonlocal variables
        closure = inspect.getclosurevars(func)
        if closure.nonlocals:
            raise ValueError(
                "EmbeddedFunction does not support local functions that capture "
                "nonlocal variables in their closure"
            )

        # 4) Extract raw source for the function
        try:
            raw_src = inspect.getsource(func)
        except OSError as exc:
            raise ValueError(f"Cannot extract source for {func!r}: {exc}") from exc

        # 5) Trim down to just the def block for this specific function
        src = _extract_function_source(raw_src, qualname, func.__name__)

        # 6) Infer raw deps (module_name -> file_or_None) + aliases
        raw_deps, aliases = _infer_raw_dependencies_and_aliases(func)

        # 7) Add the defining module; track package_root
        module_name = getattr(func, "__module__", None)
        package_root: Optional[str] = None
        module_file: Optional[str] = None
        source_file_content: Optional[str] = None

        if module_name:
            mod = sys.modules.get(module_name)
            if mod is not None:
                module_file = getattr(mod, "__file__", None)
                if module_file:
                    module_file = os.path.abspath(module_file)
                    package_root = _find_package_root_from_file(module_file)
                    # Always include the defining module itself
                    raw_deps.setdefault(module_name, module_file)

        # 8) Convert closure/module deps -> DependencyInfo list
        base_dep_infos: List[DependencyInfo] = _build_dependency_infos(raw_deps)

        # 9) Add imported modules from the defining file (also DependencyInfo)
        imported_dep_infos: List[DependencyInfo] = []
        if module_file:
            with open(module_file, "r", encoding="utf-8") as f:
                source_file_content = f.read()
            imported_dep_infos = cls._infer_imported_modules_from_file(
                module_file=module_file,
                src=source_file_content,
                package_root=package_root,
            )

        # 10) Merge and dedupe DependencyInfo entries
        merged: dict[tuple[str, str], DependencyInfo] = {}

        for info in base_dep_infos + imported_dep_infos:
            key = (info.root_module, info.submodule)
            # last-one-wins; they're usually identical anyway
            merged[key] = info

        dep_infos = list(merged.values())

        # 11) Build the EmbeddedFunction dataclass
        built = cls(
            name=func.__name__,
            source=src,
            source_file_content=source_file_content,
            dependencies_map=dep_infos,
            aliases=aliases,
            package_root=package_root,
        )

        # Cache the original callable for local use
        built._compiled = func

        return built

    # ---------- rebuild ----------
    def build(self, globals_ns: Dict[str, Any] | None = None) -> Callable[..., Any]:
        """
        Rebuild the function object from stored source.

        globals_ns:
            Optional globals dict to exec into. If None, a fresh empty dict
            is used (so you must import everything you need inside the function).

        We also:
          - add dependency root paths to sys.path (if they exist),
          - restore module aliases (e.g. MATH_MOD -> math) based on the
            original closure's module-valued globals.
        """
        if globals_ns is None:
            globals_ns = {}

        # 1) Make sure dependency roots are on sys.path
        import sys as _sys
        import os as _os

        for dep in self.dependencies_map:
            root_path = dep.root_path
            if (
                root_path
                and _os.path.isdir(root_path)
                and root_path not in _sys.path
            ):
                # prepend so user code takes precedence over site-packages
                _sys.path.insert(0, root_path)

        # 2) Restore module aliases first so the function body can use them
        for alias, module_name in self.aliases.items():
            try:
                globals_ns[alias] = importlib.import_module(module_name)
            except Exception:
                # If import fails, leave the alias missing; calling the function
                # will raise NameError, which is explicit enough.
                pass

        # 3) Exec the function source
        exec(self.source, globals_ns)  # noqa: S102 - intentional
        fn = globals_ns.get(self.name)
        if not callable(fn):
            raise RuntimeError(
                f"After exec, object named {self.name!r} is not callable "
                f"(got {type(fn)!r})"
            )
        return fn

    def _ensure_compiled(self) -> Callable[..., Any]:
        """
        Lazily compile the function once and cache it.
        """
        if self._compiled is None:
            self._compiled = self.build()
        return self._compiled

    @property
    def func(self):
        return self._ensure_compiled()

    # ---------- make it callable ----------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the embedded function as if it were a normal Python function.
        """
        fn = self._ensure_compiled()
        return fn(*args, **kwargs)

    # ---------- dependency inspection ----------

    def list_dependencies(self) -> List[DependencyInfo]:
        """
        Return the dependency infos:
            DependencyInfo(root_module, submodule, root_path)
        """
        return list(self.dependencies_map)

    def check_dependencies(self) -> Dict[str, DependencyCheckResult]:
        """
        Check whether dependencies are importable *on this machine* and where
        they resolve to.

        Uses the submodule (e.g. "databricks.sdk") as the thing to import.

        Returns:
            {
              submodule: DependencyCheckResult(...),
              ...
            }
        """
        results: Dict[str, DependencyCheckResult] = {}

        for info in self.dependencies_map:
            try:
                mod = importlib.import_module(info.submodule)
                importable = True
                runtime_file = getattr(mod, "__file__", None)
                error: Optional[str] = None
            except Exception as exc:
                importable = False
                runtime_file = None
                error = repr(exc)

            results[info.submodule] = DependencyCheckResult(
                root_module=info.root_module,
                root_path=info.root_path,
                importable=importable,
                runtime_file=runtime_file,
                error=error,
            )

        return results

    def to_command(
        self,
        args: list,
        kwargs: dict,
        env_keys: Optional[List[str]] = None
    ) -> str:
        """
        Build a Python command string suitable for Databricks `command_execution`.

        The generated code will:
          - best-effort add dependency root paths to sys.path
          - restore module aliases used by the function
          - reconstruct args/kwargs via dill (sent from driver)
          - call the function with *args / **kwargs
          - capture stdout/stderr produced during the call
          - print a base64-encoded dill-serialized dict:
                {
                  "result_b64": <base64 of dill(result or DataFrame(result))>,
                  "stdout": "<captured stdout>",
                  "stderr": "<captured stderr>",
                }
        """
        import dill  # driver-side
        import base64 as _b64
        import os as _os

        # Collect unique, non-empty dependency root paths
        roots = sorted({
            _os.path.dirname(d.root_path) for d in self.dependencies_map if d.root_path
        })

        if self.package_root:
            roots += _os.path.dirname(self.package_root)

        imports = set(d.submodule for d in self.dependencies_map)

        lines: list[str] = [
            "import sys",
            "import os",
            "import base64",
            "import importlib",
            "import json",
            "import dill",
            "import pandas",
            "import io",
            "import contextlib",
        ]

        # --- embed env vars into remote process ---
        if env_keys:
            env_vars = {
                k: os.getenv(k)
                for k in env_keys
                if os.getenv(k) is not None
            }

            if env_vars:
                # env_vars is a plain dict on the driver; we stringify it into code
                lines.append(f"__embedded_env = {env_vars!r}")
                lines.extend([
                    "for __k, __v in __embedded_env.items():",
                    "    if __v is not None:",
                    "        os.environ[__k] = __v",
                ])

        # --- sys.path bootstrapping on remote ---
        if roots:
            lines.append(f"for __p in {roots!r}:")
            lines.extend([
                "    if __p and __p not in sys.path:",
                "        if os.path.isdir(__p):",
                "            sys.path.insert(0, __p)",
            ])

        # --- best-effort import of dependencies on remote ---
        if imports:
            lines.extend([
                "try:\n    import %s\nexcept:\n    pass" % _ for _ in imports
            ])

        # --- restore module aliases on remote ---
        if self.aliases:
            lines.append(f"__embedded_aliases = {self.aliases!r}")
            lines.append("for __name, __mod in __embedded_aliases.items():")
            lines.append("    globals()[__name] = importlib.import_module(__mod)")

        # --- driver-side serialization helpers ---
        def _b64_dumps(obj: Any) -> str:
            return _b64.b64encode(dill.dumps(obj)).decode("utf-8")

        # Serialize args/kwargs as plain data
        args_ser = _b64_dumps(tuple(args))
        kwargs_ser = _b64_dumps(dict(kwargs))

        # Prefer full file content if we have it, otherwise just the function source
        source_block = self.source_file_content or self.source

        lines.extend([
            "",
            source_block,
            "",
        ])

        # --- remote-side deserialization + log capture ---
        lines.extend([
            f"_embedded_func = {self.name}",
            f"_embedded_args = dill.loads(base64.b64decode({args_ser!r}.encode('utf-8')))",
            f"_embedded_kwargs = dill.loads(base64.b64decode({kwargs_ser!r}.encode('utf-8')))",
            "",
            "_embedded_result = _embedded_func(*_embedded_args, **_embedded_kwargs)",
            "try:",
            "    _ser_result = dill.dumps(_embedded_result)",
            "except Exception:",
            "    _ser_result = dill.dumps(pandas.DataFrame(_embedded_result))",
            "",
            "payload = {",
            "    'result_b64': base64.b64encode(_ser_result).decode('utf-8'),",
            "}",
            "print('<<<EMBEDDED_RESULT_START>>>' + json.dumps(payload) + '<<<EMBEDDED_RESULT_END>>>')",
        ])

        return "\n".join(lines)
