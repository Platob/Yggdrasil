from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Never, Iterable, Optional
import importlib

__all__ = ["Dummy"]


@dataclass(frozen=True, slots=True)
class Dummy:
    _module_path: tuple[str, ...]

    @classmethod
    def from_name(
        cls,
        *module_path: str | Iterable[str],
        to_class: Optional[bool] = None,
        try_import: bool = False,
    ) -> Any:
        """
        Examples:
          Dummy.from_name("pyarrow", try_import=True, to_class=False) -> module or Dummy
          Dummy.from_name("pyarrow", "compute", to_class=False)       -> Dummy(("pyarrow","compute"))
          Dummy.from_name(["pyarrow","compute"], to_class=False)      -> Dummy(("pyarrow","compute"))

          Dummy.from_name("pyarrow", "Table", to_class=True, try_import=True) -> pa.Table (class) or Dummy
          Dummy.from_name("pyarrow", "compute", "cast", to_class=True, try_import=True) -> resolved attr or Dummy
        """
        # Normalize inputs into a flat tuple[str, ...]
        parts: list[str] = []
        for p in module_path:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, Iterable):
                parts.extend(str(x) for x in p)
            else:
                parts.append(str(p))

        path = tuple(parts) if parts else ("<unknown>",)

        if to_class is None:
            # Class name should by default start by upper case letter
            to_class = path[-1][0].isupper()

        # If caller doesn't want resolution, always return a Dummy path (except try_import module shortcut)
        if not to_class:
            if try_import:
                root = path[0]
                try:
                    return importlib.import_module(root)
                except ModuleNotFoundError:
                    return cls(path)
            return cls(path)

        # to_class=True: resolve as far as possible (module import + attribute walk).
        if not try_import:
            # Contract: if you didn't allow importing, we can't actually resolve anything.
            return cls(path)

        root = path[0]

        # Try to import the *longest* module prefix we can.
        # Example: ("a","b","C") -> try import "a.b" then getattr "C".
        mod = None
        imported_upto = 0
        for i in range(len(path), 0, -1):
            mod_name = ".".join(path[:i])
            try:
                mod = importlib.import_module(mod_name)
                imported_upto = i
                break
            except ModuleNotFoundError:
                continue

        # If nothing imports, bail to Dummy
        if mod is None:
            # One more attempt: import root only (helps when full prefix import fails weirdly)
            try:
                mod = importlib.import_module(root)
                imported_upto = 1
            except ModuleNotFoundError:
                return cls(path)

        # Walk remaining parts as attributes
        obj: Any = mod
        for name in path[imported_upto:]:
            try:
                obj = getattr(obj, name)
            except AttributeError:
                return cls(path)

        return obj

    def materialize(self, raise_error: bool = True) -> Any:
        """
        Try to import and resolve the dotted path.

        - If _path is ("pyarrow",) -> returns imported module
        - If _path is ("pyarrow","compute") -> tries:
            1) import "pyarrow.compute" as a module
            2) else import "pyarrow" then getattr "compute"
        - If anything fails:
            - raise_error=True  -> raise ModuleNotFoundError with nice message
            - raise_error=False -> return Dummy(self._path)
        """
        if not self._module_path:
            if raise_error:
                self._raise("import")
            return Dummy(("<unknown>",))

        root = self._module_path[0]
        target = self.dotted()

        # 1) Try importing the full dotted path directly (best case: it's a module)
        try:
            obj: Any = importlib.import_module(target)
        except ModuleNotFoundError:
            # 2) Fallback: import root, then resolve attributes
            try:
                obj = importlib.import_module(root)
            except ModuleNotFoundError:
                if raise_error:
                    self._raise("import")
                return Dummy(self._module_path)

            for attr in self._module_path[1:]:
                try:
                    obj = getattr(obj, attr)
                except AttributeError:
                    if raise_error:
                        self._raise("import", f"attribute '{attr}' not found")
                    return Dummy(self._module_path)

        return obj

    def dotted(self) -> str:
        return ".".join(self._module_path) if self._module_path else "<unknown>"

    def _raise(self, action: str, detail: str | None = None) -> Never:
        root = self._module_path[0] if self._module_path else "<unknown>"
        target = self.dotted()
        extra = f" ({detail})" if detail else ""
        raise ModuleNotFoundError(
            f"Missing optional dependency '{root}'. Tried to {action} on '{target}'{extra}."
        )

    # --- attribute access: reconstitute chain instead of failing ---
    def __getattr__(self, name: str) -> "Dummy":
        return Dummy((*self._module_path, name))

    # --- actual usage: fail loudly ---
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._raise("call")

    def __bool__(self) -> bool:
        self._raise("coerce to bool")

    def __iter__(self):
        self._raise("iterate")

    def __len__(self) -> int:
        self._raise("get length")

    def __contains__(self, item: Any) -> bool:
        self._raise("check containment")

    def __getitem__(self, key: Any) -> Any:
        self._raise("index", repr(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        self._raise("set item", repr(key))

    def __delitem__(self, key: Any) -> None:
        self._raise("delete item", repr(key))

    def __repr__(self) -> str:
        return f"<Dummy missing '{self.dotted()}'>"
