# fake_module.py
import sys
from types import ModuleType
from typing import Any


__all__ = ["make_fake_module", "FakeObject", "FakeModule"]


class FakeObject:
    __slots__ = ("_name", "_inner")  # _inner will store monkey-patched attributes

    def __hash__(self):
        return hash(self._name)

    def __init__(self, name: str | None = None):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_inner", {})  # store any dynamically assigned attributes

    def __getattr__(self, item: str) -> "FakeObject":
        inner = object.__getattribute__(self, "_inner")
        if item in inner:
            return inner[item]  # return previously set monkey-patched value
        name = object.__getattribute__(self, "_name")
        val = FakeObject(f"{name}.{item}" if name else item)
        inner[item] = val  # cache so repeated access returns the same
        return val

    def __setattr__(self, name: str, value):
        # always store in _inner for monkey-patch support
        inner = object.__getattribute__(self, "_inner")
        inner[name] = value

    def __call__(self, *args, **kwargs):
        name = object.__getattribute__(self, "_name")
        if not args and not kwargs:
            return self
        raise ImportError(f"Cannot call {name!r}, module not installed.")

    def __await__(self):
        return None

    def __iter__(self):
        yield None

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    def __int__(self):
        return 0

    def __str__(self):
        return "None"

    def __repr__(self):
        return "<FakeNone>"

    def __getitem__(self, key):
        inner = object.__getattribute__(self, "_inner")
        name = object.__getattribute__(self, "_name")
        if key in inner:
            return inner[key]
        val = FakeObject(f"{name}[{key}]" if name else f"[{key}]")
        inner[key] = val
        return val

    def __eq__(self, other: Any) -> bool:
        return other is None


class FakeModule(ModuleType):
    """
    A module-like object:
    - If strict True: every attribute returns None (strict mode)
    - If strict False: attributes return _FakeValue objects (callable/awaitable/chainable)
    """
    def __init__(self, name: str, strict: bool = False):
        super().__init__(name)
        self.__dict__["_fake_strict"] = bool(strict)
        self.__dict__["__is_fake_module__"] = True

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return self.__dict__[name]
        if self.__dict__.get("_fake_strict", False):
            return None
        return FakeObject(name)

    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)


def make_fake_module(module_name: str, strict: bool = False, inject: bool = True) -> ModuleType:
    """
    Create a fake module object if it does not exist.

    - module_name: name for the module (e.g. "heavy.thirdparty.client")
    - strict: if True, attributes return None
    - inject: if True, insert into sys.modules
    """
    if module_name in sys.modules:
        return sys.modules[module_name]

    fake = FakeModule(module_name, strict=strict)
    if inject:
        sys.modules[module_name] = fake
    return fake
