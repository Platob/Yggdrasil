# fake_module.py
import sys
from types import ModuleType
from typing import Any


__all__ = [
    "FakeModule",
    "make_fake_module"
]


class _FakeValue:
    """
    Chainable fake value:
    - attribute access returns another _FakeValue (chainable)
    - calling returns self (so you can do fake.foo().bar().baz)
    - awaiting returns None (so await fake.foo works)
    - iterating yields nothing
    - bool(self) -> False, len(self) -> 0, int(self)->0, str(self)->'None'
    - indexing returns another _FakeValue
    - equality to None is True (so fake == None)
    """
    __slots__ = ("_name",)
    def __init__(self, name: str | None = None):
        self._name = name

    def __getattr__(self, item: str) -> "_FakeValue":
        return _FakeValue(f"{self._name}.{item}" if self._name else item)

    def __call__(self, *args, **kwargs) -> "_FakeValue":
        # return self so chained calls work; awaiting the returned object yields None
        return self

    def __await__(self):
        # make `await fake.x` or `await fake.x()` valid and produce None
        return None

    def __iter__(self):
        # empty iterator
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
        return _FakeValue(f"{self._name}[{key}]" if self._name else f"[{key}]")

    # make equality to None evaluate True so `if fake is None` won't pass, but `fake == None` will
    def __eq__(self, other: Any) -> bool:
        return other is None

    # you can optionally expose a value() method to force explicit None
    def value(self):
        return None


class _StrictNoneValue:
    """Always returns None for attribute access, calling, awaiting, indexing, iter."""
    def __getattr__(self, item):
        return None
    def __call__(self, *a, **k):
        return None
    def __await__(self):
        return None
    def __iter__(self):
        return iter(())
    def __getitem__(self, k):
        return None
    def __repr__(self):
        return "None"
    def __bool__(self):
        return False
    def value(self):
        return None


class FakeModule(ModuleType):
    """
    A module-like object:
    - If strict True: every attribute returns None (strict mode).
    - If strict False: attributes return a chainable _FakeValue (safer for call/await chains).
    """
    def __init__(self, name: str, strict: bool = False):
        super().__init__(name)
        self.__dict__["_fake_strict"] = bool(strict)
        # expose a sentinel attr in case tests need it
        self.__dict__["__is_fake_module__"] = True

    def __getattr__(self, name: str):
        # avoid recursion for our own fields
        if name in self.__dict__:
            return self.__dict__[name]
        if self.__dict__.get("_fake_strict", False):
            return None
        return _FakeValue(name)

    def __setattr__(self, name: str, value):
        # allow setting real attrs on the fake module if desired
        super().__setattr__(name, value)

    def __repr__(self):
        return f"<FakeModule {self.__name__!r} strict={self.__dict__.get('_fake_strict', False)}>"

def make_fake_module(module_name: str, strict: bool = False, inject: bool = True) -> ModuleType:
    """
    Create a fake module object.

    - module_name: name for the module (e.g. "heavy.thirdparty.client")
    - strict: if True, every attribute access immediately returns None
              if False, attributes return _FakeValue objects (callable/awaitable/chainable)
    - inject: if True, the fake is inserted into sys.modules under module_name
    """
    fake = FakeModule(module_name, strict=strict)
    if inject:
        sys.modules[module_name] = fake
    return fake
