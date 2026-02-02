from abc import ABC, abstractmethod


__all__ = ["DummyModuleClass"]


class DummyModuleClass(ABC):
    """
    Hard-fail dummy proxy: any interaction raises, except a few safe internals.
    """

    @classmethod
    @abstractmethod
    def module_name(cls) -> str:
        raise NotImplementedError

    def _raise(self, action: str, name: str | None = None):
        target = type(self).module_name()
        extra = f" '{name}'" if name else ""
        raise ModuleNotFoundError(
            f"{type(self).__name__} is a dummy for missing optional dependency "
            f"module '{target}'. Tried to {action}{extra}."
        )

    # --- attribute access / mutation ---
    def __getattribute__(self, name: str):
        # allow introspection / internals without blowing up
        if name in {"module_name", "_raise", "__class__", "__dict__", "__repr__", "__str__", "__dir__"}:
            return object.__getattribute__(self, name)
        self._raise("access attribute", name)

    def __getattr__(self, name: str):
        self._raise("access attribute", name)

    def __setattr__(self, name: str, value):
        self._raise("set attribute", name)

    def __delattr__(self, name: str):
        self._raise("delete attribute", name)

    def __dir__(self):
        # show minimal surface
        return ["module_name"]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} dummy for '{type(self).module_name()}'>"

    def __str__(self) -> str:
        return self.__repr__()

    # --- common "other" interactions ---
    def __call__(self, *args, **kwargs):
        self._raise("call module")

    def __getitem__(self, key):
        self._raise("index", str(key))

    def __setitem__(self, key, value):
        self._raise("set item", str(key))

    def __delitem__(self, key):
        self._raise("delete item", str(key))

    def __iter__(self):
        self._raise("iterate")

    def __len__(self):
        self._raise("get length")

    def __contains__(self, item):
        self._raise("check containment")

    def __bool__(self):
        self._raise("coerce to bool")


# Example:
# class PyArrowDummy(DummyModuleClass):
#     @classmethod
#     def module_name(cls) -> str:
#         return "pyarrow"
