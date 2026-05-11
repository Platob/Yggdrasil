"""Hash-based singleton mixin for dataclasses and similar value objects."""
from __future__ import annotations

import dataclasses
import threading
from typing import Any, ClassVar

__all__ = ["Singleton"]


def _wrap_init(cls: type) -> None:
    """Install an idempotency guard on ``cls.__init__`` if not already wrapped.

    ``@dataclass`` replaces ``__init__`` *after* ``__init_subclass__`` runs, so
    the wrap has to happen lazily — from ``__new__`` on the first construction.
    """
    init = cls.__init__
    if getattr(init, "_ygg_singleton_wrapped", False):
        return

    original = init

    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        # Python calls __init__ on whatever __new__ returns — either the
        # cached singleton or the freshly-stored draft. In both cases the
        # original __init__ has already run from inside __new__, so skip.
        if id(self) in Singleton._INITIALIZED_IDS:
            return
        original(self, *args, **kwargs)
        Singleton._INITIALIZED_IDS.add(id(self))

    _wrapped._ygg_singleton_wrapped = True  # type: ignore[attr-defined]
    _wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    cls.__init__ = _wrapped  # type: ignore[assignment]


class Singleton:
    """Mixin that caches one live instance per ``__hash__`` per subclass.

    Two constructions whose resulting instances hash equal collapse to the
    same live object. Pair this with a hashable dataclass (or any class that
    defines ``__hash__`` from its fields) to get value-object identity::

        @dataclass(frozen=True)
        class Endpoint(Singleton):
            host: str
            port: int = 443

        Endpoint("api", 443) is Endpoint("api", 443)   # True
        Endpoint("api", 443) is Endpoint("api", 80)    # False

    How it works
    ------------

    ``__new__`` builds a draft, runs ``__init__`` on it once so the fields
    ``__hash__`` reads are populated, then looks up ``(cls, hash(draft))``
    in the global :attr:`_INSTANCES` cache. A live instance at that key is
    returned as-is; otherwise the draft is stored and returned. Python
    invokes ``__init__`` a second time on whatever ``__new__`` returns —
    the wrapper installed lazily on the subclass's ``__init__`` skips that
    second pass via :attr:`_INITIALIZED_IDS`.

    Pickling
    --------

    Dataclass subclasses pickle as ``(cls, init_field_values)`` via
    :meth:`__reduce__`, so unpickling routes back through ``__new__`` and
    collapses to the in-process singleton when one already exists for that
    hash. Non-dataclass subclasses should override ``__reduce__`` /
    ``__getnewargs__`` themselves if they need pickle support.
    """

    # One global ``(cls, hash)`` → instance map shared across every
    # subclass. The class identity in the key keeps unrelated subclasses
    # with colliding hashes apart.
    _INSTANCES: ClassVar[dict[tuple[type, int], "Singleton"]] = {}
    _INSTANCES_LOCK: ClassVar[threading.Lock] = threading.Lock()

    # Ids of objects whose wrapped ``__init__`` has already populated the
    # instance. Kept alive transitively by :attr:`_INSTANCES`, so the ids
    # cannot be recycled while the entry matters.
    _INITIALIZED_IDS: ClassVar[set[int]] = set()

    def __new__(cls, *args: Any, **kwargs: Any) -> "Singleton":
        _wrap_init(cls)
        draft = object.__new__(cls)
        # Populate fields so __hash__ has something to read. The wrapper
        # records the id afterwards so Python's follow-up __init__ call on
        # the returned instance becomes a no-op.
        cls.__init__(draft, *args, **kwargs)
        try:
            h = hash(draft)
        except TypeError as exc:
            Singleton._INITIALIZED_IDS.discard(id(draft))
            raise TypeError(
                f"{cls.__name__} inherits Singleton but its instances are "
                f"unhashable ({exc}). Make the class hashable — e.g. "
                f"@dataclass(frozen=True) or @dataclass(unsafe_hash=True) — "
                f"so Singleton can key the global cache by hash."
            ) from None
        key = (cls, h)
        with cls._INSTANCES_LOCK:
            cached = cls._INSTANCES.get(key)
            if cached is not None:
                # Draft is about to be garbage-collected — drop its id so
                # the next allocation that happens to reuse it isn't
                # mistaken for an already-initialized instance.
                Singleton._INITIALIZED_IDS.discard(id(draft))
                return cached
            cls._INSTANCES[key] = draft
            return draft

    def __reduce__(self) -> Any:
        # Pickle dataclass subclasses through the constructor so unpickle
        # routes via __new__ and collapses to the live singleton when one
        # already exists. Non-dataclass subclasses fall back to the default
        # protocol — they should override this if they need pickle support.
        cls = type(self)
        if dataclasses.is_dataclass(cls):
            args = tuple(
                getattr(self, f.name)
                for f in dataclasses.fields(cls)
                if f.init
            )
            return (cls, args)
        return super().__reduce_ex__(2)
