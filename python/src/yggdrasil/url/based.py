from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

__all__ = ["URLBased", "_URL_BASED_REGISTRY"]


# ===========================================================================
# URLBased — registry-driven dispatch for URL-addressable classes
# ===========================================================================


# Per-process registry of :class:`URLBased` subclasses, keyed by their
# canonical :class:`Scheme` member. Populated by
# :meth:`URLBased.__init_subclass__` whenever a subclass declares
# ``scheme = Scheme.X`` on the class body.
_URL_BASED_REGISTRY: dict[Any, type] = {}


class URLBased(ABC):
    """Mixin for any class addressable by a :class:`URL`.

    Subclasses declare a class-level
    ``scheme: ClassVar[Scheme | None]`` on the class body; on subclass
    creation :meth:`__init_subclass__` registers the class against
    that :class:`Scheme` member in the global
    :data:`_URL_BASED_REGISTRY`. The registry is the single source of
    truth for "what class handles ``s3://``" / ``"dbfs://"`` / …;
    callers either look it up directly (``URLBased.for_scheme(...)``)
    or hand a URL to :meth:`URLBased.dispatch` and let URLBased pick
    the right subclass.

    The two abstract hooks every subclass implements:

    - :meth:`from_url(cls, url, **kwargs)` — build an instance of the
      subclass from a :class:`URL`. Concrete subclasses typically
      forward to their own ``__init__``.
    - :meth:`to_url(self) -> URL` — render this instance back to its
      canonical URL form.

    Together these make a :class:`URLBased` round-trippable through
    a URL: ``cls.from_url(obj.to_url())`` is the identity for any
    well-behaved subclass.
    """

    #: Canonical scheme this class handles. ``None`` on the abstract
    #: base; concrete subclasses set ``Scheme.X``. May be assigned a
    #: plain string at the class-body level — :meth:`__init_subclass__`
    #: coerces it to the matching :class:`Scheme` member so the rest
    #: of the codebase can rely on the typed form.
    scheme: ClassVar["Any"] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        scheme = cls.__dict__.get("scheme", None)
        # ``None`` and the empty-string sentinel both mean "abstract /
        # not directly registrable" — intermediate bases (e.g.
        # :class:`Path`, :class:`RemotePath`, :class:`DatabricksPath`
        # before it gets its own scheme) leave it unset.
        if scheme is None or scheme == "":
            return
        from yggdrasil.enums import Scheme
        coerced = Scheme.from_(scheme)
        # Store the typed form back on the class so ``cls.scheme`` is
        # always a :class:`Scheme` member from this point on.
        cls.scheme = coerced
        existing = _URL_BASED_REGISTRY.get(coerced)
        if existing is not None and existing is not cls and not issubclass(cls, existing):
            raise RuntimeError(
                f"Duplicate URLBased registration for scheme "
                f"{coerced!r}: {cls.__name__} clashes with "
                f"{existing.__name__}."
            )
        _URL_BASED_REGISTRY[coerced] = cls

    # ------------------------------------------------------------------
    # Registry lookup
    # ------------------------------------------------------------------

    @classmethod
    def for_scheme(cls, scheme: Any) -> "type[URLBased]":
        """Return the :class:`URLBased` subclass registered for *scheme*.

        Lazy: if no subclass is registered yet, this routes through
        :meth:`Scheme.path_class` which imports the backend module on
        demand (firing :meth:`__init_subclass__` as a side effect).

        Raises :class:`ValueError` for an unknown scheme and
        :class:`ImportError` when the backend's optional dependencies
        aren't installed.
        """
        from yggdrasil.enums import Scheme
        s = Scheme.from_(scheme)
        registered = _URL_BASED_REGISTRY.get(s)
        if registered is not None:
            return registered
        # Cold dispatch — lazy-import the backend module.
        return s.path_class()

    @classmethod
    def dispatch(cls, url: Any, **kwargs: Any) -> "URLBased":
        """Build the right :class:`URLBased` subclass from *url*.

        Looks up the subclass via :meth:`for_scheme`, then delegates
        to that subclass's :meth:`from_url`. Used as the cross-cutting
        entry point when the caller has a URL but doesn't know (or
        care) which concrete class owns its scheme.

        ``URL.from_(url).scheme`` drives the lookup; an empty scheme
        falls back to the ``file://`` handler so bare paths work.
        """
        from yggdrasil.url.url import URL
        u = URL.from_(url)
        scheme = u.scheme or "file"
        target = cls.for_scheme(scheme)
        return target.from_url(u, **kwargs)

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_url(cls, url: "URL | str", **kwargs: Any) -> "URLBased":
        """Construct an instance of *cls* from *url*.

        Concrete subclasses typically forward to ``cls(url=url, **kwargs)``;
        backends with extra construction knobs (auth tokens, sessions,
        workspace clients) override to thread those through.
        """
        ...

    @abstractmethod
    def to_url(self) -> "URL":
        """The canonical :class:`URL` that addresses this instance."""
        ...
