"""Clickable-repr mixin driven by ``explore_url``.

A resource that can point at a web UI (AWS Console, Databricks workspace, an S3
object) defines :attr:`explore_url`; this mixin turns that into a repr you can
click from a repl / notebook:

* ``repr(obj)`` → ``ClassName(<url>)`` when ``explore_url`` is defined,
  otherwise ``super().__repr__()`` (the plain object / path repr);
* ``_repr_html_`` → a real ``<a>`` anchor in Jupyter / IPython, else delegates
  to ``super()._repr_html_`` if one exists.

Both paths swallow errors from ``explore_url`` so a half-constructed or
mid-teardown instance still reprs (a degraded line beats a formatting crash).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from yggdrasil.url.url import URL

__all__ = ["ExploreUrlRepr"]


class ExploreUrlRepr:
    @property
    def explore_url(self) -> "Optional[URL]":
        """Web-UI deep-link for this resource, or ``None``. Override in
        subclasses; the repr / HTML below key off it."""
        return None

    def __repr__(self) -> str:
        try:
            url = self.explore_url
        except Exception:
            url = None
        if url is None:
            return super().__repr__()
        return f"{type(self).__name__}({url!r})"

    def _repr_html_(self) -> "Optional[str]":
        try:
            url = self.explore_url
        except Exception:
            url = None
        if url is None:
            sup = getattr(super(), "_repr_html_", None)
            return sup() if callable(sup) else None
        return f'<a href="{url}" target="_blank">{type(self).__name__}: {url}</a>'
