"""Root exception type for the whole library.

``YGGException`` is the single ancestor every yggdrasil-defined
exception should derive from — directly or transitively — so a
caller can write::

    try:
        do_yggdrasil_things()
    except YGGException:
        ...

and catch every error this library deliberately raises in one branch.
Concrete subclasses live next to this module:

- :mod:`yggdrasil.exceptions.cast`  → :class:`CastError`
- :mod:`yggdrasil.exceptions.http`  → :class:`HTTPError` and the
  full HTTP status / connection / pool / location hierarchy.

When you need a new exception type, add it here (or in a peer file
under :mod:`yggdrasil.exceptions`) and subclass :class:`YGGException`.
See ``AGENTS.md`` → "Centralise exceptions in
:mod:`yggdrasil.exceptions`" for the rule.
"""
from __future__ import annotations


__all__ = ["YGGException"]


class YGGException(Exception):
    """Root of every exception the yggdrasil library raises on its own."""
