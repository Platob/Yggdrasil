from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import MISSING
from difflib import get_close_matches
from typing import Any, AnyStr, Mapping, TYPE_CHECKING, Union, Iterable

import yggdrasil.pickle.json as json_module
from yggdrasil.data.constants import TAG_PREFIX, DEFAULT_VALUE_KEY

if TYPE_CHECKING:
    from .data_field import Field


__all__ = [
    "BaseMetadata",
    "BaseChildrenFields",
    "_merge_metadata_and_tags",
    "_normalize_metadata",
    "_to_bytes",
]


SelectType = Union[str, int, "Field"]


def _to_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bool):
        return b"true" if value else b"false"
    return json_module.dumps(
        value,
        to_bytes=True,
        safe=False,
        separators=(",", ":"),
        ensure_ascii=False
    )


def _normalize_metadata(
    metadata: dict[Any, Any] | None,
    tags: dict[Any, Any] | None,
    default_value: Any = None,
) -> dict[bytes, bytes] | None:
    if not metadata and not tags and default_value is None:
        return None

    normalized = {
        _to_bytes(key): _to_bytes(value)
        for key, value in (metadata or {}).items()
        if key and value is not None
    }

    if tags:
        normalized.update(
            {
                TAG_PREFIX + _to_bytes(key): _to_bytes(value)
                for key, value in tags.items()
                if key and value
            }
        )

    if default_value is not None and default_value is not MISSING and default_value is not ...:
        try:
            normalized[DEFAULT_VALUE_KEY] = json_module.dumps(
                default_value, safe=False, to_bytes=True,
                ensure_ascii=False, separators=(",", ":")
            )
        except Exception as e:
            logging.error(f"Could not serialize default value {default_value!r}: {e}")

    return normalized or None


def _merge_metadata_and_tags(
    metadata: dict[bytes, bytes] | None,
    tags: dict[bytes, bytes] | None,
) -> dict[bytes, bytes] | None:
    merged: dict[bytes, bytes] = dict(metadata or {})

    if tags:
        merged.update(
            {
                key if key.startswith(TAG_PREFIX) else TAG_PREFIX + key: value
                for key, value in tags.items()
            }
        )

    return merged or None


class BaseMetadata(ABC):
    metadata: dict[bytes, bytes] | None

    @abstractmethod
    def _empty_tags(self) -> dict[bytes, bytes] | None:
        """Return the subtype-specific empty value for tags."""

    def _prefixed_metadata(self, prefix: bytes) -> dict[bytes, bytes]:
        if not self.metadata:
            return {}

        return {
            key[len(prefix):]: value
            for key, value in self.metadata.items()
            if key.startswith(prefix)
        }

    def _update_prefixed_metadata(
        self,
        prefix: bytes,
        values: Mapping[bytes | str, bytes | str | object] | None,
    ) -> None:
        if not values:
            return

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

        self.metadata.update(
            {
                prefix + _to_bytes(key): _to_bytes(value)
                for key, value in values.items()
                if key and value is not None
            }
        )

    def _tag_value(self, key: bytes | str) -> bytes | None:
        if not self.metadata:
            return None
        return self.metadata.get(TAG_PREFIX + _to_bytes(key))

    def _set_tag_value(self, key: bytes | str, value: Any | None) -> None:
        if value is not None:
            if not self.metadata:
                object.__setattr__(self, "metadata", {})
            self.metadata[TAG_PREFIX + _to_bytes(key)] = _to_bytes(value)
            self._on_metadata_mutated()

    def _unset_tag_value(self, key: bytes | str) -> None:
        if not self.metadata:
            return
        self.metadata.pop(TAG_PREFIX + _to_bytes(key), None)
        if not self.metadata:
            object.__setattr__(self, "metadata", None)
        self._on_metadata_mutated()

    def _on_metadata_mutated(self) -> None:
        """Hook called whenever this metadata holder's dict changes.

        :class:`~yggdrasil.data.data_field.Field` overrides this to
        clear its cached arrow / polars / spark projections (and
        cascade to parents). Default is a no-op so non-Field metadata
        holders pay nothing.
        """
        return None

    def _tag_flag(self, key: bytes | str) -> bool:
        value = self._tag_value(key)
        return bool(value)

    def _tag_text(self, key: bytes | str) -> str | None:
        value = self._tag_value(key)
        if not value:
            return None
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    @property
    def tags(self) -> dict[bytes, bytes] | None:
        tags = self._prefixed_metadata(TAG_PREFIX)
        if tags:
            return tags
        return self._empty_tags()

    @tags.setter
    def tags(self, value: Mapping[AnyStr, AnyStr] | None):
        self.update_tags(value)

    def update_tags(self, value: Mapping[AnyStr, AnyStr] | None) -> None:
        self._update_prefixed_metadata(TAG_PREFIX, value)

    @property
    def partition_by(self) -> bool:
        return self._tag_flag(b"partition_by")

    @property
    def cluster_by(self) -> bool:
        return self._tag_flag(b"cluster_by")

    @property
    def primary_key(self) -> bool:
        return self._tag_flag(b"primary_key")

    @property
    def foreign_key(self) -> bool:
        return self._tag_flag(b"foreign_key")

    @property
    def constraint_key(self) -> bool:
        return self._tag_flag(b"constraint_key")

    @property
    def sorted(self) -> bool:
        return self._tag_flag(b"sorted")

    @property
    def comment(self):
        if not self.metadata:
            return None

        comment = self.metadata.get(b"comment", None)

        if comment:
            return comment.decode('utf-8')
        return None


class BaseChildrenFields(ABC):

    def __getitem__(self, item):
        return self.field(item)

    def __iter__(self):
        return iter(self.children_fields)

    @property
    @abstractmethod
    def children_fields(self) -> list["Field"]:
        ...

    @property
    def primary_fields(self):
        return [f for f in self.children_fields if f.primary_key]

    @property
    def partition_fields(self):
        return [f for f in self.children_fields if f.partition_by]

    @property
    def cluster_fields(self):
        return [f for f in self.children_fields if f.cluster_by]

    # ------------------------------------------------------------------
    # Field lookup map — cached on Field, recomputed on miss elsewhere
    # ------------------------------------------------------------------

    def _build_field_name_maps(
        self,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Return ``(exact_map, casefold_map)`` over ``children_fields``.

        Both map field name → index. The exact map dispatches the
        common case (case-sensitive lookup); the casefold map is the
        fallback for ``"PRICE"`` -> ``"price"`` style requests.
        Earlier children win on collision so legacy positional
        semantics are preserved.
        """
        exact: dict[str, int] = {}
        fold: dict[str, int] = {}
        for i, f in enumerate(self.children_fields):
            exact.setdefault(f.name, i)
            fold.setdefault(f.name.casefold(), i)
        return exact, fold

    def _field_name_index_maps(
        self,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Cached ``(exact, casefold)`` name→index maps.

        :class:`~yggdrasil.data.data_field.Field` populates per-instance
        slots (``_field_name_map`` / ``_field_name_fold_map``) so the
        maps survive across calls and get cleared on
        :meth:`Field._invalidate_cache`. Subtypes that don't carry
        those slots (e.g. struct/map/array dtypes) recompute on each
        call — still O(N) worst case but the dict path is faster than
        the legacy double-loop above a handful of fields.
        """
        exact = getattr(self, "_field_name_map", None)
        fold = getattr(self, "_field_name_fold_map", None)
        if exact is not None and fold is not None:
            return exact, fold
        exact, fold = self._build_field_name_maps()
        # Best-effort: cache when the holder allows attribute assignment
        # (Field uses ``object.__setattr__`` to bypass frozen).
        try:
            object.__setattr__(self, "_field_name_map", exact)
            object.__setattr__(self, "_field_name_fold_map", fold)
        except (AttributeError, TypeError):
            pass
        return exact, fold

    def field(
        self,
        name_or_index: str | int | None = None,
        *,
        name: str | None = None,
        index: int | None = None,
        raise_error: bool = True,
    ) -> "Field | None":
        return self.field_by(name_or_index, name=name, index=index, raise_error=raise_error)

    def field_at(
        self,
        index: int,
        raise_error: bool = True,
    ) -> "Field":
        try:
            return self.children_fields[index]
        except IndexError as e:
            if raise_error:
                raise IndexError(
                    f"Index {index} is out of range for array with {len(self.children_fields)} children"
                ) from e
            return None

    def field_by(
        self,
        name_or_index: str | int | None = None,
        *,
        name: str | None = None,
        index: int | None = None,
        raise_error: bool = True,
    ) -> "Field | None":
        inferred_name = name_or_index if isinstance(name_or_index, str) else None
        inferred_index = name_or_index if isinstance(name_or_index, int) else None

        if name is not None and inferred_name is not None and name != inferred_name:
            raise ValueError(
                "Conflicting field name arguments.\n"
                f"- name_or_index provided name={inferred_name!r}\n"
                f"- explicit name provided name={name!r}\n"
                "Use only one name source, or pass matching values."
            )

        if index is not None and inferred_index is not None and index != inferred_index:
            raise ValueError(
                "Conflicting field index arguments.\n"
                f"- name_or_index provided index={inferred_index!r}\n"
                f"- explicit index provided index={index!r}\n"
                "Use only one index source, or pass matching values."
            )

        name = name if name is not None else inferred_name
        index = index if index is not None else inferred_index

        available_fields = self.children_fields
        available_names = [f.name for f in available_fields]

        indexed_field = None
        if index is not None:
            indexed_field = self.field_at(index, raise_error=False)
            if indexed_field is not None:
                if name is None:
                    return indexed_field
                if (
                    indexed_field.name == name
                    or indexed_field.name.casefold() == name.casefold()
                ):
                    return indexed_field

        if name is not None:
            exact, fold = self._field_name_index_maps()
            hit = exact.get(name)
            if hit is None:
                hit = fold.get(name.casefold())
            if hit is not None:
                return available_fields[hit]

        if not raise_error:
            return None

        if name is None and index is None:
            raise KeyError(
                "field_by() requires either a field name or a field index.\n"
                "Examples:\n"
                "  dtype.field_by('price')\n"
                "  dtype.field_by(0)\n"
                "  dtype.field_by(name='price')\n"
                "  dtype.field_by(index=0)"
            )

        if name is None:
            raise KeyError(
                f"No field found at index {index}.\n"
                f"Valid index range: 0..{len(available_fields) - 1}\n"
                f"Available fields: {available_names}"
            )

        suggestions = get_close_matches(name, available_names, n=3, cutoff=0.5)

        if indexed_field is not None:
            suggestion_text = (
                f"\nThe provided index {index} points to field {indexed_field.name!r}, "
                f"which does not match requested name {name!r}."
            )
        else:
            suggestion_text = ""

        if suggestions:
            suggestion_text += f"\nDid you mean: {', '.join(repr(s) for s in suggestions)}?"

        raise KeyError(
            f"No field named {name!r}."
            f"{suggestion_text}\n"
            f"Available fields: {available_names}\n"
            "You can inspect valid names with `.field_names()` "
            "or look up by position with `.field_by(index=...)`."
        )

    @property
    def names(self):
        return [f.name for f in self.children_fields]

    def field_names(self) -> list[str]:
        return [f.name for f in self.children_fields]

    def select(
        self,
        identifiers: "SelectType | Iterable[SelectType]" = (),
        *others: SelectType,
        raise_error: bool = True,
    ) -> list["Field"]:
        """Resolve one or more identifiers into the matching :class:`Field` objects.

        Accepts strings (resolved by name), ints (resolved by index),
        and existing :class:`Field` instances (resolved by ``.name``
        against this container — so callers can copy a field set
        between sibling schemas without first stringifying everything).

        Calling shapes that all work the same way:

        * ``schema.select("price")`` — single identifier.
        * ``schema.select("price", "qty", 0)`` — multiple positionals.
        * ``schema.select(["price", "qty"])`` — single iterable.
        * ``schema.select(other_schema.children_fields)`` — copy
          a sibling's fields by name into this schema.
        * ``schema.select("price", ["qty", "ts"], 0)`` — mixed; each
          positional is itself flattened so iterables and scalars
          can be interleaved.

        :param identifiers:
            First identifier or iterable of identifiers.
        :param others:
            Additional identifiers. Each is flattened the same way
            as the first.
        :param raise_error:
            ``True`` (default) — missing identifiers raise via
            :meth:`field_by` with the same suggestion-rich error
            message used elsewhere. ``False`` — missing identifiers
            yield ``None`` in the returned list, preserving caller
            order.

        :returns:
            A list of :class:`Field` (or ``Field | None`` when
            ``raise_error=False``), one entry per resolved identifier
            in caller order. Duplicates in the input produce
            duplicates in the output — this is intentional, since
            ``select`` is the natural place to express a projection
            and projections sometimes repeat columns.

        :raises KeyError:
            With suggestions, when ``raise_error`` is True and an
            identifier doesn't resolve.
        :raises TypeError:
            When an identifier is not a ``str`` / ``int`` / ``Field``.
        """
        # Flatten ``identifiers`` plus all positional ``others`` into
        # a single ordered list. We accept both a scalar and an
        # iterable in the first slot, plus any number of positional
        # tail args, so the call sites listed in the docstring all
        # converge on the same internal representation.
        flat: list[SelectType] = []

        # ``identifiers`` may itself be a single SelectType (str / int
        # / Field) or an iterable of them. Strings are iterable as
        # characters, so we have to special-case them — same for the
        # imported Field class which we detect via duck-typing on
        # ``.name`` to avoid the runtime import cycle.
        self._extend_select_targets(flat, identifiers)
        for other in others:
            self._extend_select_targets(flat, other)

        # Resolve each entry through ``field_by``. The strict branch
        # propagates ``field_by``'s rich error path (with suggestions);
        # the lenient branch yields ``None`` for misses so callers
        # can post-process the gaps.
        resolved: list["Field"] = []
        for ident in flat:
            field = self._resolve_select_target(ident, raise_error=raise_error)
            resolved.append(field)
        return resolved

    @staticmethod
    def _extend_select_targets(
        out: list["SelectType"],
        item: "SelectType | Iterable[SelectType]",
    ) -> None:
        """Append *item* to *out*, flattening one level of iterable.

        Treats strings and :class:`Field` instances as scalars
        (they're not "containers of identifiers" even though the
        former is iterable). Anything else with ``__iter__`` is
        flattened. Mappings are explicitly handled as scalars too —
        iterating a dict yields its keys, which would silently change
        the meaning of ``select(some_dict)`` in surprising ways.
        """
        # Scalar fast paths. Strings come first because every other
        # branch would treat them as iterables of characters.
        if isinstance(item, (str, int)):
            out.append(item)
            return
        # Field detection by duck-typing on ``.name`` rather than
        # isinstance — avoids the runtime import that the
        # TYPE_CHECKING block above already pushed off to type-check
        # time only.
        if hasattr(item, "name") and not isinstance(item, (list, tuple, set, frozenset)):
            # Anything carrying a ``.name`` and not obviously a
            # collection: treat as a scalar Field-like.
            out.append(item)  # type: ignore[arg-type]
            return
        if isinstance(item, Mapping):
            # A mapping isn't a meaningful identifier shape — flag it
            # explicitly rather than letting the iterable path silently
            # walk its keys.
            raise TypeError(
                f"select() does not accept a mapping as an identifier; "
                f"got {type(item).__name__}. Pass the keys or values "
                "explicitly if that's what you meant."
            )
        # Generic iterable: flatten one level. We don't recurse —
        # ``select(["a", ["b", "c"]])`` is rejected so the call shape
        # stays predictable.
        try:
            iterator = iter(item)
        except TypeError:
            raise TypeError(
                f"select() expected str / int / Field / Iterable of those, "
                f"got {type(item).__name__}: {item!r}"
            ) from None
        for sub in iterator:
            if isinstance(sub, (str, int)) or hasattr(sub, "name"):
                out.append(sub)
            else:
                raise TypeError(
                    f"select() identifiers must be str / int / Field; "
                    f"got nested {type(sub).__name__}: {sub!r}. "
                    "Flatten the input before passing it in — select() "
                    "only flattens one level so the call shape stays "
                    "predictable."
                )

    def _resolve_select_target(
        self,
        ident: "SelectType",
        *,
        raise_error: bool,
    ) -> "Field | None":
        """Resolve a single ``SelectType`` into a :class:`Field`.

        Strings → ``field_by(name=ident)``. Ints → ``field_by(index=ident)``.
        :class:`Field` instances → ``field_by(name=ident.name)`` so
        the lookup happens against ``self`` (the right behaviour when
        callers thread fields between sister schemas).
        """
        if isinstance(ident, str):
            return self.field_by(name=ident, raise_error=raise_error)
        if isinstance(ident, int):
            return self.field_by(index=ident, raise_error=raise_error)
        # Field-shaped — duck-type via ``.name``.
        name = getattr(ident, "name", None)
        if name is None:
            if raise_error:
                raise TypeError(
                    f"select() identifier of type {type(ident).__name__} "
                    "has no usable .name attribute."
                )
            return None
        return self.field_by(name=name, raise_error=raise_error)