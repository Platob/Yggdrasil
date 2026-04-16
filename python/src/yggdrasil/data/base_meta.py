from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING
from difflib import get_close_matches
from typing import Any, AnyStr, Mapping, TYPE_CHECKING

import yggdrasil.pickle.json as json_module
from yggdrasil.data.constants import DBX_META_PREFIX, TAG_PREFIX, DEFAULT_VALUE_KEY

if TYPE_CHECKING:
    from .data_field import Field


__all__ = [
    "BaseMetadata",
    "BaseChildrenFields",
    "_merge_metadata_and_tags",
    "_normalize_metadata",
    "_to_bytes",
]


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
        normalized[DEFAULT_VALUE_KEY] = json_module.dumps(
            default_value, safe=False, to_bytes=True,
            ensure_ascii=False, separators=(",", ":")
        )

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

    def _tag_flag(self, key: bytes | str) -> bool:
        value = self._tag_value(key)
        return bool(value and value.startswith(b"t"))

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
    def databricks_metadata(self) -> dict[bytes, bytes]:
        return self._prefixed_metadata(DBX_META_PREFIX)

    @databricks_metadata.setter
    def databricks_metadata(
        self,
        values: Mapping[bytes | str, bytes | str | object] | None,
    ) -> None:
        self._update_prefixed_metadata(DBX_META_PREFIX, values)


class BaseChildrenFields(ABC):

    def __getitem__(self, item):
        return self.field(item)

    def __iter__(self):
        return iter(self.children_fields)

    @property
    @abstractmethod
    def children_fields(self) -> list["Field"]:
        ...

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
            for field in available_fields:
                if field.name == name:
                    return field

            folded = name.casefold()
            for field in available_fields:
                if field.name.casefold() == folded:
                    return field

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