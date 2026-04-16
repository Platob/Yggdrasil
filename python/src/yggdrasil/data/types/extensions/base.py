from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields as dc_fields
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.io import SaveMode

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst
    from yggdrasil.data.data_field import Field


__all__ = [
    "ExtensionType",
    "get_extension_type",
    "get_extension_registry",
]


LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension type registry
# ---------------------------------------------------------------------------
# Maps extension_name → ExtensionType subclass.
# Populated automatically by __init_subclass__ when a concrete subclass
# defines `extension_name`.
_EXTENSION_REGISTRY: dict[str, type[ExtensionType]] = {}


def get_extension_type(name: str) -> type["ExtensionType"]:
    """Look up a registered ExtensionType subclass by name.

    Raises KeyError with useful diagnostics when the name isn't found.
    """
    cls = _EXTENSION_REGISTRY.get(name)
    if cls is not None:
        return cls

    available = sorted(_EXTENSION_REGISTRY.keys()) or ["(none registered)"]
    raise KeyError(
        f"No extension type registered under the name {name!r}. "
        f"Available extension types: {available}. "
        "Make sure the module that defines this extension type has been imported."
    )


def get_extension_registry() -> dict[str, type["ExtensionType"]]:
    """Return a snapshot of the current extension type registry."""
    return dict(_EXTENSION_REGISTRY)


# ---------------------------------------------------------------------------
# PyArrow extension type bridge
# ---------------------------------------------------------------------------
# Each concrete ExtensionType subclass gets a companion pa.ExtensionType
# that delegates serialization to the Yggdrasil side.  This keeps the
# frozen-dataclass design intact while still playing nicely with Arrow IPC.


def _make_arrow_ext_class(
    ext_cls: type["ExtensionType"],
) -> type[pa.ExtensionType]:
    """Build a pa.ExtensionType adapter class for *ext_cls*.

    The generated class stores the extension_name and delegates
    serialize/deserialize to the Yggdrasil ExtensionType subclass.
    """

    class _ArrowBridge(pa.ExtensionType):
        _ygg_cls: ClassVar[type[ExtensionType]] = ext_cls

        def __init__(self, storage_type: pa.DataType, metadata_bytes: bytes = b""):
            self._metadata_bytes = metadata_bytes
            # pa.ExtensionType.__init__ *must* be called with storage_type and
            # extension_name.
            pa.ExtensionType.__init__(self, storage_type, ext_cls.extension_name)

        def __arrow_ext_serialize__(self) -> bytes:
            return self._metadata_bytes

        @classmethod
        def __arrow_ext_deserialize__(
            cls, storage_type: pa.DataType, serialized: bytes
        ) -> "_ArrowBridge":
            return cls(storage_type, metadata_bytes=serialized)

    # Give it a unique class name so repr/debugging is nicer.
    _ArrowBridge.__name__ = f"_Arrow_{ext_cls.__name__}"
    _ArrowBridge.__qualname__ = f"_Arrow_{ext_cls.__name__}"
    return _ArrowBridge


# ---------------------------------------------------------------------------
# ExtensionType base class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtensionType(DataType):
    """Base class for user-defined extension types in Yggdrasil.

    Subclasses must define two class-level attributes:
        extension_name : str
            Globally unique name registered with PyArrow.
        storage_type   : pa.DataType
            The underlying Arrow type used for physical storage.

    Subclasses can add dataclass fields for type-level parameters
    (e.g. precision, timezone, item_type, etc.).  These are automatically
    included in serialization via ``serialize_metadata`` / ``deserialize_metadata``.

    Example::

        @dataclass(frozen=True)
        class UuidType(ExtensionType):
            extension_name: ClassVar[str] = "yggdrasil.uuid"
            storage_type: ClassVar[pa.DataType] = pa.binary(16)

    The companion PyArrow extension type is created and registered
    automatically when the subclass is defined.
    """

    # Subclasses must override these as ClassVar:
    extension_name: ClassVar[str] = ""
    storage_type: ClassVar[pa.DataType] = pa.null()

    # Filled in by __init_subclass__ — the pa.ExtensionType adapter.
    _arrow_bridge_cls: ClassVar[type[pa.ExtensionType] | None] = None

    # ------------------------------------------------------------------
    # Subclass hook — auto-register on definition
    # ------------------------------------------------------------------
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        name = cls.__dict__.get("extension_name") or getattr(cls, "extension_name", "")
        if not name:
            # Abstract intermediate classes (no extension_name) are fine — skip.
            return

        # Guard against collisions.
        existing = _EXTENSION_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise TypeError(
                f"Extension name collision: {name!r} is already registered "
                f"by {existing.__qualname__}. "
                f"Pick a different extension_name for {cls.__qualname__}, "
                f"or unregister the existing one first."
            )

        # Build the Arrow bridge class and register it with PyArrow.
        bridge = _make_arrow_ext_class(cls)
        cls._arrow_bridge_cls = bridge

        try:
            pa.register_extension_type(bridge(cls.storage_type))
        except pa.ArrowKeyError:
            # Already registered in PyArrow (e.g. module re-import). That's fine —
            # we still update our own registry so the Yggdrasil side stays current.
            LOGGER.debug(
                "PyArrow already has extension type %r registered; "
                "skipping re-registration.",
                name,
            )

        _EXTENSION_REGISTRY[name] = cls
        LOGGER.debug("Registered extension type %r → %s", name, cls.__qualname__)

    # ------------------------------------------------------------------
    # DataType protocol — type_id
    # ------------------------------------------------------------------
    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.EXTENSION

    # ------------------------------------------------------------------
    # Children fields — delegate to storage type
    # ------------------------------------------------------------------
    @property
    def children_fields(self) -> list["Field"]:
        # Extension types carry no children of their own; the storage type
        # might (e.g. struct-backed extensions), but we expose none at the
        # extension level.  Subclasses can override if needed.
        return []

    # ------------------------------------------------------------------
    # Serialization — JSON by default, override for custom wire format
    # ------------------------------------------------------------------
    def _own_field_values(self) -> dict[str, Any]:
        """Collect dataclass instance fields (NOT ClassVars) as a plain dict.

        This drives the default serialize/deserialize round-trip.
        """
        out: dict[str, Any] = {}
        for f in dc_fields(self):
            value = getattr(self, f.name)
            # Keep it JSON-friendly — basic scalars only.
            out[f.name] = value
        return out

    def serialize_metadata(self) -> bytes:
        """Serialize type-level metadata to bytes for Arrow IPC.

        Default implementation: JSON-encode all dataclass fields.
        Override this if you need a tighter wire format.
        """
        payload = self._own_field_values()
        if not payload:
            return b""
        return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    @classmethod
    def deserialize_metadata(cls, serialized: bytes) -> "ExtensionType":
        """Reconstruct an instance from serialized metadata bytes.

        Default implementation: JSON-decode and pass as kwargs.
        """
        if not serialized:
            return cls()
        payload = json.loads(serialized)
        return cls(**payload)

    # ------------------------------------------------------------------
    # Arrow conversion
    # ------------------------------------------------------------------
    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        if not isinstance(dtype, pa.ExtensionType):
            return False

        # If called on the base ExtensionType, check the registry.
        if cls is ExtensionType:
            return dtype.extension_name in _EXTENSION_REGISTRY

        # Called on a concrete subclass — match by name.
        return getattr(dtype, "extension_name", None) == cls.extension_name

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "ExtensionType":
        if not isinstance(dtype, pa.ExtensionType):
            raise TypeError(
                f"Expected a PyArrow ExtensionType, got {type(dtype).__name__}: {dtype!r}. "
                "Use DataType.from_arrow_type() for standard Arrow types."
            )

        name = dtype.extension_name
        if cls is ExtensionType:
            target_cls = _EXTENSION_REGISTRY.get(name)
            if target_cls is None:
                available = sorted(_EXTENSION_REGISTRY.keys()) or ["(none registered)"]
                raise TypeError(
                    f"No Yggdrasil extension type registered for Arrow extension {name!r}. "
                    f"Known extension types: {available}. "
                    "Make sure the module defining this extension has been imported."
                )
        else:
            if name != cls.extension_name:
                raise TypeError(
                    f"Arrow extension name {name!r} does not match "
                    f"{cls.__qualname__}.extension_name={cls.extension_name!r}."
                )
            target_cls = cls

        serialized = dtype.__arrow_ext_serialize__()
        return target_cls.deserialize_metadata(serialized)

    def to_arrow(self) -> pa.DataType:
        bridge_cls = self._arrow_bridge_cls
        if bridge_cls is None:
            raise TypeError(
                f"{type(self).__qualname__} has no Arrow bridge class. "
                "Make sure it defines 'extension_name' and 'storage_type' as ClassVars."
            )
        return bridge_cls(type(self).storage_type, metadata_bytes=self.serialize_metadata())

    # ------------------------------------------------------------------
    # Polars conversion — fall back to storage type
    # ------------------------------------------------------------------
    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        # Polars has no extension type concept — we never claim to handle
        # a Polars type directly. Subclasses can override if they register
        # a custom Polars dtype some day.
        return False

    def to_polars(self) -> "polars.DataType":
        """Convert to the nearest Polars equivalent (the storage type).

        Polars doesn't have user-defined extension types, so we fall through
        to the physical storage representation.  This is lossy — the extension
        semantics are lost — but it keeps the data flowing.
        """
        from yggdrasil.data.types.base import DataType as _DT
        storage_dtype = _DT.from_arrow_type(type(self).storage_type)
        return storage_dtype.to_polars()

    # ------------------------------------------------------------------
    # Spark conversion — fall back to storage type
    # ------------------------------------------------------------------
    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        # Same deal as Polars — no native extension type support.
        return False

    def to_spark(self) -> "pst.DataType":
        """Convert to the nearest Spark SQL equivalent (the storage type)."""
        from yggdrasil.data.types.base import DataType as _DT
        storage_dtype = _DT.from_arrow_type(type(self).storage_type)
        return storage_dtype.to_spark()

    # ------------------------------------------------------------------
    # Databricks DDL — fall back to storage type
    # ------------------------------------------------------------------
    def to_databricks_ddl(self) -> str:
        """Produce a Databricks DDL string from the storage type.

        Extension semantics are not representable in DDL, so this returns
        the DDL for the underlying physical storage.
        """
        from yggdrasil.data.types.base import DataType as _DT
        storage_dtype = _DT.from_arrow_type(type(self).storage_type)
        return storage_dtype.to_databricks_ddl()

    # ------------------------------------------------------------------
    # Dict round-trip
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        type_id = value.get("id")
        if type_id == int(DataTypeId.EXTENSION):
            # If we're the base class, accept any EXTENSION id.
            if cls is ExtensionType:
                return True
            # Concrete subclass — also check extension_name.
            return value.get("extension_name") == cls.extension_name
        return False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExtensionType":
        ext_name = value.get("extension_name", "")

        if cls is ExtensionType:
            if ext_name:
                target_cls = _EXTENSION_REGISTRY.get(ext_name)
                if target_cls is None:
                    available = sorted(_EXTENSION_REGISTRY.keys()) or ["(none registered)"]
                    raise ValueError(
                        f"No extension type registered under name {ext_name!r}. "
                        f"Known extension types: {available}."
                    )
                return target_cls.from_dict(value)
            # No extension_name in payload — cannot reconstruct.
            raise ValueError(
                "Cannot deserialize an ExtensionType from a dict without "
                "'extension_name'. Provide the extension_name key so the "
                "correct subclass can be resolved."
            )

        # Concrete subclass — pull kwargs from the dict.
        kwargs: dict[str, Any] = {}
        own_fields = {f.name for f in dc_fields(cls)}
        for key, val in value.items():
            if key in own_fields:
                kwargs[key] = val
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "id": int(DataTypeId.EXTENSION),
            "name": DataTypeId.EXTENSION.name,
            "extension_name": type(self).extension_name,
        }
        base.update(self._own_field_values())
        return base

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "DataType":
        if not isinstance(other, ExtensionType):
            return self

        # Same concrete class → keep self (frozen, no parameter merging).
        if type(self) is type(other):
            return self

        # Different extension types — can't merge meaningfully.
        # Fall back to self; the caller can detect via type checks.
        return self

    # ------------------------------------------------------------------
    # Casting helpers
    # ------------------------------------------------------------------
    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: Any,
    ) -> pa.Array:
        """Cast an Arrow array into this extension type.

        Unwraps extension arrays to their storage first, casts through
        the storage type, then wraps back into the extension type.
        """
        # Unwrap source if it's an extension array.
        src = array
        if isinstance(src.type, pa.ExtensionType):
            src = src.storage

        target_arrow = self.to_arrow()
        storage_target = type(self).storage_type

        # Cast to storage type via the normal path.
        import pyarrow.compute as pc
        casted = pc.cast(src, target_type=storage_target, safe=getattr(options, "safe", True))

        # Wrap into the extension type.
        return pa.ExtensionArray.from_storage(target_arrow, casted)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        params = self._own_field_values()
        if params:
            param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
            return f"{type(self).__name__}({param_str})"
        return f"{type(self).__name__}()"

    # ------------------------------------------------------------------
    # Default values
    # ------------------------------------------------------------------
    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        raise NotImplementedError(
            f"{type(self).__qualname__}.default_pyobj(nullable=False) is not implemented. "
            "Override default_pyobj() in your extension type to provide a non-null default."
        )

    def default_arrow_scalar(self, nullable: bool = True) -> pa.Scalar:
        if nullable:
            return pa.scalar(None, type=type(self).storage_type)
        return pa.scalar(self.default_pyobj(nullable=False), type=type(self).storage_type)
