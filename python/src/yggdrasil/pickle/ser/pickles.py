from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "PickleSerialized",
    "StdlibPickleSerialized",
    "DillSerialized",
    "CloudPickleSerialized",
]


@dataclass(frozen=True, slots=True)
class PickleSerialized(Serialized[Any]):
    """Base class for pickle-family serialized Python objects."""

    TAG: ClassVar[int]

    def loads(self) -> Any:
        raise NotImplementedError

    @property
    def value(self) -> Any:
        return self.loads()

    def as_python(self) -> Any:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        try:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return Serialized.build(tag=Tags.PICKLE, data=data, metadata=metadata, codec=codec)
        except Exception:
            pass

        try:
            import dill  # type: ignore

            data = dill.dumps(obj)
            return Serialized.build(tag=Tags.DILL, data=data, metadata=metadata, codec=codec)
        except Exception:
            pass

        try:
            import cloudpickle  # type: ignore

            data = cloudpickle.dumps(obj)
            return Serialized.build(tag=Tags.CLOUDPICKLE, data=data, metadata=metadata, codec=codec)
        except Exception:
            return None

@dataclass(frozen=True, slots=True)
class StdlibPickleSerialized(PickleSerialized):
    TAG: ClassVar[int] = Tags.PICKLE

    def loads(self) -> Any:
        return pickle.loads(self.decode())


@dataclass(frozen=True, slots=True)
class DillSerialized(PickleSerialized):
    TAG: ClassVar[int] = Tags.DILL

    def loads(self) -> Any:
        try:
            import dill
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("dill is required to load DILL payloads") from exc
        return dill.loads(self.decode())


@dataclass(frozen=True, slots=True)
class CloudPickleSerialized(PickleSerialized):
    TAG: ClassVar[int] = Tags.CLOUDPICKLE

    def loads(self) -> Any:
        try:
            import cloudpickle
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("cloudpickle is required to load CLOUDPICKLE payloads") from exc
        return cloudpickle.loads(self.decode())


for cls in PickleSerialized.__subclasses__():
    Tags.register_class(cls, tag=cls.TAG)