# yggdrasil.pyutils.serde.py

import gzip
import io
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Iterable, Generator

from yggdrasil.io.dynamic_buffer import DynamicBuffer, DynamicBufferConfig

__all__ = [
    "ObjectSerdeProtocol",
    "ObjectSerde",
    "ObjectSerdeCompression",
    "ObjectSerdeFormat"
]


def _import_module(
    module_name: str,
    *,
    install: bool = True,
    pip_name: str | None = None,
    upgrade: bool = False,
):
    from .pyenv.environment import PyEnv

    return PyEnv.current().import_module(
        module_name=module_name,
        install=install,
        pip_name=pip_name,
        upgrade=upgrade
    )


class ObjectSerdeProtocol(int, Enum):
    """A string representing the encoding used for an ObjectSerde."""
    RAW = 0
    DILL = 1
    PICKLE = 2
    JSON = 3
    ARROW = 4
    PANDAS = 5
    POLARS = 6


class ObjectSerdeFormat(int, Enum):
    """A string representing the format used for an ObjectSerde."""
    BINARY = 0
    PICKLE = 1
    PARQUET = 2


class ObjectSerdeCompression(int, Enum):
    """A string representing the compression used for an ObjectSerde."""
    NONE = 0
    GZIP = 1
    LZ4 = 2
    ZSTD = 3


try:
    import dill
    DILL_MODULE = dill
except ImportError:
    DILL_MODULE = None


@dataclass(frozen=True, slots=True)
class ObjectSerde:
    """A wrapper for an object that has been encoded to bytes."""
    protocol: ObjectSerdeProtocol
    format: ObjectSerdeFormat
    compression: ObjectSerdeCompression
    io: DynamicBuffer | None = None
    items: Iterable["ObjectSerde"] | None = None

    @staticmethod
    def full_namespace(obj: Any, *, fallback: str = "") -> str:
        cls = obj if isinstance(obj, type) else getattr(obj, "__class__", None)
        if cls is None:
            return fallback

        mod = getattr(cls, "__module__", None)
        qual = getattr(cls, "__qualname__", None) or getattr(cls, "__name__", None)
        if not mod or not qual:
            return fallback

        return f"{mod}.{qual}"

    # ---------------------------
    # ENCODE (unchanged)
    # ---------------------------
    @classmethod
    def encode(
        cls,
        obj: Any,
        *,
        buffer_config: DynamicBufferConfig | None = None,
        byref: bool = False,
        recurse: bool = False,
    ) -> "ObjectSerde":
        buffer_config = DynamicBufferConfig() if buffer_config is None else buffer_config
        namespace = cls.full_namespace(obj)

        if obj is None:
            return cls(
                protocol=ObjectSerdeProtocol.NULL
            )

        elif isinstance(obj, str):
            return cls(
                str=obj
            )

        elif isinstance(obj, bytes):
            compression = None

            if len(obj) > buffer_config.spill_bytes:
                compression = ObjectSerdeCompression.GZIP
                obj = gzip.compress(obj)

            buf = DynamicBuffer()
            buf.write(obj)
            buf.seek(0)

            return cls(
                protocol=ObjectSerdeProtocol.BYTES,
                io=buf,
                compression=compression
            )

        if namespace.startswith("pandas."):
            return cls.pandas_encode(
                obj,
                buffer_config=buffer_config,
                byref=byref, recurse=recurse
            )

        if namespace.startswith("polars."):
            return cls.polars_encode(
                obj,
                buffer_config=buffer_config,
                byref=byref, recurse=recurse
            )

        if isinstance(obj, (list, tuple, set, Generator)):
            return cls(
                protocol=ObjectSerdeProtocol.ARRAY,
                namespace=namespace,
                items=[
                    cls.encode(
                        _,
                        buffer_config=buffer_config, byref=byref, recurse=recurse
                    )
                    for _ in obj
                ]
            )

        elif isinstance(obj, dict):
            return cls(
                protocol=ObjectSerdeProtocol.MAP,
                namespace=namespace,
                items=[
                    (
                        cls.encode(
                            k,
                            buffer_config=buffer_config, byref=byref, recurse=recurse
                        ),
                        cls.encode(
                            v,
                            buffer_config=buffer_config, byref=byref, recurse=recurse
                        ),
                    )
                    for k, v in obj.items()
                ]
            )

        return cls.dill_encode(
            obj,
            buffer_config=buffer_config,
            byref=byref, recurse=recurse
        )

    @classmethod
    def dill_encode(
        cls,
        obj: object,
        *,
        buffer_config: DynamicBufferConfig | None = None,
        byref: bool = False,
        recurse: bool = False,
        namespace: str | None = None,
    ) -> "ObjectSerde":
        global DILL_MODULE

        if DILL_MODULE is None:
            DILL_MODULE = _import_module(module_name="dill", pip_name="dill")

        buffer = DynamicBuffer(config=buffer_config)
        DILL_MODULE.dump(obj, buffer, byref=byref, recurse=recurse)
        buffer.seek(0)

        if not namespace:
            namespace = cls.full_namespace(obj)

        return cls(
            protocol=ObjectSerdeProtocol.DILL,
            format=ObjectSerdeFormat.BINARY,
            namespace=namespace,
            io=buffer,
        )

    @classmethod
    def pandas_encode(
        cls,
        obj: object,
        *,
        buffer_config: DynamicBufferConfig | None = None,
        byref: bool = False,
        recurse: bool = False,
        namespace: str | None = None,
    ) -> "ObjectSerde":
        try:
            import pandas as pd
        except ImportError:
            pd = _import_module(module_name="pandas", pip_name="pandas")

        buffer = DynamicBuffer(config=buffer_config)

        if not namespace:
            namespace = cls.full_namespace(obj)

        if isinstance(obj, pd.DataFrame):
            compression = ObjectSerdeCompression.ZSTD

            try:
                obj.to_parquet(buffer, compression=compression.value)
                fmt = ObjectSerdeFormat.PARQUET
            except:
                obj.to_pickle(buffer, compression=compression.value)
                fmt = ObjectSerdeFormat.PICKLE

            buffer.seek(0)

            return cls(
                protocol=ObjectSerdeProtocol.PANDAS,
                format=fmt,
                io=buffer,
                namespace=namespace,
                compression=compression,
            )

        return cls.dill_encode(
            obj,
            buffer_config=buffer_config, byref=byref, recurse=recurse,
            namespace=namespace
        )

    @classmethod
    def polars_encode(
        cls,
        obj: object,
        *,
        buffer_config: DynamicBufferConfig | None = None,
        byref: bool = False,
        recurse: bool = False,
        namespace: Optional[str] = None,
    ) -> "ObjectSerde":
        try:
            import polars as pl
        except ImportError:
            pl = _import_module(module_name="polars", pip_name="polars")

        if not namespace:
            namespace = cls.full_namespace(obj)

        buffer = DynamicBuffer()

        if isinstance(obj, pl.DataFrame):
            compression = ObjectSerdeCompression.ZSTD
            obj.write_parquet(buffer, compression="zstd")
            buffer.seek(0)

            return cls(
                protocol=ObjectSerdeProtocol.POLARS,
                format=ObjectSerdeFormat.PARQUET,
                io=buffer,
                namespace=namespace,
                compression=compression,
            )

        if isinstance(obj, pl.LazyFrame):
            compression = ObjectSerdeCompression.ZSTD
            obj.collect().write_parquet(buffer, compression="zstd")
            buffer.seek(0)

            return cls(
                protocol=ObjectSerdeProtocol.POLARS,
                format=ObjectSerdeFormat.PARQUET,
                io=buffer,
                namespace=namespace,
                compression=compression,
            )

        return cls.dill_encode(
            obj, buffer_config=buffer_config,
            byref=byref, recurse=recurse, namespace=namespace
        )

    # ---------------------------
    # DECODE (refactored)
    # ---------------------------
    def decode(self, *, clear: bool = True) -> object:
        if self.protocol == ObjectSerdeProtocol.NULL:
            return None

        elif self.protocol == ObjectSerdeProtocol.STR:
            return self.str

        elif self.protocol == ObjectSerdeProtocol.DILL:
            return self.dill_decode(clear=clear)

        elif self.protocol == ObjectSerdeProtocol.PANDAS:
            return self.pandas_decode(clear=clear)

        elif self.protocol == ObjectSerdeProtocol.POLARS:
            return self.polars_decode(clear=clear)

        raise ValueError(
            f"Unsupported serde protocol {self.protocol}"
        )

    def cleanup_io(self) -> None:
        """
        Best-effort release of the underlying buffer memory.
        Works even though ObjectSerde is frozen because we're mutating the buffer object itself.
        """
        # Try to close first (some buffers release memory on close)
        try:
            self.io.close()
            return
        except Exception:
            pass

        # If close isn't supported / doesn't free, truncate aggressively
        try:
            self.io.seek(0)
        except Exception:
            pass

        try:
            self.io.truncate(0)
        except Exception:
            pass

    def dill_decode(self, *, clear: bool = True) -> object:
        global DILL_MODULE

        pos = self.io.tell()
        try:
            raw = self.io.read()

            if self.compression is None:
                payload = raw
            elif self.compression == ObjectSerdeCompression.GZIP:
                payload = gzip.decompress(raw)
            else:
                raise ValueError(f"Unsupported compression: {self.compression}")

            if DILL_MODULE is None:
                DILL_MODULE = _import_module(module_name="dill", pip_name="dill", install=True)

            return DILL_MODULE.load(io.BytesIO(payload))
        finally:
            if clear:
                self.cleanup_io()
            else:
                self.io.seek(pos)

    def pandas_decode(self, *, clear: bool = True) -> object:
        try:
            import pandas as pd
        except ImportError:
            pd = _import_module(module_name="pandas", pip_name="pandas", install=True)

        pos = self.io.tell()
        try:
            if self.format == ObjectSerdeFormat.PARQUET:
                return pd.read_parquet(self.io)
            elif self.format == ObjectSerdeFormat.PICKLE:
                cpr = self.compression if self.compression is None else self.compression.value
                return pd.read_pickle(self.io, compression=cpr)
            else:
                raise ValueError(

                )
        finally:
            if clear:
                self.cleanup_io()
            else:
                self.io.seek(pos)

    def polars_decode(self, *, clear: bool = True) -> object:
        try:
            import polars as pl
        except ImportError:
            pl = _import_module(module_name="polars", pip_name="polars", install=True)

        assert self.format == ObjectSerdeFormat.PARQUET

        pos = self.io.tell()
        try:
            return pl.read_parquet(self.io)
        finally:
            if clear:
                self.cleanup_io()
            else:
                self.io.seek(pos)
