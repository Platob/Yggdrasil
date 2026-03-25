"""Base options dataclass for :class:`~yggdrasil.io.buffer.media_io.MediaIO`.

:class:`MediaOptions` collects every parameter that can influence how an
Arrow table is read from / written to a :class:`~yggdrasil.io.buffer.BytesIO`
buffer. Format-specific subclasses extend it with codec-level knobs, but the
fields defined here are shared across all formats.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional, Sequence

from ..enums.save_mode import SaveMode

__all__ = ["MediaOptions"]

_MISSING = object()

_ALLOWED_COMPRESSION = {"auto", "none", "off", "zstd", "snappy", "gzip", "lz4"}


@dataclass
class MediaOptions:
    """Base options shared by all :class:`MediaIO` subclasses."""

    # global read properties
    columns: Optional[Sequence[str]] = None
    use_threads: bool = True
    ignore_empty: bool = True
    lazy: bool = False
    raise_error: bool = True

    # global read / write properties
    batch_size: int = 0

    # global write properties
    mode: SaveMode = SaveMode.AUTO
    match_by: Sequence[str] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate all fields in-place."""
        self.columns = self._normalize_columns(self.columns)
        self.use_threads = self._validate_bool("use_threads", self.use_threads)
        self.ignore_empty = self._validate_bool("ignore_empty", self.ignore_empty)
        self.lazy = self._validate_bool("lazy", self.lazy)
        self.raise_error = self._validate_bool("raise_error", self.raise_error)
        self.batch_size = self._normalize_batch_size(self.batch_size)
        self.mode = SaveMode.parse(self.mode, default=SaveMode.AUTO)
        self.match_by = self._normalize_match_by(self.match_by)

        self._validate_subclass_fields()

    @classmethod
    def check_parameters(
        cls,
        options: MediaOptions | None = None,
        *,
        mode: SaveMode | str | None | Any = _MISSING,
        match_by: Sequence[str] | str | None | Any = _MISSING,
        columns: Optional[Sequence[str]] | Any = _MISSING,
        use_threads: bool | Any = _MISSING,
        ignore_empty: bool | Any = _MISSING,
        lazy: bool | Any = _MISSING,
        raise_error: bool | Any = _MISSING,
        batch_size: int | None | Any = _MISSING,
        **kwargs: Any,
    ) -> MediaOptions:
        """Merge explicit overrides into an options instance and validate."""
        base = cls._coerce_options(options)
        updates = cls._collect_updates(
            mode=mode,
            match_by=match_by,
            columns=columns,
            use_threads=use_threads,
            ignore_empty=ignore_empty,
            lazy=lazy,
            raise_error=raise_error,
            batch_size=batch_size,
            **kwargs,
        )
        cls._validate_known_fields(updates)

        for key, value in updates.items():
            setattr(base, key, value)

        base.__post_init__()
        return base

    @classmethod
    def _coerce_options(cls, options: MediaOptions | None) -> MediaOptions:
        """Return an instance of *cls* copied from *options* when needed."""
        if options is None:
            return cls()

        if not isinstance(options, MediaOptions):
            raise TypeError(
                f"options must be a MediaOptions instance or None, got {type(options)!r}"
            )

        if isinstance(options, cls):
            return options

        out = cls()
        for f in fields(cls):
            if hasattr(options, f.name):
                setattr(out, f.name, getattr(options, f.name))
        return out

    @classmethod
    def _collect_updates(cls, **kwargs: Any) -> dict[str, Any]:
        """Drop sentinel values and keep only explicitly supplied overrides."""
        return {key: value for key, value in kwargs.items() if value is not _MISSING}

    @classmethod
    def _validate_known_fields(cls, updates: dict[str, Any]) -> None:
        """Reject unknown option names for this class."""
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(key for key in updates if key not in allowed)
        if unknown:
            raise TypeError(
                f"{cls.__name__}.check_parameters got unexpected parameter(s): "
                f"{', '.join(unknown)}"
            )

    @staticmethod
    def _validate_bool(name: str, value: Any) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool, got {type(value).__name__}")
        return value

    @staticmethod
    def _normalize_columns(value: Optional[Sequence[str]]) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            raise TypeError("columns must be a sequence of strings, not a single string/bytes")

        try:
            items = list(value)
        except TypeError as e:
            raise TypeError("columns must be a sequence of strings") from e

        if not all(isinstance(item, str) for item in items):
            bad = [type(item).__name__ for item in items if not isinstance(item, str)]
            raise TypeError(f"columns must contain only str, found: {bad[:3]}")

        return items

    @staticmethod
    def _normalize_batch_size(value: int | None) -> int:
        if value is None:
            return 0
        if not isinstance(value, int):
            raise TypeError(f"batch_size must be int or None, got {type(value).__name__}")
        return max(0, value)

    @staticmethod
    def _normalize_match_by(value: Sequence[str] | str | None) -> tuple[str, ...] | None:
        if value is None:
            return None

        if isinstance(value, bytes):
            raise TypeError("match_by must be str or a sequence of str, not bytes")

        if isinstance(value, str):
            return (value,)

        try:
            items = list(value)
        except TypeError as e:
            raise TypeError("match_by must be a string or a sequence of strings") from e

        if not items:
            return None

        if not all(isinstance(item, str) for item in items):
            bad = [type(item).__name__ for item in items if not isinstance(item, str)]
            raise TypeError(f"match_by must contain only str, found: {bad[:3]}")

        return tuple(items)

    def _validate_subclass_fields(self) -> None:
        """Validate optional subclass fields when present."""
        if hasattr(self, "compression"):
            self._validate_compression(getattr(self, "compression"))

        if hasattr(self, "compression_level"):
            self._validate_compression_level(getattr(self, "compression_level"))

        if hasattr(self, "skip_rows"):
            self._validate_skip_rows(getattr(self, "skip_rows"))

        if hasattr(self, "zip_compression"):
            self._validate_zip_compression(getattr(self, "zip_compression"))

    @staticmethod
    def _validate_compression(value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            raise TypeError(f"compression must be str|None, got {type(value).__name__}")
        if value not in _ALLOWED_COMPRESSION:
            raise ValueError(
                f"compression must be one of {sorted(_ALLOWED_COMPRESSION)}, got {value!r}"
            )

    @staticmethod
    def _validate_compression_level(value: Any) -> None:
        if value is not None and not isinstance(value, int):
            raise TypeError(f"compression_level must be int|None, got {type(value).__name__}")

    @staticmethod
    def _validate_skip_rows(value: Any) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError("skip_rows must be a non-negative int")

    @staticmethod
    def _validate_zip_compression(value: Any) -> None:
        if not isinstance(value, int):
            raise TypeError(f"zip_compression must be int, got {type(value).__name__}")