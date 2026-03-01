from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Sequence, Any

from ..enums.save_mode import SaveMode

__all__ = [
    "MediaOptions",
]

_MISSING = object()

_ALLOWED_COMPRESSION = {"auto", "none", "off", "zstd", "snappy", "gzip", "lz4"}


@dataclass(slots=True)
class MediaOptions:
    # global read properties
    columns: Optional[Sequence[str]] = None
    use_threads: bool = True
    ignore_empty: bool = True
    lazy: bool = False
    raise_error: bool = True

    # global write properties (fixed types)
    mode: SaveMode = SaveMode.AUTO
    match_by: Sequence[str] | None = None

    @classmethod
    def check_parameters(
        cls,
        options: Optional["MediaOptions"] = None,
        *,
        mode: SaveMode | str | None | Any = _MISSING,
        match_by: Sequence[str] | str | None | Any = _MISSING,
        columns: Optional[Sequence[str]] = _MISSING,
        use_threads: bool = _MISSING,
        ignore_empty: bool = _MISSING,
        lazy: bool = _MISSING,
        raise_error: bool = _MISSING,
        **kwargs,
    ) -> "MediaOptions":
        """
        Merge/validate parameters into an options instance.

        - Only explicitly provided kwargs override existing values
        - Unknown kwargs raise TypeError
        - mode normalized via SaveMode.parse -> SaveMode
        - match_by normalized to tuple[str, ...] | None (still fits Sequence[str] | None)
        """
        if options is None:
            out = cls()
        else:
            if not isinstance(options, MediaOptions):
                raise TypeError(f"options must be a MediaOptions instance or None, got {type(options)!r}")

            if isinstance(options, cls):
                out = options
            else:
                # upcast into cls
                out = cls()
                for f in fields(cls):
                    if hasattr(options, f.name):
                        setattr(out, f.name, getattr(options, f.name))

        allowed = {f.name for f in fields(cls)}

        updates: dict[str, Any] = {}

        if mode is not _MISSING:
            updates["mode"] = mode
        if match_by is not _MISSING:
            updates["match_by"] = match_by
        if columns is not _MISSING:
            updates["columns"] = columns
        if use_threads is not _MISSING:
            updates["use_threads"] = use_threads
        if ignore_empty is not _MISSING:
            updates["ignore_empty"] = ignore_empty
        if lazy is not _MISSING:
            updates["lazy"] = lazy
        if raise_error is not _MISSING:
            updates["raise_error"] = raise_error

        updates.update(kwargs)

        unknown = [k for k in updates.keys() if k not in allowed]
        if unknown:
            raise TypeError(
                f"{cls.__name__}.check_parameters got unexpected parameter(s): {', '.join(sorted(unknown))}"
            )

        # ---- validations / normalizations ----

        if "columns" in updates:
            cols = updates["columns"]
            if cols is None:
                pass
            elif isinstance(cols, (str, bytes)):
                raise TypeError("columns must be a sequence of strings, not a single string/bytes")
            else:
                try:
                    cols_list = list(cols)
                except TypeError as e:
                    raise TypeError("columns must be a sequence of strings") from e
                if not all(isinstance(c, str) for c in cols_list):
                    bad = [type(c).__name__ for c in cols_list if not isinstance(c, str)]
                    raise TypeError(f"columns must contain only str, found: {bad[:3]}")
                updates["columns"] = tuple(cols_list)

        for b in ("use_threads", "ignore_empty", "lazy", "raise_error"):
            if b in updates and not isinstance(updates[b], bool):
                raise TypeError(f"{b} must be bool, got {type(updates[b]).__name__}")

        if "mode" in updates:
            updates["mode"] = SaveMode.parse(updates["mode"], default=SaveMode.AUTO)

        if "match_by" in updates:
            mb = updates["match_by"]
            if mb is None:
                pass
            elif isinstance(mb, (str, bytes)):
                if isinstance(mb, bytes):
                    raise TypeError("match_by must be str or a sequence of str, not bytes")
                updates["match_by"] = (mb,)
            else:
                try:
                    mb_list = list(mb)
                except TypeError as e:
                    raise TypeError("match_by must be a string or a sequence of strings") from e
                if not mb_list:
                    updates["match_by"] = None
                else:
                    if not all(isinstance(c, str) for c in mb_list):
                        bad = [type(c).__name__ for c in mb_list if not isinstance(c, str)]
                        raise TypeError(f"match_by must contain only str, found: {bad[:3]}")
                    updates["match_by"] = tuple(mb_list)

        # common subclass validations
        if "compression" in updates:
            comp = updates["compression"]
            if comp is not None:
                if not isinstance(comp, str):
                    raise TypeError(f"compression must be str|None, got {type(comp).__name__}")
                if comp not in _ALLOWED_COMPRESSION:
                    raise ValueError(f"compression must be one of {sorted(_ALLOWED_COMPRESSION)}, got {comp!r}")

        if "compression_level" in updates:
            lvl = updates["compression_level"]
            if lvl is not None and not isinstance(lvl, int):
                raise TypeError(f"compression_level must be int|None, got {type(lvl).__name__}")

        if "skip_rows" in updates:
            sr = updates["skip_rows"]
            if not isinstance(sr, int) or sr < 0:
                raise ValueError("skip_rows must be a non-negative int")

        if "zip_compression" in updates:
            zc = updates["zip_compression"]
            if not isinstance(zc, int):
                raise TypeError(f"zip_compression must be int, got {type(zc).__name__}")

        # apply
        for k, v in updates.items():
            setattr(out, k, v)

        return out
