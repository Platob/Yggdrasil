# yggdrasil/io/buffer/json_io.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, Optional

import yggdrasil.pickle.json as json_mod
from .bytes_io import BytesIO
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["JsonOptions", "JsonIO"]


@dataclass(slots=True)
class JsonOptions(MediaOptions):
    # ---- common read options (used by pyarrow.json) ----
    columns: list[str] | None = None
    use_threads: bool = True
    allow_newlines_in_values: bool = False

    # ---- encoding for manual writes ----
    encoding: str = "utf-8"
    errors: str = "strict"

    @classmethod
    def resolve(cls, *, options: Self | None = None, **overrides) -> Self:
        base = options or cls()
        valid = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        unknown = set(overrides) - set(valid)
        if unknown:
            raise TypeError(f"{cls.__name__}.resolve(): unknown option(s): {sorted(unknown)}")
        for k, v in overrides.items():
            setattr(base, k, v)
        return base


@dataclass(slots=True)
class JsonIO(MediaIO[JsonOptions]):
    """
    JSON array-of-objects IO on top of BytesIO.

    - write_arrow_table writes ONE JSON document: a list[object] (array) of rows (single write)
    - read_arrow_table reads using pyarrow.json.open_json(...).read_all()
    """

    buffer: BytesIO

    @classmethod
    def check_options(cls, options: Optional[JsonOptions], *args, **kwargs) -> JsonOptions:
        return JsonOptions.check_parameters(
            options=options,
            **kwargs
        )

    def read_pylist(self):
        if self.buffer.size <= 0:
            return []

        with self.buffer.view() as f:
            parsed = json_mod.load(f)

        if not isinstance(parsed, list):
            parsed = [parsed]

        return parsed

    def write_pylist(
        self,
        data: list,
    ):
        with self.buffer.view() as f:
            json_mod.dump(data, f)

    def _read_arrow_table(self, *, options: JsonOptions) -> "pyarrow.Table":
        from yggdrasil.arrow.lib import pyarrow as _pa
        records = self.read_pylist()

        if not records:
            return _pa.table({})

        # records is list[dict] => correct API
        return _pa.Table.from_pylist(records)

    def _write_arrow_table(self, *, table: "pyarrow.Table", options: JsonOptions) -> None:
        records = table.to_pylist()

        return self.write_pylist(records)