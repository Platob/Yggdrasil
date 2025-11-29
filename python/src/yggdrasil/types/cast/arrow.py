from dataclasses import dataclass
from typing import Union, Optional

import pyarrow as pa
import pyarrow.compute as pc

__all__ = [
    "ArrowCastOptions",
    "cast_arrow_array",
    "cast_arrow_table",
    "cast_arrow_batch"
]


@dataclass
class ArrowCastOptions:
    safe: bool = False
    add_missing_columns: bool = True
    strict_match_names: bool = False
    allow_add_columns: bool =  False
    rename: bool = True
    memory_pool: Optional[pa.MemoryPool] = None

    def __post_init__(self):
        if self.safe is None:
            self.safe = False

        if self.add_missing_columns is None:
            self.add_missing_columns = True

        if self.strict_match_names is None:
            self.strict_match_names = False

        if self.allow_add_columns is None:
            self.allow_add_columns = False

        if self.rename is None:
            self.rename = True


DEFAULT_CAST_OPTIONS = ArrowCastOptions()


def cast_arrow_array(
    data: Union[pa.ChunkedArray, pa.Array],
    arrow_type: pa.DataType,
    options: Optional[ArrowCastOptions] = None
):
    options = options or DEFAULT_CAST_OPTIONS

    return pc.cast(
        data,
        target_type=arrow_type,
        safe=options.safe,
        memory_pool=options.memory_pool
    )


def cast_arrow_table(
    data: Union[pa.Table],
    options: Optional[ArrowCastOptions] = None
):
    ...


def cast_arrow_batch(
    self,
    data: Union[pa.RecordBatch],
    options: Optional[ArrowCastOptions] = None
):
    ...