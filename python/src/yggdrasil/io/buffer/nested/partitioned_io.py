"""Hive-partitioned folder IO.

Partition handling moved into :class:`FolderIO` itself: configure
``partition_columns`` on a :class:`FolderIO` and the read/write
paths inject / route automatically. :class:`PartitionedFolderIO`
remains as a backwards-compatible alias that requires
``partition_columns`` at construction so a partitioned-by-contract
caller fails loudly when partitions aren't declared.

:class:`PartitionedOptions` is :class:`FolderOptions` re-exported
under the historical name for callers importing it directly.
"""

from __future__ import annotations

from typing import Any, ClassVar, Sequence

from yggdrasil.io.enums import MimeType, MimeTypes

from .folder_io import FolderIO, FolderOptions

__all__ = ["PartitionedFolderIO", "PartitionedOptions"]


# Backwards-compatible alias — :class:`FolderOptions` already carries
# the partition knobs (``partition_columns``, ``sort_partitions``,
# ``partition_strict``).
PartitionedOptions = FolderOptions


class PartitionedFolderIO(FolderIO):
    """A :class:`FolderIO` that demands partition columns up front.

    The base :class:`FolderIO` is partition-capable on its own;
    :class:`PartitionedFolderIO` adds nothing functional but
    enforces that ``partition_columns`` is supplied (or inferable)
    so a partitioned-by-contract caller doesn't silently degrade
    to flat-folder behavior on an empty or misshapen tree.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_mime_type(cls) -> MimeType:
        return MimeTypes.PARTITIONED_FOLDER

    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        partition_columns: "Sequence[Any] | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data,
            path=path,
            partition_columns=partition_columns,
            **kwargs,
        )
