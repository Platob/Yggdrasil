from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

import pyarrow as pa
from databricks.sdk.service.catalog import ColumnInfo as CatalogColumnInfo
from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo
from yggdrasil.databricks.sql.types import column_info_to_arrow_field

if TYPE_CHECKING:
    from .table import Table

__all__ = ["Column"]

@dataclass(frozen=True)
class Column:
    table: "Table"
    name: str
    arrow_field: pa.Field = field(repr=False, compare=False, hash=False)

    @classmethod
    def from_api(
        cls,
        table: "Table",
        infos: SQLColumnInfo | CatalogColumnInfo
    ):
        arrow_field = column_info_to_arrow_field(infos)
        col = cls(
            table=table,
            name=arrow_field.name,
            arrow_field=arrow_field,
        )

        return col

    @property
    def engine(self):
        return self.table.sql

    @property
    def metadata(self) -> Mapping[bytes, bytes]:
        return self.arrow_field.metadata

    def set_tags_ddl(self, tags: Mapping[str, str]):
        str_tags = ", ".join(
            "'%s' = '%s'" % (k, v) for k, v in tags.items() if k and v
        )

        if not str_tags:
            return self

        query = f"ALTER TABLE {self.table.full_name(safe=True)} ALTER COLUMN `{self.name}` SET TAGS ({str_tags})"

        return query

    def set_tags(
        self,
        tags: Mapping[str, str],
    ):
        if tags:
            query = self.set_tags_ddl(tags)
            self.engine.execute(query)
        return self
