import os
from dataclasses import dataclass
from typing import Optional


__all__ = [
    "TableLocation",
    "CatalogSchemaTable"
]


@dataclass(frozen=True)
class CatalogSchemaTable:
    catalog_name: str
    schema_name: str
    table_name: str

    def clean(self):
        setattr(self, "catalog_name", self.strip_entity_name(self.catalog_name))
        setattr(self, "schema_name", self.strip_entity_name(self.schema_name))
        setattr(self, "table_name", self.strip_entity_name(self.table_name))

    @classmethod
    def parse(cls, path: str):
        if isinstance(path, CatalogSchemaTable):
            return path

        catalog_name = None
        schema_name = None
        table_name = None

        if isinstance(path, str):
            path = path.split(".")

        if isinstance(path, (list, tuple, set)):
            if len(path) >= 3:
                catalog_name = path[-3]
                schema_name = path[-2]
                table_name = path[-1]

        catalog_name = catalog_name or os.environ.get("YGG_DEFAULT_CATALOG")
        schema_name = schema_name or os.environ.get("YGG_DEFAULT_SCHEMA")
        table_name = table_name or os.environ.get("YGG_DEFAULT_TABLE")

        if not catalog_name or not schema_name or not table_name:
            raise ValueError(f"Cannot build catalog schema table from {path}")

        return CatalogSchemaTable(
            catalog_name=cls.strip_entity_name(catalog_name),
            schema_name=cls.strip_entity_name(schema_name),
            table_name=cls.strip_entity_name(table_name)
        )


    @classmethod
    def strip_entity_name(cls, path: str) -> Optional[str]:
        if not path:
            return None

        sane = path.strip()

        for start, end in [
            ("[", "]"), ("(", ")"),
            ("`", "`")
        ]:
            if sane.endswith(start):
                sane = sane[:-len(start)]
            if sane.endswith(end):
                sane = sane[:-len(end)]

        return sane

    def entity_full_name(self, decorator: str, separator: str) -> str:
        return separator.join(f"{decorator}{item}{decorator}" for item in [
            self.catalog_name,
            self.schema_name,
            self.table_name
        ])

    def delta_table_full_name(self):
        return self.entity_full_name(decorator="`", separator=".")


@dataclass(frozen=True)
class TableLocation:
    fs_path: str | None
    entitiy: CatalogSchemaTable | None

    @classmethod
    def parse_any(cls, obj: "TableLocation" | str) -> "TableLocation":
        if isinstance(obj, TableLocation):
            return obj

        if not obj:
            raise ValueError("TableLocation path cannot be empty")

        obj = obj.replace("\\", "/")

        if "/" in obj:
            return cls(fs_path=obj, entitiy=None)

        sql_table = CatalogSchemaTable.parse(obj)
        return cls(fs_path=None, entitiy=sql_table)
