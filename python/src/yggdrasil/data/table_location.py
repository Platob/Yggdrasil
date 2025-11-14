from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TableLocation:
    full_path: str
    catalog_name: Optional[str]
    schema_name: Optional[str]
    table_name: Optional[str]

    @classmethod
    def parse(cls, path: str) -> "TableLocation":
        if not path:
            raise ValueError("TableLocation path cannot be empty")

        if "/" in path:
            raise ValueError("TableLocation path cannot contain '/'")

        parts = path.rsplit("/")[-1].split(".")

        if len(parts) != 3:
            return cls(
                full_path=path,
                catalog_name=None,
                schema_name=None,
                table_name=None,
            )

        catalog_name, schema_name, table_name = parts

        return cls(
            full_path=path,
            catalog_name=cls.strip_special(catalog_name),
            schema_name=cls.strip_special(schema_name),
            table_name=cls.strip_special(table_name),
        )

    @classmethod
    def strip_special(cls, path: str) -> Optional[str]:
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

    def sql_full_name(self, decorator: str, separator: str) -> str:
        return separator.join(f"{decorator}{item}{decorator}" for item in [
            self.catalog_name,
            self.schema_name,
            self.table_name
        ])

    def delta_table_name(self):
        return self.sql_full_name(decorator="`", separator=".")
