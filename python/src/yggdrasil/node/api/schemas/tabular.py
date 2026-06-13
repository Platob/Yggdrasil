"""Tabular inspection schema."""
from __future__ import annotations

from pydantic import BaseModel


class TabularInspectResult(BaseModel):
    path: str
    row_count: int
    col_count: int
    schema_: list[dict]  # [{name, type}]
    editable: bool       # True for CSV/JSON, False for parquet (footer-only)
    format: str          # "parquet", "csv", ...

    # `schema` shadows BaseModel.schema; expose the field under the friendly
    # name on the wire while keeping the attribute usable in Python.
    model_config = {"populate_by_name": True}

    def __init__(self, **data):
        if "schema" in data and "schema_" not in data:
            data["schema_"] = data.pop("schema")
        super().__init__(**data)

    def model_dump(self, **kwargs):  # noqa: D401 - serialize with `schema` key
        d = super().model_dump(**kwargs)
        d["schema"] = d.pop("schema_")
        return d
