from __future__ import annotations

import json

import pytest

pytest.importorskip("databricks.sdk")

from databricks.sdk.service.catalog import ColumnInfo, ColumnTypeName

from yggdrasil.databricks.sql.types import parse_catalog_column_info_field


class TestParseCatalogColumnInfoStruct:
    def test_struct_child_not_null_survives_when_type_json_present(self):
        # Catalog ``type_text`` for a struct column drops child nullability;
        # ``type_json`` keeps it. Merging both should not clobber the
        # JSON-derived ``NOT NULL`` flags.
        type_json = json.dumps({
            "name": "request_url",
            "type": {
                "type": "struct",
                "fields": [
                    {"name": "scheme", "type": "string", "nullable": False, "metadata": {}},
                    {"name": "userinfo", "type": "string", "nullable": True, "metadata": {}},
                    {"name": "host", "type": "string", "nullable": False, "metadata": {}},
                    {"name": "path", "type": "string", "nullable": False, "metadata": {}},
                    {"name": "query", "type": "string", "nullable": True, "metadata": {}},
                ],
            },
            "nullable": True,
            "metadata": {},
        })
        obj = ColumnInfo(
            name="request_url",
            type_name=ColumnTypeName.STRUCT,
            type_text="struct<scheme:string,userinfo:string,host:string,path:string,query:string>",
            type_json=type_json,
            nullable=True,
        )

        f = parse_catalog_column_info_field(obj)

        children = {c.name: c.nullable for c in f.dtype.fields}
        assert children == {
            "scheme": False,
            "userinfo": True,
            "host": False,
            "path": False,
            "query": True,
        }
        ddl = f.to_databricks_ddl(put_name=False, put_not_null=False, put_comment=False)
        assert "`scheme` STRING NOT NULL" in ddl
        assert "`host` STRING NOT NULL" in ddl
        assert "`path` STRING NOT NULL" in ddl
        assert "`userinfo` STRING," in ddl
        assert "`query` STRING>" in ddl

    def test_struct_falls_back_to_type_text_when_type_json_missing(self):
        # Without ``type_json`` the parser must still populate the
        # struct fields from ``type_text``; nullability defaults to
        # ``True`` since ``type_text`` doesn't carry it.
        obj = ColumnInfo(
            name="request_url",
            type_name=ColumnTypeName.STRUCT,
            type_text="struct<scheme:string,host:string,port:int>",
            type_json=None,
            nullable=True,
        )

        f = parse_catalog_column_info_field(obj)

        children = [(c.name, c.nullable) for c in f.dtype.fields]
        assert children == [
            ("scheme", True),
            ("host", True),
            ("port", True),
        ]
