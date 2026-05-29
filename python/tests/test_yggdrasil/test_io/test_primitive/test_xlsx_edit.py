"""ExcelFile workbook surface: dims, windowed range reads, surgical edits."""
from __future__ import annotations

import pytest

openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("fastexcel")

from yggdrasil.io.primitive.xlsx_file import ExcelFile, XLSXFile
from yggdrasil.path.local_path import LocalPath


def _seed(path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["id", "city", "temp"])
    for i in range(1, 6):
        ws.append([i, "Oslo", i * 1.5])
    ws["E1"] = "total"
    ws["E2"] = "=SUM(C2:C6)"   # a formula we expect to survive edits
    meta = wb.create_sheet("Meta")
    meta.append(["k", "v"])
    meta.append(["owner", "ygg"])
    wb.save(str(path))


class TestAliasAndDims:
    def test_excel_alias_is_xlsx(self):
        assert ExcelFile is XLSXFile

    def test_sheet_infos(self, tmp_path):
        p = tmp_path / "b.xlsx"
        _seed(p)
        with LocalPath(str(p)).open("rb") as wb:
            infos = wb.sheet_infos()
        by_name = {i["name"]: i for i in infos}
        assert set(by_name) == {"Data", "Meta"}
        assert by_name["Data"]["rows"] == 6          # header + 5 rows
        assert by_name["Data"]["cols"] == 5          # A..E (formula col)
        assert by_name["Meta"]["rows"] == 2


class TestRangeRead:
    def test_n_rows_window(self, tmp_path):
        p = tmp_path / "b.xlsx"
        _seed(p)
        with LocalPath(str(p)).open("rb") as wb:
            t = wb.read_range("Data", n_rows=2)
        assert t.num_rows == 2
        assert t.column("id").to_pylist() == [1.0, 2.0]

    def test_column_subset(self, tmp_path):
        p = tmp_path / "b.xlsx"
        _seed(p)
        with LocalPath(str(p)).open("rb") as wb:
            t = wb.read_range("Data", columns=["id", "city"])
        assert t.column_names[:2] == ["id", "city"]
        assert "temp" not in t.column_names

    def test_skip_rows(self, tmp_path):
        p = tmp_path / "b.xlsx"
        _seed(p)
        with LocalPath(str(p)).open("rb") as wb:
            # header consumed, then skip the first 2 data rows
            t = wb.read_range("Data", skip_rows=2)
        assert t.num_rows == 3


class TestSurgicalEdit:
    def test_apply_edits_preserves_formula_and_other_sheets(self, tmp_path):
        p = tmp_path / "b.xlsx"
        _seed(p)
        with LocalPath(str(p)).open("rb") as wb:
            n = wb.apply_edits("Data", [(2, 2, "Bergen"), (3, 2, "Tromso")])
        assert n == 2
        out = openpyxl.load_workbook(str(p))
        assert out["Data"]["B2"].value == "Bergen"
        assert out["Data"]["B3"].value == "Tromso"
        assert out["Data"]["E2"].value == "=SUM(C2:C6)"   # formula intact
        assert out["Data"]["A2"].value == 1               # untouched cell intact
        assert out["Meta"]["A2"].value == "owner"          # other sheet intact

    def test_edit_range_block(self, tmp_path):
        p = tmp_path / "b.xlsx"
        _seed(p)
        with LocalPath(str(p)).open("rb") as wb:
            n = wb.edit_range("Data", 2, 1, [[100, "X"], [200, "Y"]])
        assert n == 4
        out = openpyxl.load_workbook(str(p))
        assert out["Data"]["A2"].value == 100
        assert out["Data"]["B2"].value == "X"
        assert out["Data"]["A3"].value == 200
        assert out["Data"]["B3"].value == "Y"

    def test_apply_edits_unknown_sheet_requires_create(self, tmp_path):
        p = tmp_path / "b.xlsx"
        _seed(p)
        with LocalPath(str(p)).open("rb") as wb:
            with pytest.raises(KeyError):
                wb.apply_edits("Ghost", [(1, 1, "x")])
        with LocalPath(str(p)).open("rb") as wb:
            wb.apply_edits("Ghost", [(1, 1, "x")], create=True)
        out = openpyxl.load_workbook(str(p))
        assert "Ghost" in out.sheetnames
        assert out["Ghost"]["A1"].value == "x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
