from pyarrow.dataset import FileFormat, ParquetFileFormat, CsvFileFormat, JsonFileFormat


__all__ = [
    "FileFormat",
    "ExcelFileFormat",
    "ParquetFileFormat",
    "CsvFileFormat",
    "JsonFileFormat"
]


class ExcelFileFormat(FileFormat):
    pass
