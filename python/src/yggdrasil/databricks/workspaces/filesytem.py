__all__ = [
    "DatabricksFileSystem",
    "DatabricksFileSystemHandler"
]

from typing import TYPE_CHECKING, Any, Union, List, Optional

from pyarrow import PythonFile
from pyarrow.fs import FileSystem, FileInfo, FileSelector, PyFileSystem, FileSystemHandler

if TYPE_CHECKING:
    from ..workspaces.workspace import Workspace
    from .path import DatabricksPath


class DatabricksFileSystemHandler(FileSystemHandler):

    def __init__(
        self,
        workspace: "Workspace",
    ):
        super().__init__()
        self.workspace = workspace

    def __enter__(self):
        return self.connect(clone=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workspace.__exit__(exc_type, exc_val, exc_tb)

    def _parse_path(self, obj: Any) -> "DatabricksPath":
        from .path import DatabricksPath

        return DatabricksPath.parse(obj, workspace=self.workspace)

    def connect(self, clone: bool = True):
        workspace = self.connect(clone=clone)

        if clone:
            return DatabricksFileSystemHandler(
                workspace=workspace
            )

        self.workspace = workspace
        return self

    def close(self):
        self.workspace.close()

    def copy_file(self, src, dest, *, chunk_size: int = 4 * 1024 * 1024):
        src = self._parse_path(src)
        dest = self._parse_path(dest)

        with src.open("rb") as r, dest.open("wb") as w:
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                w.write(chunk)

    def create_dir(self, path, *args, recursive: bool = True, **kwargs):
        return self._parse_path(path).mkdir(parents=recursive)

    def delete_dir(self, path):
        return self._parse_path(path).rmdir(recursive=True)

    def delete_dir_contents(self, path, *args, accept_root_dir: bool = False, **kwargs):
        return self._parse_path(path).rmdir(recursive=True)

    def delete_root_dir_contents(self):
        return self.delete_dir_contents("/", accept_root_dir=True)

    def delete_file(self, path):
        return self._parse_path(path).rmfile()

    def equals(self, other: FileSystem):
        return self == other

    def from_uri(self, uri):
        uri = self._parse_path(uri)

        return self.__class__(
            workspace=uri.workspace
        )

    def get_file_info(
        self,
        paths_or_selector: Union[FileSelector, str, "DatabricksPath", List[Union[str, "DatabricksPath"]]]
    ) -> Union[FileInfo, List[FileInfo]]:
        from .path import DatabricksPath

        if isinstance(paths_or_selector, (str, DatabricksPath)):
            result = self._parse_path(paths_or_selector).file_info

            return result

        if isinstance(paths_or_selector, FileSelector):
            return self.get_file_info_selector(paths_or_selector)

        return [
            self.get_file_info(obj)
            for obj in paths_or_selector
        ]

    def get_file_info_selector(
        self,
        selector: FileSelector
    ):
        base_dir = self._parse_path(selector.base_dir)

        return [
            p.file_info
            for p in base_dir.ls(
                recursive=selector.recursive,
                allow_not_found=selector.allow_not_found
            )
        ]

    def get_type_name(self):
        return "dbfs"

    def move(self, src, dest):
        src = self._parse_path(src)

        src.copy_to(dest)

        src.remove(recursive=True)

    def normalize_path(self, path):
        return self._parse_path(path).full_path()

    def open(
        self,
        path,
        mode: str = "r+",
        encoding: Optional[str] = None,
    ):
        return self._parse_path(path).open(mode=mode, encoding=encoding, clone=False)

    def open_append_stream(self, path, compression='detect', buffer_size=None, metadata=None):
        return self._parse_path(path).open(mode="ab")

    def open_input_file(self, path, mode: str = "rb", **kwargs):
        buf = self._parse_path(path).open(mode=mode).connect(clone=True)

        return PythonFile(
            buf,
            mode=mode
        )

    def open_input_stream(self, path, compression='detect', buffer_size=None):
        return self._parse_path(path).open(mode="rb")

    def open_output_stream(self, path, compression='detect', buffer_size=None, metadata=None):
        return self._parse_path(path).open(mode="wb")


class DatabricksFileSystem(PyFileSystem):

    def __init__(self, handler): # real signature unknown; restored from __doc__
        super().__init__(handler)
