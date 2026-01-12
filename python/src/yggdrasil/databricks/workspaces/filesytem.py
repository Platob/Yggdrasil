"""databricks.workspaces.filesytem module documentation."""

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
        """
        __init__ documentation.
        
        Args:
            workspace: Parameter.
        
        Returns:
            None.
        """

        super().__init__()
        self.workspace = workspace

    def __enter__(self):
        """
        __enter__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.connect(clone=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        __exit__ documentation.
        
        Args:
            exc_type: Parameter.
            exc_val: Parameter.
            exc_tb: Parameter.
        
        Returns:
            The result.
        """

        self.workspace.__exit__(exc_type, exc_val, exc_tb)

    def _parse_path(self, obj: Any) -> "DatabricksPath":
        """
        _parse_path documentation.
        
        Args:
            obj: Parameter.
        
        Returns:
            The result.
        """

        from .path import DatabricksPath

        return DatabricksPath.parse(obj, workspace=self.workspace)

    def connect(self, clone: bool = True):
        """
        connect documentation.
        
        Args:
            clone: Parameter.
        
        Returns:
            The result.
        """

        workspace = self.connect(clone=clone)

        if clone:
            return DatabricksFileSystemHandler(
                workspace=workspace
            )

        self.workspace = workspace
        return self

    def close(self):
        """
        close documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        self.workspace.close()

    def copy_file(self, src, dest, *, chunk_size: int = 4 * 1024 * 1024):
        """
        copy_file documentation.
        
        Args:
            src: Parameter.
            dest: Parameter.
            chunk_size: Parameter.
        
        Returns:
            The result.
        """

        src = self._parse_path(src)
        dest = self._parse_path(dest)

        with src.open("rb") as r, dest.open("wb") as w:
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                w.write(chunk)

    def create_dir(self, path, *args, recursive: bool = True, **kwargs):
        """
        create_dir documentation.
        
        Args:
            path: Parameter.
            recursive: Parameter.
            *args: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        return self._parse_path(path).mkdir(parents=recursive)

    def delete_dir(self, path):
        """
        delete_dir documentation.
        
        Args:
            path: Parameter.
        
        Returns:
            The result.
        """

        return self._parse_path(path).rmdir(recursive=True)

    def delete_dir_contents(self, path, *args, accept_root_dir: bool = False, **kwargs):
        """
        delete_dir_contents documentation.
        
        Args:
            path: Parameter.
            accept_root_dir: Parameter.
            *args: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        return self._parse_path(path).rmdir(recursive=True)

    def delete_root_dir_contents(self):
        """
        delete_root_dir_contents documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.delete_dir_contents("/", accept_root_dir=True)

    def delete_file(self, path):
        """
        delete_file documentation.
        
        Args:
            path: Parameter.
        
        Returns:
            The result.
        """

        return self._parse_path(path).rmfile()

    def equals(self, other: FileSystem):
        """
        equals documentation.
        
        Args:
            other: Parameter.
        
        Returns:
            The result.
        """

        return self == other

    def from_uri(self, uri):
        """
        from_uri documentation.
        
        Args:
            uri: Parameter.
        
        Returns:
            The result.
        """

        uri = self._parse_path(uri)

        return self.__class__(
            workspace=uri.workspace
        )

    def get_file_info(
        self,
        paths_or_selector: Union[FileSelector, str, "DatabricksPath", List[Union[str, "DatabricksPath"]]]
    ) -> Union[FileInfo, List[FileInfo]]:
        """
        get_file_info documentation.
        
        Args:
            paths_or_selector: Parameter.
        
        Returns:
            The result.
        """

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
        """
        get_file_info_selector documentation.
        
        Args:
            selector: Parameter.
        
        Returns:
            The result.
        """

        base_dir = self._parse_path(selector.base_dir)

        return [
            p.file_info
            for p in base_dir.ls(
                recursive=selector.recursive,
                allow_not_found=selector.allow_not_found
            )
        ]

    def get_type_name(self):
        """
        get_type_name documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return "dbfs"

    def move(self, src, dest):
        """
        move documentation.
        
        Args:
            src: Parameter.
            dest: Parameter.
        
        Returns:
            The result.
        """

        src = self._parse_path(src)

        src.copy_to(dest)

        src.remove(recursive=True)

    def normalize_path(self, path):
        """
        normalize_path documentation.
        
        Args:
            path: Parameter.
        
        Returns:
            The result.
        """

        return self._parse_path(path).full_path()

    def open(
        self,
        path,
        mode: str = "r+",
        encoding: Optional[str] = None,
    ):
        """
        open documentation.
        
        Args:
            path: Parameter.
            mode: Parameter.
            encoding: Parameter.
        
        Returns:
            The result.
        """

        return self._parse_path(path).open(mode=mode, encoding=encoding, clone=False)

    def open_append_stream(self, path, compression='detect', buffer_size=None, metadata=None):
        """
        open_append_stream documentation.
        
        Args:
            path: Parameter.
            compression: Parameter.
            buffer_size: Parameter.
            metadata: Parameter.
        
        Returns:
            The result.
        """

        return self._parse_path(path).open(mode="ab")

    def open_input_file(self, path, mode: str = "rb", **kwargs):
        """
        open_input_file documentation.
        
        Args:
            path: Parameter.
            mode: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        buf = self._parse_path(path).open(mode=mode).connect(clone=True)

        return PythonFile(
            buf,
            mode=mode
        )

    def open_input_stream(self, path, compression='detect', buffer_size=None):
        """
        open_input_stream documentation.
        
        Args:
            path: Parameter.
            compression: Parameter.
            buffer_size: Parameter.
        
        Returns:
            The result.
        """

        return self._parse_path(path).open(mode="rb")

    def open_output_stream(self, path, compression='detect', buffer_size=None, metadata=None):
        """
        open_output_stream documentation.
        
        Args:
            path: Parameter.
            compression: Parameter.
            buffer_size: Parameter.
            metadata: Parameter.
        
        Returns:
            The result.
        """

        return self._parse_path(path).open(mode="wb")


class DatabricksFileSystem(PyFileSystem):

    def __init__(self, handler): # real signature unknown; restored from __doc__
        """
        __init__ documentation.
        
        Args:
            handler: Parameter.
        
        Returns:
            None.
        """

        super().__init__(handler)
