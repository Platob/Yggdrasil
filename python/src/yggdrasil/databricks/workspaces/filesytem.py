"""PyArrow filesystem wrappers for Databricks paths."""

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
    """PyArrow FileSystemHandler backed by Databricks paths."""

    def __init__(
        self,
        workspace: "Workspace",
    ):
        """Create a handler bound to a Workspace.

        Args:
            workspace: Workspace instance to use.
        """
        super().__init__()
        self.workspace = workspace

    def __enter__(self):
        """Enter a context manager and connect to the workspace.

        Returns:
            A connected DatabricksFileSystemHandler instance.
        """
        return self.connect(clone=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the workspace.

        Args:
            exc_type: Exception type, if raised.
            exc_val: Exception value, if raised.
            exc_tb: Exception traceback, if raised.
        """
        self.workspace.__exit__(exc_type, exc_val, exc_tb)

    def _parse_path(self, obj: Any) -> "DatabricksPath":
        """Parse a path-like object into a DatabricksPath.

        Args:
            obj: Path-like object to parse.

        Returns:
            A DatabricksPath instance.
        """
        from .path import DatabricksPath

        return DatabricksPath.parse(obj, workspace=self.workspace)

    def connect(self, clone: bool = True):
        """Connect the workspace and optionally return a cloned handler.

        Args:
            clone: Whether to return a cloned handler.

        Returns:
            A connected handler.
        """
        workspace = self.connect(clone=clone)

        if clone:
            return DatabricksFileSystemHandler(
                workspace=workspace
            )

        self.workspace = workspace
        return self

    def close(self):
        """Close the underlying workspace client.

        Returns:
            None.
        """
        self.workspace.close()

    def copy_file(self, src, dest, *, chunk_size: int = 4 * 1024 * 1024):
        """Copy a file between Databricks paths.

        Args:
            src: Source path.
            dest: Destination path.
            chunk_size: Chunk size in bytes.
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
        """Create a directory at the given path.

        Args:
            path: Directory path to create.
            recursive: Whether to create parents.

        Returns:
            The created DatabricksPath instance.
        """
        return self._parse_path(path).mkdir(parents=recursive)

    def delete_dir(self, path):
        """Delete a directory recursively.

        Args:
            path: Directory path to delete.
        """
        return self._parse_path(path).rmdir(recursive=True)

    def delete_dir_contents(self, path, *args, accept_root_dir: bool = False, **kwargs):
        """Delete the contents of a directory.

        Args:
            path: Directory path whose contents should be removed.
            accept_root_dir: Whether to allow deleting root contents.
        """
        return self._parse_path(path).rmdir(recursive=True)

    def delete_root_dir_contents(self):
        """Delete the contents of the root directory."""
        return self.delete_dir_contents("/", accept_root_dir=True)

    def delete_file(self, path):
        """Delete a single file.

        Args:
            path: File path to delete.
        """
        return self._parse_path(path).rmfile()

    def equals(self, other: FileSystem):
        """Return True if the filesystem handler matches another.

        Args:
            other: Another FileSystem instance.

        Returns:
            True if equal, otherwise False.
        """
        return self == other

    def from_uri(self, uri):
        """Return a handler for the workspace in the provided URI.

        Args:
            uri: URI or path to parse.

        Returns:
            A DatabricksFileSystemHandler for the URI.
        """
        uri = self._parse_path(uri)

        return self.__class__(
            workspace=uri.workspace
        )

    def get_file_info(
        self,
        paths_or_selector: Union[FileSelector, str, "DatabricksPath", List[Union[str, "DatabricksPath"]]]
    ) -> Union[FileInfo, List[FileInfo]]:
        """Return FileInfo objects for paths or selectors.

        Args:
            paths_or_selector: Path(s) or a FileSelector.

        Returns:
            A FileInfo or list of FileInfo objects.
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
        """Return FileInfo entries for a FileSelector.

        Args:
            selector: FileSelector describing the listing.

        Returns:
            A list of FileInfo entries.
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
        """Return the filesystem type name.

        Returns:
            The filesystem type name string.
        """
        return "dbfs"

    def move(self, src, dest):
        """Move a file by copying then deleting.

        Args:
            src: Source path.
            dest: Destination path.
        """
        src = self._parse_path(src)

        src.copy_to(dest)

        src.remove(recursive=True)

    def normalize_path(self, path):
        """Normalize a path to a full Databricks path string.

        Args:
            path: Path to normalize.

        Returns:
            The normalized full path string.
        """
        return self._parse_path(path).full_path()

    def open(
        self,
        path,
        mode: str = "r+",
        encoding: Optional[str] = None,
    ):
        """Open a file path as a Databricks IO stream.

        Args:
            path: Path to open.
            mode: File mode string.
            encoding: Optional text encoding.

        Returns:
            A DatabricksIO instance.
        """
        return self._parse_path(path).open(mode=mode, encoding=encoding, clone=False)

    def open_append_stream(self, path, compression='detect', buffer_size=None, metadata=None):
        """Open an append stream.

        Args:
            path: Path to open.
            compression: Optional compression hint.
            buffer_size: Optional buffer size.
            metadata: Optional metadata.

        Returns:
            A DatabricksIO instance.
        """
        return self._parse_path(path).open(mode="ab")

    def open_input_file(self, path, mode: str = "rb", **kwargs):
        """Open an input file as a PyArrow PythonFile.

        Args:
            path: Path to open.
            mode: File mode string.
            **kwargs: Additional options.

        Returns:
            A PyArrow PythonFile instance.
        """
        buf = self._parse_path(path).open(mode=mode).connect(clone=True)

        return PythonFile(
            buf,
            mode=mode
        )

    def open_input_stream(self, path, compression='detect', buffer_size=None):
        """Open an input stream for reading bytes.

        Args:
            path: Path to open.
            compression: Optional compression hint.
            buffer_size: Optional buffer size.

        Returns:
            A DatabricksIO instance.
        """
        return self._parse_path(path).open(mode="rb")

    def open_output_stream(self, path, compression='detect', buffer_size=None, metadata=None):
        """Open an output stream for writing bytes.

        Args:
            path: Path to open.
            compression: Optional compression hint.
            buffer_size: Optional buffer size.
            metadata: Optional metadata.

        Returns:
            A DatabricksIO instance.
        """
        return self._parse_path(path).open(mode="wb")


class DatabricksFileSystem(PyFileSystem):
    """PyArrow filesystem wrapper for Databricks paths."""

    def __init__(self, handler): # real signature unknown; restored from __doc__
        """Initialize the filesystem with a handler.

        Args:
            handler: FileSystemHandler instance.
        """
        super().__init__(handler)
