"""NodePath ��� pathlib-like interface for node filesystem access.

Provides a ``pathlib.Path``-like API that transparently works with
local files or remote node filesystems via the ``/api/fs`` endpoints.

Local access uses direct filesystem calls.  Remote access uses
HTTP with streaming for large files.

Usage::

    from yggdrasil.node.path import NodePath

    # Local node files (direct filesystem)
    p = NodePath("data/input.csv")
    content = p.read_text()
    p.write_text("hello")
    for child in p.iterdir():
        print(child.name, child.is_dir())

    # Remote node files (via HTTP)
    remote = NodePath("data/input.csv", node_url="http://node-2:8100")
    content = remote.read_bytes()
    remote.write_bytes(b"data")
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator
from urllib.parse import urlparse


def _is_local_url(url: str) -> bool:
    """Check if a URL points to the local node.

    Compares the host and port against known localhost addresses and the
    configured YGG_NODE_PORT. When a "remote" URL actually points back to
    this node, we can skip HTTP and use direct filesystem access instead.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port
        # If no explicit port in URL, it's not targeting our node port
        if port is None:
            return False
        local_port = int(os.environ.get("YGG_NODE_PORT", "8100"))
        if port != local_port:
            return False
        return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")
    except (ValueError, TypeError):
        return False


class NodePath:
    """Path-like object for node filesystem access.

    When ``node_url`` is None, operates directly on the local filesystem
    within the node's data directory.  When ``node_url`` is set, uses
    HTTP calls to the remote node's ``/api/fs`` endpoints.

    Supports ``npfs://`` protocol URLs::

        npfs://node-host:port/path/to/content
        npfs://localhost:8100/data/input.csv
        npfs:///local/path  (triple slash = local node)
    """

    __slots__ = ("_path", "_node_url", "_root")

    def __init__(
        self,
        path: str = "",
        *,
        node_url: str | None = None,
        _root: Path | None = None,
    ) -> None:
        self._path = PurePosixPath(path.lstrip("/"))
        # Optimize: if node_url points to localhost, treat as local access
        # to skip unnecessary HTTP round-trips.
        if node_url and _is_local_url(node_url):
            node_url = None
        self._node_url = node_url
        if _root is None and node_url is None:
            from .config import get_settings
            self._root = get_settings().data_root / "files"
        else:
            self._root = _root

    # ── NPFS Protocol URL support ────────────────────────────

    @classmethod
    def from_url(cls, url: str) -> "NodePath":
        """Parse npfs://host:port/path URL.

        Supports:
        - ``npfs://host:port/path`` — remote node
        - ``npfs:///path`` — local node (triple slash)
        - Plain paths (no scheme) — treated as local
        """
        if url.startswith("npfs://"):
            parsed = urlparse(url)
            host = parsed.hostname or ""
            port = parsed.port or 8100
            path = parsed.path.lstrip("/")
            if not host or _is_local_url(f"http://{host}:{port}"):
                return cls(path)
            return cls(path, node_url=f"http://{host}:{port}")
        return cls(url)

    def to_url(self) -> str:
        """Convert to npfs:// URL.

        Local paths use triple-slash (``npfs:///path``).
        Remote paths include host and port.
        """
        if self.is_local:
            return f"npfs:///{self._path}"
        parsed = urlparse(self._node_url)
        return f"npfs://{parsed.hostname}:{parsed.port or 8100}/{self._path}"

    @classmethod
    def from_(cls, value) -> "NodePath | None":
        """Polymorphic constructor — accepts str, NodePath, None.

        Returns None if value is None, passes through NodePath instances,
        and parses strings as npfs:// URLs or plain paths.
        """
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        return cls.from_url(str(value))

    # ── Mirror support ───────────────────────────────────────

    def _get_node_home(self) -> Path:
        """Get the node home directory."""
        from .config import get_settings
        return get_settings().node_home

    def mirror_local(self) -> "NodePath":
        """Get the local mirror path for a remote resource.

        Remote files are mirrored under .ygg/mirrors/{node_id}/.
        Local paths return self unchanged.
        """
        if self.is_local:
            return self
        parsed = urlparse(self._node_url)
        node_id = f"{parsed.hostname}_{parsed.port or 8100}"
        mirror_root = self._get_node_home() / "mirrors" / node_id
        return NodePath(str(self._path), _root=mirror_root)

    def sync_from_remote(self) -> "NodePath":
        """Download remote file to local mirror.

        For local paths, returns self unchanged. For remote paths,
        fetches the content and writes it to the local mirror location.
        """
        if self.is_local:
            return self
        local = self.mirror_local()
        data = self.read_bytes()
        local.write_bytes(data)
        return local

    @property
    def is_local(self) -> bool:
        return self._node_url is None

    @property
    def name(self) -> str:
        return self._path.name or ""

    @property
    def parent(self) -> NodePath:
        return NodePath(
            str(self._path.parent) if str(self._path) != "." else "",
            node_url=self._node_url,
            _root=self._root,
        )

    @property
    def suffix(self) -> str:
        return self._path.suffix

    @property
    def stem(self) -> str:
        return self._path.stem

    def __truediv__(self, other: str) -> NodePath:
        return NodePath(
            str(self._path / other),
            node_url=self._node_url,
            _root=self._root,
        )

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        target = self._node_url or "local"
        return f"NodePath({str(self._path)!r}, node={target!r})"

    # ── Local helpers ────────────────────────────────────────

    def _local_path(self) -> Path:
        resolved = (self._root / str(self._path)).resolve()
        root_str = str(self._root.resolve())
        if not str(resolved).startswith(root_str):
            raise PermissionError("Path traversal not allowed")
        return resolved

    # ── Read operations ──────────────────────────────────────

    def exists(self) -> bool:
        if self.is_local:
            return self._local_path().exists()
        try:
            self.stat()
            return True
        except Exception:
            return False

    def is_dir(self) -> bool:
        if self.is_local:
            return self._local_path().is_dir()
        try:
            info = self.stat()
            return info.get("is_dir", False)
        except Exception:
            return False

    def is_file(self) -> bool:
        if self.is_local:
            return self._local_path().is_file()
        return not self.is_dir()

    def stat(self) -> dict:
        if self.is_local:
            p = self._local_path()
            s = p.stat()
            return {
                "path": str(self._path),
                "name": p.name,
                "is_dir": p.is_dir(),
                "size": s.st_size,
                "modified_at": datetime.fromtimestamp(s.st_mtime, tz=timezone.utc).isoformat(),
                "created_at": datetime.fromtimestamp(s.st_ctime, tz=timezone.utc).isoformat(),
            }
        return _get(f"{self._node_url}/api/fs/stat?path={_quote(str(self._path))}")

    def read_text(self, encoding: str = "utf-8") -> str:
        if self.is_local:
            return self._local_path().read_text(encoding=encoding)
        resp = _get(f"{self._node_url}/api/fs/read?path={_quote(str(self._path))}")
        content = resp.get("content", "")
        if resp.get("encoding") == "base64":
            return base64.b64decode(content).decode(encoding)
        return content

    def read_bytes(self) -> bytes:
        if self.is_local:
            return self._local_path().read_bytes()
        resp = _get(f"{self._node_url}/api/fs/read?path={_quote(str(self._path))}")
        content = resp.get("content", "")
        if resp.get("encoding") == "base64":
            return base64.b64decode(content)
        return content.encode()

    def iterdir(self) -> Iterator[NodePath]:
        if self.is_local:
            for child in sorted(self._local_path().iterdir()):
                yield NodePath(
                    str(self._path / child.name),
                    node_url=self._node_url,
                    _root=self._root,
                )
            return
        resp = _get(f"{self._node_url}/api/fs/ls?path={_quote(str(self._path))}")
        for entry in resp.get("entries", []):
            yield NodePath(
                entry["path"],
                node_url=self._node_url,
                _root=self._root,
            )

    # ── Write operations ─────────────────────────────────────

    def write_text(self, content: str, encoding: str = "utf-8") -> None:
        if self.is_local:
            p = self._local_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding=encoding)
            return
        _post(f"{self._node_url}/api/fs/write", {
            "path": str(self._path),
            "content": content,
            "encoding": "utf-8",
            "mkdir": True,
        })

    def write_bytes(self, data: bytes) -> None:
        if self.is_local:
            p = self._local_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            return
        _post(f"{self._node_url}/api/fs/write", {
            "path": str(self._path),
            "content": base64.b64encode(data).decode(),
            "encoding": "base64",
            "mkdir": True,
        })

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        if self.is_local:
            self._local_path().mkdir(parents=parents, exist_ok=exist_ok)
            return
        _post(f"{self._node_url}/api/fs/mkdir?path={_quote(str(self._path))}", {})

    def unlink(self, missing_ok: bool = False) -> None:
        if self.is_local:
            p = self._local_path()
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
            elif not missing_ok:
                raise FileNotFoundError(str(self._path))
            return
        try:
            _delete(f"{self._node_url}/api/fs/delete?path={_quote(str(self._path))}")
        except Exception:
            if not missing_ok:
                raise

    def rename(self, target: str) -> NodePath:
        if self.is_local:
            src = self._local_path()
            dst = (self._root / target.lstrip("/")).resolve()
            src.rename(dst)
            return NodePath(target, node_url=self._node_url, _root=self._root)
        _post(f"{self._node_url}/api/fs/move", {
            "source": str(self._path),
            "destination": target,
        })
        return NodePath(target, node_url=self._node_url, _root=self._root)

    # ── Streaming ────────────────────────────────────────────

    def stream_read(self, chunk_size: int = 65536) -> Iterator[bytes]:
        if self.is_local:
            with open(self._local_path(), "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            return
        req = urllib.request.Request(
            f"{self._node_url}/api/fs/stream?path={_quote(str(self._path))}",
            headers={"Accept": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    # ── Copy between nodes ───────────────────────────────────

    def copy_to(self, target: NodePath) -> NodePath:
        """Copy this file/directory to another NodePath (possibly on a different node)."""
        if self.is_dir():
            target.mkdir()
            for child in self.iterdir():
                child.copy_to(target / child.name)
            return target

        if self.is_local and target.is_local:
            shutil.copy2(str(self._local_path()), str(target._local_path()))
        else:
            data = self.read_bytes()
            target.write_bytes(data)
        return target


# ── HTTP helpers ─────────────────────────────────────────────

def _quote(s: str) -> str:
    from urllib.parse import quote
    return quote(s, safe="")


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _post(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _delete(url: str) -> None:
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=30):
        pass
