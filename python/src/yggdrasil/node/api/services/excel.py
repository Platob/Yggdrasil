"""Excel-facing node service.

A thin façade that shapes the node's existing capabilities for Excel /
Power Query / an Office.js add-in:

- **run Python → table** — exec a snippet in a chosen :class:`PyEnv`
  (or the node interpreter), grab a named dataframe, hand it back as a
  typed table.
- **read a file → table** — parquet / csv / json / arrow decoded into a
  table so the sheet gets typed columns.
- **write a table → file** — push a table (uploaded as parquet / arrow /
  csv) to a path under the node root.
- **walk the remote filesystem** — a navigation tree for the connector.

Tables serialize to Parquet (default — Power Query's ``Parquet.Document``
reads it natively), Arrow IPC (the add-in decodes it with apache-arrow
in JS), or JSON records. Paths resolve through :class:`FsService`, which
keeps everything under the node root with traversal protection.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import pyarrow as pa
from fastapi.concurrency import run_in_threadpool

from ... import transport
from ...config import Settings
from yggdrasil.enums import MimeTypes
from yggdrasil.enums.media_type import MediaType
from yggdrasil.exceptions.api import BadRequestError, TimeoutError as APITimeoutError
from yggdrasil.io.holder import IO
from yggdrasil.path import Path as YggPath
from yggdrasil.version import __version__
from ...exceptions import NotFoundError
from ..schemas.excel import (
    ExcelInfo,
    ExcelQueryRequest,
    ExcelTreeNode,
    ExcelTreeResponse,
    ExcelWriteResponse,
)
from .fs import FsService
from .pyenv import PyEnvService

LOGGER = logging.getLogger(__name__)

# Self-contained driver: exec the user's snippet as a script, pull the
# named dataframe out of its namespace, coerce common frame types to an
# Arrow table, and write it as Parquet. Deliberately depends only on
# pyarrow + stdlib so it runs in any PyEnv (which may not have ygg).
_DRIVER = r"""
import runpy, sys
import pyarrow as pa
import pyarrow.parquet as pq

out_path, df_name, max_rows, code_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ns = runpy.run_path(code_path)
if df_name not in ns:
    sys.stderr.write("snippet did not define %r" % df_name)
    sys.exit(3)
obj = ns[df_name]


def to_table(o):
    if isinstance(o, pa.Table):
        return o
    if isinstance(o, pa.RecordBatch):
        return pa.Table.from_batches([o])
    mod = type(o).__module__ or ""
    if mod.startswith("polars"):
        if hasattr(o, "collect"):
            o = o.collect()
        return o.to_arrow()
    if mod.startswith("pandas"):
        return pa.Table.from_pandas(o, preserve_index=False)
    if isinstance(o, dict):
        return pa.table(o)
    if isinstance(o, list):
        return pa.Table.from_pylist(o)
    raise TypeError("cannot turn %r into a table" % type(o).__name__)


t = to_table(obj)
if max_rows > 0 and t.num_rows > max_rows:
    t = t.slice(0, max_rows)
pq.write_table(t, out_path)
"""


class ExcelService:
    def __init__(
        self,
        settings: Settings,
        *,
        fs: FsService,
        pyenv: PyEnvService,
    ) -> None:
        self.settings = settings
        self.fs = fs
        self.pyenv = pyenv

    # -- identity -----------------------------------------------------------

    def info(self) -> ExcelInfo:
        return ExcelInfo(
            node_id=self.settings.node_id,
            node_name=self.settings.app_name,
            version=__version__,
            capabilities=["python", "fs.read", "fs.write", "fs.tree"],
        )

    # -- serialization ------------------------------------------------------

    @staticmethod
    def serialize_table(table: pa.Table, fmt: str) -> tuple[bytes, str]:
        """Encode *table* for the wire; content types come from MimeTypes.

        Arrow uses the IPC *stream* framing (``MimeTypes.ARROW_STREAM``),
        which the project's transport owns; Parquet is the Power Query
        default; JSON is records.
        """
        if fmt == "arrow":
            return transport.write_arrow_stream_bytes(table), MimeTypes.ARROW_STREAM.value
        if fmt == "json":
            import json
            return json.dumps(table.to_pylist(), default=str).encode(), MimeTypes.JSON.value
        if fmt == "parquet":
            return transport.write_parquet_bytes(table), MimeTypes.PARQUET.value
        raise BadRequestError(
            f"Unknown table format {fmt!r}; expected one of parquet/arrow/json."
        )

    # -- python execution ---------------------------------------------------

    async def run_python(self, req: ExcelQueryRequest) -> pa.Table:
        return await run_in_threadpool(partial(self._run_python, req))

    def _run_python(self, req: ExcelQueryRequest) -> pa.Table:
        if not req.code.strip():
            raise BadRequestError("code must not be empty")

        python_bin = sys.executable
        if req.env:
            resolved = self.pyenv.python_path_by_name(req.env)
            if resolved is None:
                raise NotFoundError(
                    f"PyEnv {req.env!r} not found or not ready."
                )
            python_bin = resolved

        with tempfile.TemporaryDirectory(prefix="ygg-excel-") as tmp:
            tmp_path = Path(tmp)
            code_file = tmp_path / "snippet.py"
            code_file.write_text(req.code)
            out_file = tmp_path / "out.parquet"

            if req.packages:
                self._pip_install(python_bin, req.packages)

            # Run env = node environment + the PyEnv's stored vars + the
            # request's own vars (request wins). None when nothing extra.
            stored = self.pyenv.env_vars_for_name(req.env) if req.env else {}
            extra = {**stored, **{str(k): str(v) for k, v in req.env_vars.items()}}
            run_env = None
            if extra:
                import os
                run_env = {**os.environ, **extra}

            timeout = req.timeout or self.settings.max_python_timeout
            try:
                proc = subprocess.run(
                    [python_bin, "-c", _DRIVER, str(out_file), req.df_name,
                     str(req.max_rows or 0), str(code_file)],
                    capture_output=True, text=True, timeout=timeout, env=run_env,
                )
            except subprocess.TimeoutExpired:
                raise APITimeoutError(f"snippet exceeded its {timeout:g}s timeout")
            if proc.returncode != 0:
                raise BadRequestError(
                    (proc.stderr or proc.stdout or "snippet failed").strip()
                )
            if not out_file.exists():
                raise BadRequestError("snippet produced no table output")
            return transport.read_parquet_bytes(out_file.read_bytes())

    @staticmethod
    def _pip_install(python_bin: str, packages: list[str]) -> None:
        import shutil
        uv = shutil.which("uv")
        cmd = (
            [uv, "pip", "install", "--python", python_bin, *packages]
            if uv else [python_bin, "-m", "pip", "install", *packages]
        )
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise APITimeoutError("package install exceeded its 600s timeout")
        if proc.returncode != 0:
            raise BadRequestError(
                f"package install failed: {(proc.stderr or proc.stdout).strip()}"
            )

    # -- file <-> table -----------------------------------------------------

    async def read_table(self, path: str, fmt: str | None = None) -> pa.Table:
        return await run_in_threadpool(partial(self._read_table, path, fmt))

    def _read_table(self, path: str, fmt: str | None) -> pa.Table:
        # ``FsService._resolve`` does the join + traversal guard (stays
        # under the node root); the yggdrasil Path then owns the IO —
        # ``as_media`` dispatches parquet/csv/json/arrow by extension (or
        # the ``source_format`` override) straight to a typed Arrow read.
        resolved = self.fs._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"File {path!r} not found")
        media = MediaType.from_(fmt) if fmt else None
        try:
            with YggPath.from_(str(resolved)).open("rb", media_type=media) as bio:
                return bio.read_arrow_table()
        except (KeyError, NotImplementedError) as exc:
            raise BadRequestError(
                f"cannot read {path!r} as a table "
                f"({fmt or resolved.suffix.lstrip('.') or 'unknown'}): {exc}"
            )

    async def write_table(self, path: str, data: bytes, content_type: str) -> ExcelWriteResponse:
        return await run_in_threadpool(partial(self._write_table, path, data, content_type))

    def _write_table(self, path: str, data: bytes, content_type: str) -> ExcelWriteResponse:
        if not data:
            raise BadRequestError("empty request body — nothing to write")
        ct = (content_type or "").lower()
        try:
            # Arrow stream framing isn't the media layer's IPC *file* leaf,
            # so it routes through transport; everything else (parquet /
            # csv / json) decodes through the IO/media layer.
            if "arrow.stream" in ct:
                table = transport.read_arrow_stream(data)
            else:
                src = MediaType.from_(ct, default=None) or MediaType.from_("parquet")
                with IO(data).open("rb", media_type=src) as bio:
                    table = bio.read_arrow_table()
        except BadRequestError:
            raise
        except Exception as exc:
            raise BadRequestError(
                f"could not parse uploaded table (content-type {ct or 'unknown'!r}): {exc}"
            )

        resolved = self.fs._resolve(path)
        # Write through the Path/media layer — extension picks the encoder
        # (.csv → CSV, else Parquet); parent dirs are created on write.
        target = MediaType.from_(resolved.suffix.lstrip(".") or "parquet", default=None) \
            or MediaType.from_("parquet")
        with YggPath.from_(str(resolved)).open("wb", media_type=target) as bio:
            bio.write_arrow_table(table)
        return ExcelWriteResponse(
            path=path,
            rows=table.num_rows,
            columns=table.num_columns,
            bytes_written=resolved.stat().st_size,
        )

    # -- navigation ---------------------------------------------------------

    async def tree(self, path: str = "", depth: int = 3) -> ExcelTreeResponse:
        root = self.fs._resolve(path)
        nodes = await run_in_threadpool(partial(self._walk, root, depth, 0))
        return ExcelTreeResponse(node_id=self.settings.node_id, root=path or "/", tree=nodes)

    def _walk(self, directory: Path, max_depth: int, level: int) -> list[ExcelTreeNode]:
        if not directory.is_dir() or level >= max_depth:
            return []
        out: list[ExcelTreeNode] = []
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return []
        for child in entries:
            try:
                rel = str(child.relative_to(self.fs._root))
            except ValueError:
                rel = child.name
            is_dir = child.is_dir()
            node = ExcelTreeNode(
                path=rel,
                name=child.name,
                is_dir=is_dir,
                size=0 if is_dir else child.stat().st_size,
                children=self._walk(child, max_depth, level + 1) if is_dir else [],
            )
            out.append(node)
        return out
