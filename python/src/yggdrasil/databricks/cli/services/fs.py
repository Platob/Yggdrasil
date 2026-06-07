"""``ygg databricks fs`` — a filesystem CLI over the :class:`DatabricksPath`
abstraction, uniform across Workspace / Volumes / DBFS.

Every ``<uri>`` is resolved with ``DatabricksPath.from_(uri, client=...)`` so the
same verbs (``ls``, ``cat``, ``put``/``get``, ``cp``/``mv``, ``rm`` …) work — and
``cp``/``mv`` move bytes **across surfaces** (e.g. a Workspace file into a Volume)
through one read/write contract.
"""
from __future__ import annotations

import sys
from typing import Any


def _path(args: Any, build_client: Any, uri: str) -> Any:
    from yggdrasil.databricks.path import DatabricksPath

    return DatabricksPath.from_(uri, client=build_client(args))


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{size}B"


class FSCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser("fs", help="Filesystem ops over Workspace/Volumes/DBFS.")
        sub = parser.add_subparsers(dest="fs_action")

        ls = sub.add_parser("ls", help="List a directory.")
        ls.add_argument("uri")
        ls.add_argument("-l", "--long", action="store_true", help="Long format (size, kind).")
        ls.add_argument("-r", "--recursive", action="store_true")
        ls.set_defaults(handler=cls._ls)

        cat = sub.add_parser("cat", help="Print a file's contents to stdout.")
        cat.add_argument("uri")
        cat.set_defaults(handler=cls._cat)

        write = sub.add_parser("write", help="Write text/stdin/local file to a path.")
        write.add_argument("uri")
        write.add_argument("--data", default=None, help="Literal text to write.")
        write.add_argument("--file", default=None, help="Local file to read bytes from.")
        write.set_defaults(handler=cls._write)

        put = sub.add_parser("put", help="Upload a local file to a remote path.")
        put.add_argument("local")
        put.add_argument("uri")
        put.set_defaults(handler=cls._put)

        get = sub.add_parser("get", help="Download a remote file to a local path.")
        get.add_argument("uri")
        get.add_argument("local")
        get.set_defaults(handler=cls._get)

        mkdir = sub.add_parser("mkdir", help="Create a directory (parents ok).")
        mkdir.add_argument("uri")
        mkdir.set_defaults(handler=cls._mkdir)

        mknb = sub.add_parser(
            "create-notebook",
            help="Create a notebook at a /Workspace path.",
        )
        mknb.add_argument("uri")
        mknb.add_argument(
            "--language", default="PYTHON",
            help="Notebook language: PYTHON | SQL | SCALA | R (default PYTHON).",
        )
        mknb.add_argument("--data", default=None, help="Literal notebook source.")
        mknb.add_argument("--file", default=None, help="Local source file to import.")
        mknb.add_argument(
            "--overwrite", action="store_true",
            help="Replace an existing notebook.",
        )
        mknb.set_defaults(handler=cls._create_notebook)

        rm = sub.add_parser("rm", help="Remove a file or directory.")
        rm.add_argument("uri")
        rm.add_argument("-r", "--recursive", action="store_true")
        rm.set_defaults(handler=cls._rm)

        stat = sub.add_parser("stat", help="Show size / kind / mtime of a path.")
        stat.add_argument("uri")
        stat.set_defaults(handler=cls._stat)

        cp = sub.add_parser("cp", help="Copy across surfaces (Workspace↔Volume↔DBFS).")
        cp.add_argument("src")
        cp.add_argument("dst")
        cp.set_defaults(handler=cls._cp)

        mv = sub.add_parser("mv", help="Move across surfaces (copy then delete).")
        mv.add_argument("src")
        mv.add_argument("dst")
        mv.set_defaults(handler=cls._mv)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    # -- handlers --------------------------------------------------------
    @classmethod
    def _ls(cls, args: Any, build_client: Any) -> int:
        path = _path(args, build_client, args.uri)
        children = path.ls(recursive=True) if args.recursive else path.iterdir()
        for child in children:
            if args.long:
                kind = "d" if child.is_dir() else "-"
                size = "-" if child.is_dir() else _human(child.size or 0)
                sys.stdout.write(f"{kind}\t{size:>7}\t{child.full_path()}\n")
            else:
                suffix = "/" if child.is_dir() else ""
                sys.stdout.write(f"{child.name}{suffix}\n")
        return 0

    @classmethod
    def _cat(cls, args: Any, build_client: Any) -> int:
        path = _path(args, build_client, args.uri)
        sys.stdout.buffer.write(path.read_bytes())
        return 0

    @classmethod
    def _write(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style
        from pathlib import Path

        if args.data is not None:
            data = args.data.encode()
        elif args.file is not None:
            data = Path(args.file).read_bytes()
        else:
            data = sys.stdin.buffer.read()
        path = _path(args, build_client, args.uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data, overwrite=True)
        style.ok(f"wrote {_human(len(data))} → {path.full_path()}")
        return 0

    @classmethod
    def _put(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style
        from pathlib import Path

        data = Path(args.local).read_bytes()
        path = _path(args, build_client, args.uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data, overwrite=True)
        style.ok(f"uploaded {_human(len(data))} → {path.full_path()}")
        return 0

    @classmethod
    def _get(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style
        from pathlib import Path

        path = _path(args, build_client, args.uri)
        data = path.read_bytes()
        Path(args.local).write_bytes(data)
        style.ok(f"downloaded {_human(len(data))} → {args.local}")
        return 0

    @classmethod
    def _mkdir(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        path = _path(args, build_client, args.uri)
        path.mkdir(parents=True, exist_ok=True)
        style.ok(f"created {path.full_path()}")
        return 0

    @classmethod
    def _create_notebook(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style
        from pathlib import Path
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath

        path = _path(args, build_client, args.uri)
        if not isinstance(path, WorkspacePath):
            sys.stderr.write(
                f"create-notebook requires a /Workspace path, "
                f"got {path.full_path()}\n"
            )
            return 1
        if args.data is not None:
            content = args.data
        elif args.file is not None:
            content = Path(args.file).read_text()
        else:
            content = None
        path.create_notebook(
            args.language, content=content, overwrite=args.overwrite,
        )
        style.ok(f"created {args.language.upper()} notebook → {path.full_path()}")
        return 0

    @classmethod
    def _rm(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        path = _path(args, build_client, args.uri)
        if args.recursive:
            path.remove(recursive=True)
        else:
            path.unlink()
        style.ok(f"removed {path.full_path()}")
        return 0

    @classmethod
    def _stat(cls, args: Any, build_client: Any) -> int:
        import datetime as _dt

        path = _path(args, build_client, args.uri)
        if not path.exists():
            sys.stderr.write(f"no such path: {args.uri}\n")
            return 1
        kind = "directory" if path.is_dir() else "file"
        sys.stdout.write(f"path: {path.full_path()}\n")
        sys.stdout.write(f"kind: {kind}\n")
        if path.is_file():
            sys.stdout.write(f"size: {path.size} ({_human(path.size or 0)})\n")
        mtime = getattr(path, "mtime", None)
        if mtime:
            sys.stdout.write(f"mtime: {_dt.datetime.fromtimestamp(mtime).isoformat()}\n")
        return 0

    @classmethod
    def _cp(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        from yggdrasil.databricks.path import DatabricksPath

        src = DatabricksPath.from_(args.src, client=client)
        dst = DatabricksPath.from_(args.dst, client=client)
        data = src.read_bytes()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data, overwrite=True)
        style.ok(f"copied {src.full_path()} → {dst.full_path()} ({_human(len(data))})")
        return 0

    @classmethod
    def _mv(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        from yggdrasil.databricks.path import DatabricksPath

        src = DatabricksPath.from_(args.src, client=client)
        dst = DatabricksPath.from_(args.dst, client=client)
        data = src.read_bytes()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data, overwrite=True)
        src.unlink()
        style.ok(f"moved {src.full_path()} → {dst.full_path()} ({_human(len(data))})")
        return 0
