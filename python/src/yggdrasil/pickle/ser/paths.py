from __future__ import annotations

import fnmatch
import io
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import ClassVar, Mapping

from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "MAX_INLINE_DIR_BYTES",
    "PathSerialized",
]

MAX_INLINE_DIR_BYTES = 1024 * 1024  # 1 MiB


def _metadata_merge(
    base: Mapping[bytes, bytes] | None,
    extra: Mapping[bytes, bytes] | None = None,
) -> dict[bytes, bytes] | None:
    if not base and not extra:
        return None
    out: dict[bytes, bytes] = {}
    if base:
        out.update(base)
    if extra:
        out.update(extra)
    return out


def _current_os_name() -> bytes:
    # Path semantics matter more than specific kernel branding.
    return b"windows" if os.name == "nt" else b"posix"


def _sanitize_path_part(part: str) -> str:
    part = part.replace("\\", "_").replace("/", "_").replace(":", "_")
    part = re.sub(r"[\x00-\x1f]", "_", part)
    return part or "_"


def _rebuild_path(raw: str, *, source_os: bytes) -> Path:
    """
    Rebuild a serialized path on the current runtime.

    Same OS family:
        - preserve exactly via Path(raw)

    Cross-OS:
        - windows -> posix:
            C:\\tmp\\x -> /c/tmp/x
            tmp\\x     -> tmp/x
        - posix -> windows:
            /var/tmp/x -> \\var\\tmp\\x
            var/tmp/x  -> var\\tmp\\x
    """
    current_os = _current_os_name()

    if source_os == current_os:
        return Path(raw)

    if source_os == b"posix":
        p = PurePosixPath(raw)

        if current_os == b"windows":
            parts = [part for part in p.parts if part != "/"]
            if p.is_absolute():
                if not parts:
                    return Path("\\")
                return Path("\\" + "\\".join(parts))
            return Path("\\".join(parts))

        return Path(raw)

    if source_os == b"windows":
        p = PureWindowsPath(raw)

        if current_os == b"posix":
            if not p.is_absolute() and not p.drive:
                parts = [part for part in p.parts if part not in ("\\", "/")]
                return Path("/".join(parts))

            drive = p.drive.rstrip(":").lower() if p.drive else "_"
            drive = _sanitize_path_part(drive)
            tail_parts = [
                _sanitize_path_part(part)
                for part in p.parts
                if part not in ("\\", "/", p.anchor)
            ]
            translated = "/" + "/".join([drive, *tail_parts]) if tail_parts else f"/{drive}"
            return Path(translated)

        return Path(raw)

    return Path(raw)


def _dir_total_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _should_exclude_path(parts: tuple[str, ...]) -> bool:
    exact = {
        # Python
        "__pycache__",
        "__pypackages__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".tox",
        ".nox",
        ".coverage",
        "htmlcov",
        ".eggs",
        ".ipynb_checkpoints",

        # VCS
        ".git",
        ".svn",
        ".hg",
        ".bzr",

        # Virtual envs
        ".venv",
        "venv",
        "env",
        ".env",

        # Node / frontend
        "node_modules",
        ".pnpm-store",
        ".npm",
        ".yarn",
        ".yarn-cache",
        ".yarnrc",
        ".parcel-cache",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".turbo",
        ".angular",
        ".cache-loader",
        ".eslintcache",
        ".stylelintcache",

        # Rust / C / C++
        "target",
        "cmake-build-debug",
        "cmake-build-release",
        "cmake-build-relwithdebinfo",
        "cmake-build-minsizerel",
        "CMakeFiles",
        "CMakeCache.txt",
        "compile_commands.json",
        ".cache",
        ".ccls-cache",
        ".clangd",

        # Java / JVM
        ".gradle",
        ".mvn",
        "out",

        # IDE / editors
        ".idea",
        ".vscode",
        ".vs",
        ".fleet",

        # OS junk
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",

        # Infra / workflow / notebooks / data tooling
        ".dvc",
        ".snakemake",
        ".meltano",
        ".dbx",
        ".databricks",
        ".terraform",
        ".terragrunt-cache",
        ".ipynb_checkpoints",

        # Build / reports / temp
        "build",
        "dist",
        "coverage",
        "tmp",
        "temp",
    }

    patterns = {
        "*.egg-info",
        "*.dist-info",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        "*.dylib",
        "*.dll",
        "*.o",
        "*.obj",
        "*.a",
        "*.lib",
        "*.class",
    }

    for part in parts:
        if part in exact:
            return True
        if any(fnmatch.fnmatch(part, pattern) for pattern in patterns):
            return True

    return False


def _zip_directory_to_bytes(root: Path) -> bytes:
    """
    Zip a directory into bytes using stdlib io.BytesIO.
    """
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if not p.exists():
                continue

            arcname = p.relative_to(root)

            if _should_exclude_path(arcname.parts):
                continue

            if p.is_dir():
                if arcname.parts:
                    zf.writestr(f"{arcname.as_posix().rstrip('/')}/", b"")
            elif p.is_file():
                zf.write(filename=p, arcname=arcname.as_posix())

    return buf.getvalue()


def _extract_zip_bytes_to_temp_dir(data: bytes, *, dirname: str) -> Path:
    tmp_root = Path(tempfile.mkdtemp(prefix="ygg_path_dir_"))
    out_dir = tmp_root / dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(data), mode="r") as zf:
        zf.extractall(out_dir)

    return out_dir


def _write_file_bytes_to_temp_path(data: bytes, *, filename: str) -> Path:
    suffix = Path(filename).suffix
    fd, name = tempfile.mkstemp(prefix="ygg_path_file_", suffix=suffix)
    os.close(fd)
    path = Path(name)
    path.write_bytes(data)
    return path


@dataclass(frozen=True, slots=True)
class PathSerialized(Serialized[Path]):
    TAG: ClassVar[int] = Tags.PATH

    @property
    def value(self) -> Path:
        metadata = self.metadata or {}
        mode = metadata.get(b"path_mode", b"path")

        if mode == b"path":
            raw = self.decode().decode("utf-8")
            source_os = metadata.get(b"path_os", b"posix")
            return _rebuild_path(raw, source_os=source_os)

        if mode == b"file":
            filename = metadata.get(b"filename", b"payload.bin").decode("utf-8", errors="replace")
            return _write_file_bytes_to_temp_path(self.decode(), filename=filename)

        if mode == b"dir_zip":
            dirname = metadata.get(b"dirname", b"directory").decode("utf-8", errors="replace")
            return _extract_zip_bytes_to_temp_dir(self.decode(), dirname=dirname)

        raise ValueError(f"Unsupported PATH payload mode: {mode!r}")

    def as_python(self) -> Path:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if isinstance(obj, Path):
            return cls.from_value(obj, metadata=metadata, codec=codec)
        return None

    @classmethod
    def from_value(
        cls,
        path: Path,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        path = path.expanduser()
        source_os = _current_os_name()

        if path.is_file():
            payload = path.read_bytes()
            merged = _metadata_merge(
                metadata,
                {
                    b"path_mode": b"file",
                    b"filename": path.name.encode("utf-8", errors="replace"),
                    b"path_os": source_os,
                },
            )
            return cls.build(
                tag=cls.TAG,
                data=payload,
                metadata=merged,
                codec=codec,
            )

        if path.is_dir():
            try:
                total = _dir_total_bytes(path)
            except Exception:
                total = MAX_INLINE_DIR_BYTES + 1

            if total < MAX_INLINE_DIR_BYTES:
                payload = _zip_directory_to_bytes(path)
                merged = _metadata_merge(
                    metadata,
                    {
                        b"path_mode": b"dir_zip",
                        b"dirname": path.name.encode("utf-8", errors="replace"),
                        b"path_os": source_os,
                    },
                )
                return cls.build(
                    tag=cls.TAG,
                    data=payload,
                    metadata=merged,
                    codec=codec,
                )

        merged = _metadata_merge(
            metadata,
            {
                b"path_mode": b"path",
                b"path_os": source_os,
            },
        )
        return cls.build(
            tag=cls.TAG,
            data=str(path).encode("utf-8"),
            metadata=merged,
            codec=codec,
        )


Tags.register_class(PathSerialized)

for cls in PathSerialized.__subclasses__():
    Tags.register_class(cls)
