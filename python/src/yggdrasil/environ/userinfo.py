from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence

from yggdrasil.io.url import URL

__all__ = [
    "UserInfo",
    "get_user_info",
    "parse_name_from_email",
]

DatabricksLinkKind = Literal["auto", "job_run", "notebook_id", "workspace_path"]

_CURRENT_CACHE: UserInfo | None = None

_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:(?:[\\/]|$)")
_UNC_RE = re.compile(r"^(\\\\|//)([^\\/]+)[\\/]+([^\\/]+)(.*)$")

_SETUP_NAME_RE = re.compile(r"""(?s)\bname\s*=\s*["']([^"']+)["']""")
_SETUP_VERSION_RE = re.compile(r"""(?s)\bversion\s*=\s*["']([^"']+)["']""")
_SPLIT_RE = re.compile(r"[._+]+")

_LASTNAME_PARTICLES = {
    "da",
    "de",
    "del",
    "della",
    "der",
    "di",
    "du",
    "des",
    "la",
    "le",
    "les",
    "van",
    "von",
    "den",
    "ten",
    "ter",
    "st",
    "st.",
    "saint",
    "y",
}


def normalize_abs_path_for_url(path: str) -> str:
    """
    Normalize an absolute filesystem path into a URL-friendly POSIX-style path.

    Examples:
        /a//b///c              -> /a/b/c
        C:\\Users\\me\\proj    -> /C:/Users/me/proj
        \\\\server\\share\\dir -> //server/share/dir
    """
    if not path:
        return "/"

    s = os.path.expanduser(str(path).strip())
    if not s:
        return "/"

    if _looks_like_posix_absolute(s):
        return _collapse_slashes(_to_posix(s)) or "/"

    unc = _normalize_unc_path(s)
    if unc is not None:
        return unc

    drive = _normalize_windows_drive_path(s)
    if drive is not None:
        return drive

    try:
        resolved = Path(s).expanduser().resolve(strict=False).as_posix()
    except Exception:
        resolved = _to_posix(s)

    if not resolved.startswith("/"):
        resolved = f"/{resolved}"

    return _collapse_slashes(resolved) or "/"


def _looks_like_posix_absolute(path: str) -> bool:
    return path.startswith("/") and not _WIN_DRIVE_RE.match(path) and not _UNC_RE.match(path)


def _to_posix(path: str) -> str:
    return path.replace("\\", "/")


def _collapse_slashes(path: str) -> str:
    while "//" in path:
        path = path.replace("//", "/")
    return path


def _normalize_unc_path(path: str) -> str | None:
    match = _UNC_RE.match(path)
    if not match:
        return None

    server, share, rest = match.group(2), match.group(3), match.group(4) or ""
    rest = _to_posix(rest)
    if rest and not rest.startswith("/"):
        rest = f"/{rest}"

    return f"//{server}/{share}{_collapse_slashes(rest)}"


def _normalize_windows_drive_path(path: str) -> str | None:
    if not _WIN_DRIVE_RE.match(path):
        return None

    out = _to_posix(path)
    if len(out) >= 2 and out[1] == ":" and (len(out) == 2 or out[2] != "/"):
        out = f"{out[:2]}/{out[2:]}"
    return f"/{_collapse_slashes(out)}"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _load_toml(path: Path) -> dict:
    try:
        import tomllib

        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            import tomli

            return tomli.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _find_upwards(start: str, filename: str) -> Path | None:
    if not start:
        return None

    root = Path(start).resolve()
    for directory in (root, *root.parents):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def _parse_setup_py(path: Path) -> tuple[str | None, str | None]:
    text = _read_text(path)
    if not text:
        return None, None

    name = _match_group(_SETUP_NAME_RE, text)
    version = _match_group(_SETUP_VERSION_RE, text)
    return _clean_str(name), _clean_str(version)


def _match_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _clean_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _parse_pyproject(path: Path) -> tuple[str | None, str | None]:
    data = _load_toml(path)
    if not isinstance(data, dict) or not data:
        return None, None

    name: str | None = None
    version: str | None = None

    project = data.get("project")
    if isinstance(project, dict):
        name = _clean_str(project.get("name"))
        version = _clean_str(project.get("version"))

    tool = data.get("tool")
    if isinstance(tool, dict):
        poetry = tool.get("poetry")
        if isinstance(poetry, dict):
            name = name or _clean_str(poetry.get("name"))
            version = version or _clean_str(poetry.get("version"))

    return name, version


def _infer_project(cwd: str) -> tuple[str | None, str | None]:
    pyproject = _find_upwards(cwd, "pyproject.toml")
    if pyproject:
        name, version = _parse_pyproject(pyproject)
        if name or version:
            return name, version

    setup_py = _find_upwards(cwd, "setup.py")
    if setup_py:
        name, version = _parse_setup_py(setup_py)
        if name or version:
            return name, version

    return None, None


def parse_name_from_email(email: str) -> tuple[str, str, str] | None:
    value = (email or "").strip()
    if "@" not in value:
        return None

    local, domain = value.rsplit("@", 1)
    local = local.strip().split("+", 1)[0]
    domain = domain.strip().lower()

    if not local or not domain:
        return None

    tokens = [_strip_trailing_digits(token) for token in _SPLIT_RE.split(local) if token]
    tokens = [token for token in tokens if token]
    if len(tokens) < 2:
        return None

    first = _smart_title(tokens[0])
    last = _build_last_name(tokens)

    return first, last, domain


def _strip_trailing_digits(value: str) -> str:
    return re.sub(r"\d+$", "", value)


def _smart_title(token: str) -> str:
    token = token.strip().replace("’", "'")
    if not token:
        return token

    chunks: list[str] = []
    for chunk in token.split("-"):
        chunks.append(_format_name_chunk(chunk))

    return "-".join(chunks)


def _format_name_chunk(chunk: str) -> str:
    if "'" not in chunk:
        return chunk[:1].upper() + chunk[1:].lower()

    left, right = chunk.split("'", 1)
    left_lower = left.lower()
    left_fmt = left_lower if left_lower in _LASTNAME_PARTICLES else left[:1].upper() + left[1:].lower()
    right_fmt = right[:1].upper() + right[1:].lower() if right else ""
    return f"{left_fmt}’{right_fmt}" if right_fmt else f"{left_fmt}’"


def _build_last_name(tokens: list[str]) -> str:
    parts = [tokens[-1]]
    index = len(tokens) - 2

    while index >= 1:
        candidate = tokens[index].replace("’", "'").lower().strip(".")
        if candidate in _LASTNAME_PARTICLES:
            parts.insert(0, tokens[index])
            index -= 1
            continue

        if len(tokens) >= 4 and len(parts) == 1:
            parts.insert(0, tokens[index])
            index -= 1
            continue

        break

    last = " ".join(_smart_title(part) for part in parts)
    return " ".join(
        word.lower() if word.replace("’", "'").lower().strip(".") in _LASTNAME_PARTICLES else word
        for word in last.split()
    )


@dataclass(frozen=True, slots=True)
class UserInfo:
    email: str | None
    key: str
    hostname: str
    url: URL | None
    git_url: URL | None
    product: str | None
    product_version: str | None

    @classmethod
    def current(cls, *, refresh: bool = False) -> UserInfo:
        global _CURRENT_CACHE

        if _CURRENT_CACHE is not None and not refresh:
            return _CURRENT_CACHE

        hostname = socket.gethostname()
        cwd = _safe_getcwd()

        project, project_version = _infer_project(cwd) if cwd else (None, None)
        git_url = _git_url_from_info(_git_info(cwd) if cwd else None)

        info = cls(
            email=_get_upn_email() or _guess_email_from_env(),
            key=_get_key(),
            hostname=hostname,
            url=_databricks_current_url(kind="auto"),
            git_url=git_url,
            product=project,
            product_version=project_version,
        )

        _CURRENT_CACHE = info
        return info

    def with_email(self, email: str | None) -> UserInfo:
        return UserInfo(
            email=email,
            key=self.key,
            hostname=self.hostname,
            url=self.url,
            git_url=self.git_url,
            product=self.product,
            product_version=self.product_version,
        )


def get_user_info(*, refresh: bool = False) -> UserInfo:
    return UserInfo.current(refresh=refresh)


def _clear_cache() -> None:
    global _CURRENT_CACHE
    _CURRENT_CACHE = None


def _safe_getcwd() -> str:
    try:
        return os.getcwd()
    except Exception:
        return ""


def _is_databricks_runtime() -> bool:
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))


def _get_key() -> str:
    if _is_databricks_runtime():
        dbx_user = _get_dbx_user()
        if dbx_user:
            return dbx_user

    whoami = _run_quiet(["whoami"])
    if whoami:
        return whoami.strip()

    return os.getenv("USERNAME") or os.getenv("USER") or os.getenv("LOGNAME") or "unknown"


def _get_upn_email() -> str | None:
    if _is_databricks_runtime():
        dbx_user = _get_dbx_user()
        if dbx_user and "@" in dbx_user:
            return dbx_user

    upn = _run_quiet(["whoami", "/UPN"])
    upn = _clean_str(upn)
    if not upn or upn.lower() == "null" or "@" not in upn:
        return None

    return upn


def _guess_email_from_env() -> str | None:
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL", "EMAIL"):
        value = _clean_str(os.getenv(key))
        if value and "@" in value:
            return value
    return None


def _get_dbx_user() -> str | None:
    try:
        from yggdrasil.databricks import DatabricksClient

        client = DatabricksClient(product="current", product_version="0.0.0")
        return client.iam.users.current_user.email
    except Exception:
        return None


def _run_quiet(cmd: Sequence[str]) -> str | None:
    try:
        output = subprocess.check_output(
            list(cmd),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return _clean_str(output)


def _ctx_tags() -> Mapping[str, str]:
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # type: ignore[name-defined]
        payload = json.loads(ctx.toJson())
        tags = payload.get("tags") or {}
        return {str(key): str(value) for key, value in tags.items()}
    except Exception:
        return {}


def _pick(*values: Optional[str]) -> str | None:
    for value in values:
        cleaned = _clean_str(value)
        if cleaned:
            return cleaned
    return None


def _strip_scheme(value: str) -> str:
    return value.removeprefix("https://").removeprefix("http://")


def _databricks_current_url(*, kind: DatabricksLinkKind = "auto") -> URL | None:
    if not _is_databricks_runtime():
        return None

    tags = _ctx_tags()

    host = _pick(
        tags.get("browserHostName"),
        _strip_scheme(os.getenv("DATABRICKS_HOST", "")),
        _strip_scheme(os.getenv("DATABRICKS_WORKSPACE_URL", "")),
    )
    org_id = _pick(
        tags.get("orgId"),
        os.getenv("DATABRICKS_ORG_ID"),
        os.getenv("DATABRICKS_WORKSPACE_ID"),
    )
    if not host or not org_id:
        return None

    base = URL.parse_dict(
        {
            "scheme": "https",
            "host": host,
            "path": "/",
            "query": f"o={org_id}",
        }
    )

    job_id = _pick(tags.get("jobId"), tags.get("job_id"), os.getenv("DATABRICKS_JOB_ID"))
    run_id = _pick(
        tags.get("jobRunId"),
        tags.get("job_run_id"),
        tags.get("runId"),
        os.getenv("DATABRICKS_RUN_ID"),
    )
    notebook_id = _pick(tags.get("notebookId"), tags.get("notebook_id"))
    notebook_path = _pick(tags.get("notebookPath"))

    candidates: list[tuple[DatabricksLinkKind, str | None]] = [
        ("job_run", f"job/{job_id}/run/{run_id}" if job_id and run_id else None),
        ("notebook_id", f"notebook/{notebook_id}" if notebook_id else None),
        ("workspace_path", f"workspace{notebook_path}" if notebook_path else None),
    ]

    for candidate_kind, fragment in candidates:
        if kind in ("auto", candidate_kind) and fragment:
            return base.with_fragment(fragment)

    return None


def _resolve_gitdir(dotgit: Path) -> Path | None:
    if dotgit.is_dir():
        return dotgit

    if not dotgit.is_file():
        return None

    text = _read_text(dotgit)
    if not text:
        return None

    first_line = text.strip().splitlines()[0].strip()
    if not first_line.lower().startswith("gitdir:"):
        return None

    relative_gitdir = first_line.split(":", 1)[1].strip()
    resolved = (dotgit.parent / relative_gitdir).resolve()
    return resolved if resolved.exists() else None


def _find_git_root(start: str) -> tuple[Path, Path] | None:
    if not start:
        return None

    root = Path(start).resolve()
    for directory in (root, *root.parents):
        dotgit = directory / ".git"
        if not dotgit.exists():
            continue

        gitdir = _resolve_gitdir(dotgit)
        if gitdir is not None:
            return directory, gitdir

    return None


def _read_packed_ref(gitdir: Path, ref: str) -> str | None:
    text = _read_text(gitdir / "packed-refs")
    if not text:
        return None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "^")):
            continue

        parts = line.split(" ")
        if len(parts) == 2 and parts[1] == ref and len(parts[0]) >= 8:
            return parts[0]

    return None


def _git_head_info(repo_root: Path, gitdir: Path) -> dict[str, str] | None:
    head_text = _read_text(gitdir / "HEAD")
    if not head_text:
        return None

    head_text = head_text.strip()
    branch: str | None = None

    if head_text.startswith("ref:"):
        ref = head_text.split(":", 1)[1].strip()
        if ref.startswith("refs/heads/"):
            branch = ref.removeprefix("refs/heads/")

    info: dict[str, str] = {"git_root": str(repo_root)}
    if branch:
        info["git_branch"] = branch

    return info


def _git_remote_origin(gitdir: Path) -> str | None:
    config = _read_text(gitdir / "config")
    if not config:
        return None

    current_remote: str | None = None
    remotes: dict[str, str] = {}

    for raw_line in config.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue

        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if section.lower().startswith("remote "):
                current_remote = _extract_quoted_value(section)
            else:
                current_remote = None
            continue

        if current_remote and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key.lower() == "url":
                remotes[current_remote] = value

    return remotes.get("origin") or next(iter(remotes.values()), None)


def _extract_quoted_value(value: str) -> str | None:
    start = value.find('"')
    end = value.rfind('"')
    if start == -1 or end == -1 or end <= start:
        return None
    return value[start + 1 : end]


def _git_info(cwd: str) -> dict[str, str] | None:
    found = _find_git_root(cwd)
    if not found:
        return None

    repo_root, gitdir = found
    info = _git_head_info(repo_root, gitdir) or {}

    remote = _git_remote_origin(gitdir)
    if remote:
        info["git_remote"] = remote

    return info if info.get("git_remote") else None


def _normalize_git_remote(remote: str) -> str:
    remote = remote.strip()

    scp_like = re.match(r"^git@([^:]+):(.+)$", remote)
    if scp_like:
        host, path = scp_like.group(1), scp_like.group(2)
        return f"https://{host}/{path.removesuffix('.git')}"

    ssh_like = re.match(r"^ssh://git@([^/]+)/(.+)$", remote)
    if ssh_like:
        host, path = ssh_like.group(1), ssh_like.group(2)
        return f"https://{host}/{path.removesuffix('.git')}"

    if remote.startswith(("http://", "https://")):
        return remote.removesuffix(".git")

    return remote


def _git_url_from_info(git: dict[str, str] | None) -> URL | None:
    if not git:
        return None

    remote = git.get("git_remote")
    if not remote:
        return None

    return URL.parse_str(_normalize_git_remote(remote), normalize=True)