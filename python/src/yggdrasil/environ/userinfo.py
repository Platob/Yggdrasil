# userinfo.py
"""
User identity + runtime context probe.

- Cross-platform best-effort identity:
  - OS: whoami / env
  - Windows: whoami /UPN for email
  - Databricks: Workspace current user if available

- Runtime URLs:
  - Databricks: clickable UI URL for current job run / notebook / workspace path
  - Local fallback: local://<hostname>/<normalized-absolute-cwd>

- Git URL (isolated):
  - Best-effort, no subprocess
  - https-ish repo URL + #<sha-or-branch>

- Project metadata:
  - Best-effort search upward from cwd for pyproject.toml or setup.py
  - Extracts: project (name), project_version (version)
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Literal

from yggdrasil.io.url import URL

__all__ = ["UserInfo", "get_user_info", "normalize_abs_path_for_url"]

_CURRENT_CACHE: "UserInfo | None" = None
DatabricksLinkKind = Literal["auto", "job_run", "notebook_id", "workspace_path"]

_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:(?:[\\/]|$)")
_UNC_RE = re.compile(r"^(\\\\|//)([^\\/]+)[\\/]+([^\\/]+)(.*)$")


# ----------------------------
# Path normalization for local:// URLs
# ----------------------------

def normalize_abs_path_for_url(cwd: str) -> str:
    """
    Normalize an absolute working directory into a POSIX-ish URL path.

    Key behaviors (important on Windows):
    - If input already looks POSIX-absolute (starts with '/'), keep it POSIX-ish:
        "/a//b///c" -> "/a/b/c"
      (avoid turning it into "/C:/a/b/c" due to Windows drive resolution)

    - Windows drive:
        "C:\\Users\\Nika\\proj" -> "/C:/Users/Nika/proj"

    - UNC:
        "\\\\server\\share\\dir" -> "//server/share/dir"
    """
    if not cwd:
        return "/"

    s = str(cwd).strip()
    if not s:
        return "/"

    s = os.path.expanduser(s)

    # If it already looks like a POSIX absolute path, normalize it as POSIX.
    # This avoids Windows resolving "/a" to "C:\\a" and then emitting "/C:/a".
    if s.startswith("/") and not _WIN_DRIVE_RE.match(s) and not _UNC_RE.match(s):
        out = s.replace("\\", "/")
        while "//" in out:
            out = out.replace("//", "/")
        return out or "/"

    # UNC -> //server/share/...
    m = _UNC_RE.match(s)
    if m:
        server, share, rest = m.group(2), m.group(3), m.group(4) or ""
        rest = rest.replace("\\", "/")
        rest = rest if rest.startswith("/") or not rest else "/" + rest
        while "//" in rest:
            rest = rest.replace("//", "/")
        return f"//{server}/{share}{rest}"

    # Windows drive -> /C:/...
    if _WIN_DRIVE_RE.match(s):
        s2 = s.replace("\\", "/")
        if len(s2) >= 2 and s2[1] == ":" and (len(s2) == 2 or s2[2] != "/"):
            s2 = s2[:2] + "/" + s2[2:]
        while "//" in s2:
            s2 = s2.replace("//", "/")
        return "/" + s2

    # Default: resolve like a normal filesystem path
    try:
        p = Path(s).expanduser().resolve(strict=False)
        out = p.as_posix()
    except Exception:
        out = s.replace("\\", "/")

    if not out.startswith("/"):
        out = "/" + out
    while "//" in out:
        out = out.replace("//", "/")
    return out or "/"


# ----------------------------
# Project metadata inference
# ----------------------------

def _read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _load_toml(path: Path) -> dict:
    # tomllib (py>=3.11) expects str for loads(); tomli is similar.
    try:
        import tomllib  # type: ignore
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            import tomli  # type: ignore
            return tomli.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _find_upwards(start: str, filename: str) -> Path | None:
    if not start:
        return None
    p = Path(start).resolve()
    for d in [p, *p.parents]:
        cand = d / filename
        if cand.exists():
            return cand
    return None


_SETUP_NAME_RE = re.compile(r"""(?s)\bname\s*=\s*["']([^"']+)["']""")
_SETUP_VERSION_RE = re.compile(r"""(?s)\bversion\s*=\s*["']([^"']+)["']""")


def _parse_setup_py(setup_py: Path) -> tuple[str | None, str | None]:
    txt = _read_text(setup_py)
    if not txt:
        return None, None

    name = None
    version = None

    m = _SETUP_NAME_RE.search(txt)
    if m:
        name = m.group(1).strip() or None

    m = _SETUP_VERSION_RE.search(txt)
    if m:
        version = m.group(1).strip() or None

    return name, version


def _parse_pyproject(pyproject: Path) -> tuple[str | None, str | None]:
    data = _load_toml(pyproject)
    if not data:
        return None, None

    name = None
    version = None

    # PEP 621
    proj = data.get("project") if isinstance(data, dict) else None
    if isinstance(proj, dict):
        if isinstance(proj.get("name"), str):
            name = proj.get("name")
        if isinstance(proj.get("version"), str):
            version = proj.get("version")

    # Poetry legacy
    tool = data.get("tool") if isinstance(data, dict) else None
    if isinstance(tool, dict):
        poetry = tool.get("poetry")
        if isinstance(poetry, dict):
            if name is None and isinstance(poetry.get("name"), str):
                name = poetry.get("name")
            if version is None and isinstance(poetry.get("version"), str):
                version = poetry.get("version")

    name = name.strip() if isinstance(name, str) and name.strip() else None
    version = version.strip() if isinstance(version, str) and version.strip() else None
    return name, version


def _infer_project(cwd: str) -> tuple[str | None, str | None]:
    pyproject = _find_upwards(cwd, "pyproject.toml")
    if pyproject is not None:
        n, v = _parse_pyproject(pyproject)
        if n or v:
            return n, v

    setup_py = _find_upwards(cwd, "setup.py")
    if setup_py is not None:
        n, v = _parse_setup_py(setup_py)
        if n or v:
            return n, v

    return None, None


# ----------------------------
# UserInfo
# ----------------------------

@dataclass(frozen=True, slots=True)
class UserInfo:
    email: str | None
    key: str
    hostname: str

    url: URL            # Databricks UI URL if possible, else local://<host>/<abs-cwd>
    git_url: URL | None # Repo URL if inferable, else None

    product: str | None
    product_version: str | None

    @classmethod
    def current(cls, *, refresh: bool = False) -> "UserInfo":
        global _CURRENT_CACHE
        if _CURRENT_CACHE is not None and not refresh:
            return _CURRENT_CACHE

        hostname = socket.gethostname()
        key = _get_key()
        email = _get_upn_email() or _guess_email_from_env()

        cwd = _safe_getcwd()

        project, project_version = _infer_project(cwd) if cwd else (None, None)

        git = _git_info(cwd) if cwd else None
        git_url = _git_url_from_info(git)

        url = _current_compute_url(hostname=hostname, cwd=cwd)

        _CURRENT_CACHE = cls(
            email=email,
            key=key,
            hostname=hostname,
            url=url,
            git_url=git_url,
            product=project,
            product_version=project_version,
        )
        return _CURRENT_CACHE

    def with_email(self, email: str | None) -> "UserInfo":
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


# ----------------------------
# Identity probes
# ----------------------------

def _get_key() -> str:
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        u = _get_dbx_user()
        if u:
            return u

    name = _run_quiet(["whoami"])
    if name:
        return name.strip()

    return os.getenv("USERNAME") or os.getenv("USER") or os.getenv("LOGNAME") or "unknown"


def _get_upn_email() -> str | None:
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        u = _get_dbx_user()
        if u and "@" in u:
            return u

    upn = _run_quiet(["whoami", "/UPN"])
    if not upn:
        return None

    upn = upn.strip()
    if not upn or upn.lower() == "null":
        return None

    return upn if "@" in upn else None


def _guess_email_from_env() -> str | None:
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL", "EMAIL"):
        v = os.getenv(k)
        if v and "@" in v:
            return v.strip()
    return None


def _get_dbx_user() -> str | None:
    try:
        from yggdrasil.databricks import Workspace

        ws = Workspace(product="current", product_version="0.0.0")
        u = ws.current_user.user_name
        return u.strip() if u else None
    except Exception:
        return None


def _run_quiet(cmd: Sequence[str]) -> str | None:
    try:
        out = subprocess.check_output(list(cmd), text=True, stderr=subprocess.DEVNULL)
        out = out.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


# ----------------------------
# Databricks URL construction
# ----------------------------

def _ctx_tags() -> Mapping[str, str]:
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # type: ignore[name-defined]
        j = json.loads(ctx.toJson())
        tags = j.get("tags") or {}
        return {str(k): str(v) for k, v in tags.items()}
    except Exception:
        return {}


def _pick(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v:
            s = str(v).strip()
            if s:
                return s
    return None


def _databricks_current_url(*, kind: DatabricksLinkKind = "auto") -> URL | None:
    if not os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return None

    tags = _ctx_tags()

    host = _pick(
        tags.get("browserHostName"),
        os.getenv("DATABRICKS_HOST", "").removeprefix("https://").removeprefix("http://"),
        os.getenv("DATABRICKS_WORKSPACE_URL", "").removeprefix("https://").removeprefix("http://"),
    )
    org_id = _pick(
        tags.get("orgId"),
        os.getenv("DATABRICKS_ORG_ID"),
        os.getenv("DATABRICKS_WORKSPACE_ID"),
    )
    if not host or not org_id:
        return None

    base = URL.parse_dict({"scheme": "https", "host": host, "path": "/", "query": f"o={org_id}"})

    job_id = _pick(tags.get("jobId"), tags.get("job_id"), os.getenv("DATABRICKS_JOB_ID"))
    run_id = _pick(
        tags.get("jobRunId"),
        tags.get("job_run_id"),
        tags.get("runId"),
        os.getenv("DATABRICKS_RUN_ID"),
    )
    notebook_id = _pick(tags.get("notebookId"), tags.get("notebook_id"))
    notebook_path = _pick(tags.get("notebookPath"))

    if kind in ("auto", "job_run"):
        if job_id and run_id:
            return base.with_fragment(f"job/{job_id}/run/{run_id}")
        if kind == "job_run":
            return None

    if kind in ("auto", "notebook_id"):
        if notebook_id:
            return base.with_fragment(f"notebook/{notebook_id}")
        if kind == "notebook_id":
            return None

    if kind in ("auto", "workspace_path"):
        if notebook_path:
            return base.with_fragment(f"workspace{notebook_path}")
        if kind == "workspace_path":
            return None

    return None


# ----------------------------
# Local compute URL fallback
# ----------------------------

def _local_compute_url(*, hostname: str, cwd: str) -> URL:
    path = normalize_abs_path_for_url(cwd)
    return URL.parse_dict({"scheme": "local", "host": hostname, "path": path})


def _current_compute_url(*, hostname: str, cwd: str) -> URL:
    u = _databricks_current_url(kind="auto")
    if u is not None:
        return u
    return _local_compute_url(hostname=hostname, cwd=cwd)


# ----------------------------
# Git inference (no subprocess)
# ----------------------------

def _resolve_gitdir(dotgit: Path) -> Path | None:
    if dotgit.is_dir():
        return dotgit
    if dotgit.is_file():
        txt = _read_text(dotgit)
        if not txt:
            return None
        line = txt.strip().splitlines()[0].strip()
        if line.lower().startswith("gitdir:"):
            rel = line.split(":", 1)[1].strip()
            gd = (dotgit.parent / rel).resolve()
            return gd if gd.exists() else None
    return None


def _find_git_root(start: str) -> tuple[Path, Path] | None:
    if not start:
        return None
    p = Path(start).resolve()
    for d in [p, *p.parents]:
        dotgit = d / ".git"
        if dotgit.exists():
            gd = _resolve_gitdir(dotgit)
            if gd is not None:
                return d, gd
    return None


def _read_packed_ref(gitdir: Path, ref: str) -> str | None:
    txt = _read_text(gitdir / "packed-refs")
    if not txt:
        return None
    for line in txt.splitlines():
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.strip().split(" ")
        if len(parts) == 2 and parts[1] == ref:
            sha = parts[0].strip()
            return sha if len(sha) >= 8 else None
    return None


def _git_head_info(repo_root: Path, gitdir: Path) -> dict[str, str] | None:
    head_txt = _read_text(gitdir / "HEAD")
    if not head_txt:
        return None
    head_txt = head_txt.strip()

    branch: str | None = None
    sha: str | None = None

    if head_txt.startswith("ref:"):
        ref = head_txt.split(":", 1)[1].strip()
        if ref.startswith("refs/heads/"):
            branch = ref[len("refs/heads/") :]

        ref_txt = _read_text(gitdir / ref)
        if ref_txt:
            sha = ref_txt.strip()
        else:
            sha = _read_packed_ref(gitdir, ref)
    else:
        sha = head_txt  # detached HEAD

    if sha:
        sha = sha.strip()
        sha = sha[:12] if len(sha) >= 8 else None

    out = {"git_root": str(repo_root)}
    if branch:
        out["git_branch"] = branch
    if sha:
        out["git_sha"] = sha
    return out


def _git_remote_origin(gitdir: Path) -> str | None:
    cfg = _read_text(gitdir / "config")
    if not cfg:
        return None

    current_remote: str | None = None
    remotes: dict[str, str] = {}

    for raw in cfg.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue

        if line.startswith("[") and line.endswith("]"):
            inside = line[1:-1].strip()
            if inside.lower().startswith("remote "):
                q1 = inside.find('"')
                q2 = inside.rfind('"')
                current_remote = inside[q1 + 1 : q2] if q1 != -1 and q2 != -1 and q2 > q1 else None
            else:
                current_remote = None
            continue

        if current_remote and "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            if k.lower() == "url":
                remotes[current_remote] = v

    if "origin" in remotes:
        return remotes["origin"]
    return next(iter(remotes.values()), None)


def _git_info(cwd: str) -> dict[str, str] | None:
    found = _find_git_root(cwd)
    if not found:
        return None

    repo_root, gitdir = found
    head = _git_head_info(repo_root, gitdir) or {}
    remote = _git_remote_origin(gitdir)
    if remote:
        head["git_remote"] = remote

    return head if head.get("git_remote") else None


def _normalize_git_remote(remote: str) -> str:
    r = remote.strip()

    m = re.match(r"^git@([^:]+):(.+)$", r)
    if m:
        host, path = m.group(1), m.group(2)
        path = path[:-4] if path.endswith(".git") else path
        return f"https://{host}/{path}"

    m = re.match(r"^ssh://git@([^/]+)/(.+)$", r)
    if m:
        host, path = m.group(1), m.group(2)
        path = path[:-4] if path.endswith(".git") else path
        return f"https://{host}/{path}"

    if r.startswith("http://") or r.startswith("https://"):
        return r[:-4] if r.endswith(".git") else r

    return r


def _git_url_from_info(git: dict[str, str] | None) -> URL | None:
    if not git:
        return None

    remote = git.get("git_remote")
    if not remote:
        return None

    base = _normalize_git_remote(remote)
    u = URL.parse_str(base, normalize=True)

    ref = git.get("git_sha") or git.get("git_branch")
    if ref:
        u = u.with_fragment(ref)

    return u