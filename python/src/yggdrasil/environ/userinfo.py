from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence

from yggdrasil.io.url import URL

__all__ = [
    "UserInfo",
    "get_user_info",
    "normalize_abs_path_for_url",
    "parse_name_from_email",
]

# ── types ─────────────────────────────────────────────────────────────────────

DatabricksLinkKind = Literal["auto", "job_run", "notebook_id", "workspace_path"]

# ── module-level cache ────────────────────────────────────────────────────────

_CURRENT_CACHE: UserInfo | None = None

# ── constants ─────────────────────────────────────────────────────────────────

_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:(?:[\\/]|$)")
_UNC_RE = re.compile(r"^(\\\\|//)([^\\/]+)[\\/]+([^\\/]+)(.*)$")
_SETUP_NAME_RE = re.compile(r"""(?s)\bname\s*=\s*["']([^"']+)["']""")
_SETUP_VERSION_RE = re.compile(r"""(?s)\bversion\s*=\s*["']([^"']+)["']""")
_SPLIT_RE = re.compile(r"[._+]+")

_LASTNAME_PARTICLES = {
    "da", "de", "del", "della", "der", "di", "du", "des",
    "la", "le", "les", "van", "von", "den", "ten", "ter",
    "st", "st.", "saint", "y",
}


# ── class ─────────────────────────────────────────────────────────────────────


class UserInfo:
    __slots__ = (
        "_key",
        "_cwd",
        "_hostname",
        "_email",
        "_email_loaded",
        "_url",
        "_url_loaded",
        "_git_info_cache",
        "_git_info_loaded",
        "_git_url",
        "_git_url_loaded",
        "_project_cache",
        "_project_loaded",
        "_name_parts_cache",
        "_name_parts_loaded",
    )

    def __init__(self, *, key: str, cwd: str) -> None:
        self._key = key
        self._cwd = cwd

        self._hostname: str | None = None

        self._email: str | None = None
        self._email_loaded = False

        self._url: URL | None = None
        self._url_loaded = False

        self._git_info_cache: dict[str, str] | None = None
        self._git_info_loaded = False

        self._git_url: URL | None = None
        self._git_url_loaded = False

        self._project_cache: tuple[str | None, str | None] | None = None
        self._project_loaded = False

        self._name_parts_cache: tuple[str | None, str | None] | None = None
        self._name_parts_loaded = False

    @classmethod
    def current(cls) -> "UserInfo":
        return get_user_info()

    def with_email(self, email: str | None) -> "UserInfo":
        clone = UserInfo(key=self.key, cwd=self.cwd)
        clone._hostname = self._hostname

        clone._email = email
        clone._email_loaded = True

        clone._url = self._url
        clone._url_loaded = self._url_loaded

        clone._git_info_cache = self._git_info_cache
        clone._git_info_loaded = self._git_info_loaded

        clone._git_url = self._git_url
        clone._git_url_loaded = self._git_url_loaded

        clone._project_cache = self._project_cache
        clone._project_loaded = self._project_loaded

        clone._name_parts_cache = None
        clone._name_parts_loaded = False
        return clone

    def __repr__(self) -> str:
        return (
            f"UserInfo("
            f"email={self.email!r}, "
            f"first_name={self.first_name!r}, "
            f"last_name={self.last_name!r}, "
            f"key={self.key!r}, "
            f"hostname={self.hostname!r}, "
            f"url={self.url!r}, "
            f"git_url={self.git_url!r}, "
            f"product={self.product!r}, "
            f"product_version={self.product_version!r}"
            f")"
        )

    @property
    def key(self) -> str:
        return self._key

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def hostname(self) -> str:
        value = self._hostname
        if value is None:
            value = socket.gethostname()
            self._hostname = value
        return value

    @property
    def email(self) -> str | None:
        if not self._email_loaded:
            self._email = _get_upn_email() or _guess_email_from_env()
            self._email_loaded = True
        return self._email

    @property
    def first_name(self) -> str | None:
        return self._name_parts[0]

    @property
    def last_name(self) -> str | None:
        return self._name_parts[1]

    @property
    def url(self) -> URL | None:
        if not self._url_loaded:
            self._url = _current_compute_url(hostname=self.hostname, cwd=self.cwd)
            self._url_loaded = True
        return self._url

    @property
    def git_url(self) -> URL | None:
        if not self._git_url_loaded:
            self._git_url = _git_url_from_info(self._git_info)
            self._git_url_loaded = True
        return self._git_url

    @property
    def product(self) -> str | None:
        return self._project_info[0]

    @property
    def product_version(self) -> str | None:
        return self._project_info[1]

    @property
    def _git_info(self) -> dict[str, str] | None:
        if not self._git_info_loaded:
            self._git_info_cache = _git_info(self.cwd)
            self._git_info_loaded = True
        return self._git_info_cache

    @property
    def _project_info(self) -> tuple[str | None, str | None]:
        if not self._project_loaded:
            self._project_cache = _infer_project(self.cwd)
            self._project_loaded = True
        return self._project_cache or (None, None)

    @property
    def _name_parts(self) -> tuple[str | None, str | None]:
        if not self._name_parts_loaded:
            first_name: str | None = None
            last_name: str | None = None

            parsed = parse_name_from_email(self.email or "")
            if parsed is not None:
                first_name, last_name, _domain = parsed

            self._name_parts_cache = (first_name, last_name)
            self._name_parts_loaded = True

        return self._name_parts_cache or (None, None)


# ── public API ────────────────────────────────────────────────────────────────


def get_user_info(*, refresh: bool = False) -> UserInfo:
    """Return cached UserInfo for the current execution context."""
    global _CURRENT_CACHE

    if _CURRENT_CACHE is not None and not refresh:
        return _CURRENT_CACHE

    info = UserInfo(
        key=_get_key(),
        cwd=_safe_getcwd(),
    )
    _CURRENT_CACHE = info
    return info


def _clear_cache() -> None:
    global _CURRENT_CACHE
    _CURRENT_CACHE = None


# ── URL helpers ───────────────────────────────────────────────────────────────


def _current_compute_url(*, hostname: str, cwd: str) -> URL | None:
    """Databricks URL when in DBR, otherwise a ``local://`` URL for the cwd."""
    dbx = _databricks_current_url(kind="auto")
    if dbx is not None:
        return dbx
    if not cwd:
        return None
    path = normalize_abs_path_for_url(cwd)
    return URL.parse_dict({"scheme": "local", "host": hostname.lower(), "path": path})


def _databricks_current_url(*, kind: DatabricksLinkKind = "auto") -> URL | None:
    if not os.getenv("DATABRICKS_RUNTIME_VERSION"):
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

    base = URL.parse_dict({"scheme": "https", "host": host, "path": "/", "query": f"o={org_id}"})

    job_id = _pick(tags.get("jobId"), tags.get("job_id"), os.getenv("DATABRICKS_JOB_ID"))
    run_id = _pick(tags.get("jobRunId"), tags.get("job_run_id"), tags.get("runId"), os.getenv("DATABRICKS_RUN_ID"))
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


def _ctx_tags() -> Mapping[str, str]:
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # type: ignore[name-defined]
        payload = json.loads(ctx.toJson())
        tags = payload.get("tags") or {}
        return {str(k): str(v) for k, v in tags.items()}
    except Exception:
        return {}


# ── path normalization ────────────────────────────────────────────────────────


def normalize_abs_path_for_url(path: str) -> str:
    """Normalize an absolute filesystem path into a URL-safe POSIX string.

    Examples::

        /a//b///c              -> /a/b/c
        C:\\Users\\me\\proj    -> /C:/Users/me/proj
        \\\\server\\share\\dir -> //server/share/dir
    """
    if not path:
        return "/"
    s = os.path.expanduser(str(path).strip())
    if not s:
        return "/"

    if s.startswith("/") and not _WIN_DRIVE_RE.match(s) and not _UNC_RE.match(s):
        return _collapse_slashes(s.replace("\\", "/")) or "/"

    unc = _UNC_RE.match(s)
    if unc:
        server, share, rest = unc.group(2), unc.group(3), unc.group(4) or ""
        rest = rest.replace("\\", "/")
        if rest and not rest.startswith("/"):
            rest = f"/{rest}"
        return f"//{server}/{share}{_collapse_slashes(rest)}"

    if _WIN_DRIVE_RE.match(s):
        out = s.replace("\\", "/")
        if len(out) >= 2 and out[1] == ":" and (len(out) == 2 or out[2] != "/"):
            out = f"{out[:2]}/{out[2:]}"
        return f"/{_collapse_slashes(out)}"

    try:
        resolved = Path(s).expanduser().resolve(strict=False).as_posix()
    except Exception:
        resolved = s.replace("\\", "/")
    if not resolved.startswith("/"):
        resolved = f"/{resolved}"
    return _collapse_slashes(resolved) or "/"


def _collapse_slashes(path: str) -> str:
    while "//" in path:
        path = path.replace("//", "/")
    return path


# ── git helpers ───────────────────────────────────────────────────────────────


def _git_info(cwd: str) -> dict[str, str] | None:
    """Walk up from *cwd* to find a git repo; return branch/sha/remote dict."""
    if not cwd:
        return None

    root = Path(cwd).resolve()
    for directory in (root, *root.parents):
        dotgit = directory / ".git"
        if not dotgit.exists():
            continue
        gitdir = _resolve_gitdir(dotgit)
        if gitdir is None:
            continue

        info: dict[str, str] = {"git_root": str(directory)}

        head_text = _read_text(gitdir / "HEAD")
        if head_text:
            head_text = head_text.strip()
            if head_text.startswith("ref:"):
                ref = head_text.split(":", 1)[1].strip()
                if ref.startswith("refs/heads/"):
                    info["git_branch"] = ref.removeprefix("refs/heads/")
                    sha = _resolve_ref_sha(gitdir, ref)
                    if sha:
                        info["git_sha"] = sha[:12]

        remote = _git_remote_origin(gitdir)
        if remote:
            info["git_remote"] = remote

        return info if info.get("git_remote") else None

    return None


def _resolve_ref_sha(gitdir: Path, ref: str) -> str | None:
    """Read a loose ref file, then fall back to packed-refs."""
    sha = _read_text(gitdir / ref)
    if sha and len(sha.strip()) >= 12:
        return sha.strip()
    text = _read_text(gitdir / "packed-refs")
    if text:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "^")):
                continue
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref and len(parts[0]) >= 8:
                return parts[0]
    return None


def _resolve_gitdir(dotgit: Path) -> Path | None:
    """Resolve .git (directory or ``gitdir:`` file) to the actual git dir."""
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
    relative = first_line.split(":", 1)[1].strip()
    resolved = (dotgit.parent / relative).resolve()
    return resolved if resolved.exists() else None


def _git_remote_origin(gitdir: Path) -> str | None:
    """Return the ``origin`` remote URL (or first remote) from git config."""
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
                start, end = section.find('"'), section.rfind('"')
                current_remote = section[start + 1 : end] if start != -1 and end > start else None
            else:
                current_remote = None
            continue
        if current_remote and "=" in line:
            key, value = (p.strip() for p in line.split("=", 1))
            if key.lower() == "url":
                remotes[current_remote] = value

    return remotes.get("origin") or next(iter(remotes.values()), None)


def _git_url_from_info(git: dict[str, str] | None) -> URL | None:
    """Build a canonical HTTPS URL from a git info dict, using SHA as fragment."""
    if not git:
        return None
    remote = git.get("git_remote")
    if not remote:
        return None
    url = URL.parse_str(_normalize_git_remote(remote), normalize=True)
    sha = git.get("git_sha")
    if sha:
        url = url.with_fragment(sha)
    return url


def _normalize_git_remote(remote: str) -> str:
    remote = remote.strip()
    scp = re.match(r"^git@([^:]+):(.+)$", remote)
    if scp:
        return f"https://{scp.group(1)}/{scp.group(2).removesuffix('.git')}"
    ssh = re.match(r"^ssh://git@([^/]+)/(.+)$", remote)
    if ssh:
        return f"https://{ssh.group(1)}/{ssh.group(2).removesuffix('.git')}"
    if remote.startswith(("http://", "https://")):
        return remote.removesuffix(".git")
    return remote


# ── project inference ─────────────────────────────────────────────────────────


def _infer_project(cwd: str) -> tuple[str | None, str | None]:
    """Walk up from *cwd* looking for pyproject.toml or setup.py."""
    if not cwd:
        return None, None
    root = Path(cwd).resolve()
    for directory in (root, *root.parents):
        pyproject = directory / "pyproject.toml"
        if pyproject.exists():
            name, version = _parse_pyproject(pyproject)
            if name or version:
                return name, version
        setup_py = directory / "setup.py"
        if setup_py.exists():
            name, version = _parse_setup_py(setup_py)
            if name or version:
                return name, version
    return None, None


def _parse_pyproject(path: Path) -> tuple[str | None, str | None]:
    data = _load_toml(path)
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


def _parse_setup_py(path: Path) -> tuple[str | None, str | None]:
    text = _read_text(path)
    if not text:
        return None, None
    name_m = _SETUP_NAME_RE.search(text)
    ver_m = _SETUP_VERSION_RE.search(text)
    return (
        _clean_str(name_m.group(1) if name_m else None),
        _clean_str(ver_m.group(1) if ver_m else None),
    )


# ── system / env helpers ──────────────────────────────────────────────────────


def _safe_getcwd() -> str:
    try:
        return os.getcwd()
    except Exception:
        return ""


def _get_key() -> str:
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        dbx_user = _get_dbx_user()
        if dbx_user:
            return dbx_user
    whoami = _run_quiet(["whoami"])
    if whoami:
        return whoami.strip()
    return os.getenv("USERNAME") or os.getenv("USER") or os.getenv("LOGNAME") or "unknown"


def _get_upn_email() -> str | None:
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        dbx_user = _get_dbx_user()
        if dbx_user and "@" in dbx_user:
            return dbx_user
    upn = _clean_str(_run_quiet(["whoami", "/UPN"]))
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
        output = subprocess.check_output(list(cmd), text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return _clean_str(output)


# ── TOML / file utils ─────────────────────────────────────────────────────────


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


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _clean_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _pick(*values: Optional[str]) -> str | None:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _strip_scheme(value: str) -> str:
    return value.removeprefix("https://").removeprefix("http://")


# ── name parsing ──────────────────────────────────────────────────────────────


def parse_name_from_email(email: str) -> tuple[str, str, str] | None:
    value = (email or "").strip()
    if "@" not in value:
        return None

    local, domain = value.rsplit("@", 1)
    local = local.strip().split("+", 1)[0]
    domain = domain.strip().lower()
    if not local or not domain:
        return None

    tokens = [re.sub(r"\d+$", "", t) for t in _SPLIT_RE.split(local) if t]
    tokens = [t for t in tokens if t]
    if len(tokens) < 2:
        return None

    return _smart_title(tokens[0]), _build_last_name(tokens), domain


def _smart_title(token: str) -> str:
    token = token.strip().replace("\u2019", "'")
    return "-".join(_format_name_chunk(chunk) for chunk in token.split("-")) if token else token


def _format_name_chunk(chunk: str) -> str:
    if "'" not in chunk:
        return chunk[:1].upper() + chunk[1:].lower()
    left, right = chunk.split("'", 1)
    left_lower = left.lower()
    left_fmt = left_lower if left_lower in _LASTNAME_PARTICLES else left[:1].upper() + left[1:].lower()
    right_fmt = right[:1].upper() + right[1:].lower() if right else ""
    return f"{left_fmt}'{right_fmt}" if right_fmt else f"{left_fmt}'"


def _build_last_name(tokens: list[str]) -> str:
    parts = [tokens[-1]]
    index = len(tokens) - 2
    while index >= 1:
        candidate = tokens[index].replace("\u2019", "'").lower().strip(".")
        if candidate in _LASTNAME_PARTICLES:
            parts.insert(0, tokens[index])
            index -= 1
            continue
        if len(tokens) >= 4 and len(parts) == 1:
            parts.insert(0, tokens[index])
            index -= 1
            continue
        break
    last = " ".join(_smart_title(p) for p in parts)
    return " ".join(
        word.lower() if word.replace("\u2019", "'").lower().strip(".") in _LASTNAME_PARTICLES else word
        for word in last.split()
    )