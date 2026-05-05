from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

import pyarrow as pa

from yggdrasil.io.url import URL, URL_STRUCT

__all__ = [
    "UserInfo",
    "USERINFO_SCHEMA",
    "USERINFO_STRUCT",
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


# ── arrow schema ──────────────────────────────────────────────────────────────


# Single source of truth for the UserInfo struct shape used by request /
# response serializers. Defined as raw pyarrow (rather than wrapped in
# :class:`yggdrasil.data.schema.Schema`) because ``yggdrasil.data`` is
# imported from inside ``yggdrasil.environ``'s init path, so reaching
# back into ``data.data_field`` here would create an import cycle.
USERINFO_SCHEMA: pa.Schema = pa.schema([
    pa.field("hash",            pa.int64(),  nullable=False),
    pa.field("key",             pa.string(), nullable=False),
    pa.field("cwd",             pa.string(), nullable=False),
    pa.field("hostname",        pa.string(), nullable=False),
    pa.field("email",           pa.string(), nullable=True),
    pa.field("first_name",      pa.string(), nullable=True),
    pa.field("last_name",       pa.string(), nullable=True),
    pa.field("url",             URL_STRUCT,  nullable=True),
    pa.field("git_url",         URL_STRUCT,  nullable=True),
    pa.field("product",         pa.string(), nullable=True),
    pa.field("product_version", pa.string(), nullable=True),
])

USERINFO_STRUCT: pa.StructType = pa.struct(list(USERINFO_SCHEMA))


# ── class ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class UserInfo:
    key: str = ""
    cwd: str = ""
    hostname: str = ""
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    url: URL | None = None
    git_url: URL | None = None
    product: str | None = None
    product_version: str | None = None

    @classmethod
    def current(cls) -> "UserInfo":
        return get_user_info()

    @property
    def hash(self) -> int:
        """xxh3_64 digest over (key, hostname, email, url, git_url) — int64."""
        return _userinfo_hash(self)

    def with_email(self, email: str | None) -> "UserInfo":
        first_name, last_name = _names_from_email(email)
        return replace(
            self,
            email=email,
            first_name=first_name,
            last_name=last_name,
        )

    def to_struct_dict(self) -> dict[str, Any]:
        """Flatten into the dict shape that matches :data:`USERINFO_STRUCT`."""
        return {
            "hash":            self.hash,
            "key":             self.key,
            "cwd":             self.cwd,
            "hostname":        self.hostname,
            "email":           self.email,
            "first_name":      self.first_name,
            "last_name":       self.last_name,
            "url":             _url_to_struct(self.url),
            "git_url":         _url_to_struct(self.git_url),
            "product":         self.product,
            "product_version": self.product_version,
        }

    @classmethod
    def from_struct_dict(cls, value: Mapping[str, Any]) -> "UserInfo":
        """Inverse of :meth:`to_struct_dict` — drops the derived ``hash``."""
        return cls(
            key=str(value.get("key") or ""),
            cwd=str(value.get("cwd") or ""),
            hostname=str(value.get("hostname") or ""),
            email=_or_none(value.get("email")),
            first_name=_or_none(value.get("first_name")),
            last_name=_or_none(value.get("last_name")),
            url=_url_from_struct(value.get("url")),
            git_url=_url_from_struct(value.get("git_url")),
            product=_or_none(value.get("product")),
            product_version=_or_none(value.get("product_version")),
        )


# ── public API ────────────────────────────────────────────────────────────────


def get_user_info(*, refresh: bool = False) -> UserInfo:
    """Return cached UserInfo for the current execution context."""
    global _CURRENT_CACHE

    if _CURRENT_CACHE is not None and not refresh:
        return _CURRENT_CACHE

    cwd = _safe_getcwd()
    hostname = socket.gethostname()
    email = _get_upn_email() or _guess_email_from_env()
    first_name, last_name = _names_from_email(email)
    url = _current_compute_url(hostname=hostname, cwd=cwd)
    git_data = _git_info(cwd)
    git_url = _git_url_from_info(git_data)
    product, product_version = _infer_project(cwd)

    info = UserInfo(
        key=_get_key(),
        cwd=cwd,
        hostname=hostname,
        email=email,
        first_name=first_name,
        last_name=last_name,
        url=url,
        git_url=git_url,
        product=product,
        product_version=product_version,
    )
    _CURRENT_CACHE = info
    return info


def _clear_cache() -> None:
    global _CURRENT_CACHE
    _CURRENT_CACHE = None


# ── struct / hash helpers ─────────────────────────────────────────────────────


def _names_from_email(email: str | None) -> tuple[str | None, str | None]:
    parsed = parse_name_from_email(email or "")
    if parsed is None:
        return None, None
    first, last, _ = parsed
    return first, last


def _url_to_struct(url: URL | None) -> dict[str, Any] | None:
    if url is None:
        return None
    return {
        "scheme":   url.scheme or "",
        "userinfo": url.userinfo,
        "host":     url.host or "",
        "port":     url.port if url.port not in (None, 0) else None,
        "path":     url.path or "",
        "query":    url.query,
        "fragment": url.fragment,
    }


def _url_from_struct(value: Any) -> URL | None:
    if value is None:
        return None
    if isinstance(value, URL):
        return value
    if isinstance(value, str):
        return URL.from_str(value) if value else None
    if isinstance(value, Mapping):
        if not any(value.get(k) for k in ("scheme", "host", "path")):
            return None
        return URL.from_dict(dict(value))
    return None


def _or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _userinfo_hash(info: "UserInfo") -> int:
    """Stable signed-int64 xxh3_64 over identity-defining fields."""
    try:
        import xxhash
    except ImportError:
        return 0

    parts = [
        info.key or "",
        info.hostname or "",
        info.email or "",
        info.url.to_string() if info.url is not None else "",
        info.git_url.to_string() if info.git_url is not None else "",
    ]
    payload = "\x00".join(parts).encode("utf-8")
    u = xxhash.xxh3_64(payload).intdigest()
    return u if u < 2**63 else u - 2**64


# ── URL helpers ───────────────────────────────────────────────────────────────


def _current_compute_url(*, hostname: str, cwd: str) -> URL | None:
    """Databricks URL when in DBR, otherwise a ``local://`` URL for the cwd."""
    dbx = _databricks_current_url(kind="auto")
    if dbx is not None:
        return dbx
    if not cwd:
        return None
    path = normalize_abs_path_for_url(cwd)
    return URL.from_dict({"scheme": "local", "host": hostname.lower(), "path": path})


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

    base = URL.from_dict({"scheme": "https", "host": host, "path": "/", "query": f"o={org_id}"})

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
    url = URL.from_str(_normalize_git_remote(remote), normalize=True)
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