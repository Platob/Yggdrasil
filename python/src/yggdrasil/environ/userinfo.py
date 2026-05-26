from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping, Optional, Sequence

import pyarrow as pa

from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.url import URL

# Lazy field caches default to ``...`` (the ``Ellipsis`` singleton)
# to mean "not yet resolved". ``None`` is a valid computed result
# (no email, no detectable compute / repo, no project) and can't be
# overloaded for that role; ``Ellipsis`` is a stable, importable
# singleton so identity checks (``is ...``) are safe across pickling
# and across worker boundaries without a private sentinel object.

__all__ = [
    "UserInfo",
    "USERINFO_STRUCT",
]

# ── types ─────────────────────────────────────────────────────────────────────

DatabricksLinkKind = Literal["auto", "job_run", "notebook_id", "workspace_path"]

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
    pa.field("hostname",        pa.string(), nullable=False),
    pa.field("email",           pa.string(), nullable=True),
    pa.field("first_name",      pa.string(), nullable=True),
    pa.field("last_name",       pa.string(), nullable=True),
    pa.field("product",         pa.string(), nullable=True),
    pa.field("product_version", pa.string(), nullable=True),
])

USERINFO_STRUCT: pa.StructType = pa.struct(list(USERINFO_SCHEMA))


# ── class ─────────────────────────────────────────────────────────────────────


class UserInfo(Singleton):
    """Snapshot of the user identity for the current process.

    ``hostname`` is populated eagerly (``socket.gethostname()`` is
    cheap and answers reliably without I/O fan-out). Every other
    identity field is a lazy, memoized property:

    - ``key`` / ``email`` consult the Databricks notebook context
      (free, in-process) and AWS-managed env vars before shelling
      out to ``whoami`` / ``whoami /UPN`` or calling the Databricks
      IAM SDK.
    - ``first_name`` / ``last_name`` are derived from ``email`` —
      free once email has been resolved, but pulling them eagerly
      forces email resolution.
    - ``product`` / ``product_version`` walk parent directories
      looking for ``pyproject.toml`` / ``setup.py`` and parse TOML.
    - ``cwd`` / ``url`` / ``git_url`` probe the runtime, dbutils
      context, and the on-disk ``.git`` tree.

    All of those firing at construction would tax callers that only
    care about a subset (typical of request/response sanitization on
    the hot path). Each cache fires at most once per instance, and
    once a value is resolved — or supplied via :meth:`from_struct_dict`
    or constructor kwargs — it persists for the life of the instance.

    :class:`Singleton` integration:

    - The constructor caches one instance per ``(cls, hostname, key,
      email, first/last name, product, product version)``. Two
      ``UserInfo.current()`` calls collapse to the same instance;
      :meth:`with_email` / :meth:`from_struct_dict` produce
      different identities and live as separate instances under the
      same hostname.
    - Per-process derived caches (``_cwd_cache`` / ``_url_cache`` /
      ``_git_url_cache``) are listed in
      :attr:`_TRANSIENT_STATE_ATTRS` so cross-process pickle drops
      them; the receiver re-derives from its own context.

    Construction:

    - :meth:`current` returns the singleton for the local hostname.
    - :meth:`from_struct_dict` rebuilds an instance from a wire
      payload, with every supplied field pre-populated so no
      resolution fires on the receiver.
    - The constructor accepts the underscore-prefixed cache fields
      directly — useful for tests and for :meth:`with_email`.
    """

    _SINGLETON_TTL: ClassVar[Any] = None
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "_cwd_cache", "_url_cache", "_git_url_cache",
    })

    @classmethod
    def _singleton_key(
        cls,
        hostname: str = "",
        *,
        _key: Any = ...,
        _email: Any = ...,
        _first_name: Any = ...,
        _last_name: Any = ...,
        _product: Any = ...,
        _product_version: Any = ...,
    ) -> Any:
        # Identity key includes every wire field so two different
        # wire payloads on the same host don't collapse into each
        # other, and ``with_email`` lives as a separate instance.
        # Per-process slots (cwd/url/git_url) are excluded — they're
        # local to whoever holds the instance.
        return (
            cls, hostname,
            _key, _email,
            _first_name, _last_name,
            _product, _product_version,
        )

    def __init__(
        self,
        hostname: str = "",
        *,
        _key: Any = ...,
        _email: Any = ...,
        _first_name: Any = ...,
        _last_name: Any = ...,
        _product: Any = ...,
        _product_version: Any = ...,
        singleton_ttl: Any = ...,
    ) -> None:
        # ``Singleton.__new__`` may return a cached instance, in
        # which case ``__init__`` runs a second time — guard so we
        # don't clobber already-resolved lazy slots.
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

        self.hostname = hostname
        self._key = _key
        self._email = _email
        self._first_name = _first_name
        self._last_name = _last_name
        self._product = _product
        self._product_version = _product_version

        # Per-process derived caches. They're in
        # ``_TRANSIENT_STATE_ATTRS`` so cross-process pickle drops
        # them; the receiver re-derives from its own filesystem /
        # runtime / dbutils context.
        self._cwd_cache: Any = ...
        self._url_cache: Any = ...
        self._git_url_cache: Any = ...

        self._initialized = True

    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Route unpickling through ``__new__`` with identity kwargs.

        Pickle reconstructs the instance via
        ``cls.__new__(cls, hostname, **wire_kwargs)``, so the
        singleton machinery collapses the cross-process restore onto
        the live in-process instance whose key matches.
        """
        return (self.hostname,), {
            "_key": self._key,
            "_email": self._email,
            "_first_name": self._first_name,
            "_last_name": self._last_name,
            "_product": self._product,
            "_product_version": self._product_version,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # ``Singleton.__setstate__`` resets transient slots to
        # ``None``; our property accessors use ``...`` as the
        # "not yet resolved" sentinel (``None`` is a legal resolved
        # value for ``url`` / ``git_url``), so re-seed with ``...``
        # so the receiver re-derives ``cwd`` / ``url`` / ``git_url``
        # from its own filesystem / runtime instead of caching the
        # sender's ``None``.
        if getattr(self, "_initialized", False):
            return
        self.__dict__.update(state)
        for attr in self._TRANSIENT_STATE_ATTRS:
            self.__dict__[attr] = ...

    def __repr__(self) -> str:
        return (
            f"UserInfo(hostname={self.hostname!r}, "
            f"key={self._key!r}, email={self._email!r})"
        )

    @classmethod
    def current(cls) -> "UserInfo":
        return get_user_info()

    @property
    def hash(self) -> int:
        """xxh3_64 digest over (key, hostname, email, url, git_url) — int64."""
        return _userinfo_hash(self)

    @property
    def key(self) -> str:
        cached = self._key
        if cached is not ...:
            return cached
        result = _resolve_key()
        object.__setattr__(self, "_key", result)
        return result

    @property
    def email(self) -> str | None:
        cached = self._email
        if cached is not ...:
            return cached
        result = _resolve_email()
        object.__setattr__(self, "_email", result)
        return result

    @property
    def first_name(self) -> str | None:
        cached = self._first_name
        if cached is not ...:
            return cached
        # Names derive from email — resolving one resolves the pair,
        # so the second access skips the regex on the cache hit.
        first, last = _names_from_email(self.email)
        object.__setattr__(self, "_first_name", first)
        object.__setattr__(self, "_last_name", last)
        return first

    @property
    def last_name(self) -> str | None:
        cached = self._last_name
        if cached is not ...:
            return cached
        first, last = _names_from_email(self.email)
        object.__setattr__(self, "_first_name", first)
        object.__setattr__(self, "_last_name", last)
        return last

    @property
    def product(self) -> str | None:
        cached = self._product
        if cached is not ...:
            return cached
        # Like the name pair, project name and version are resolved
        # from a single ``pyproject.toml`` parse — fill both slots.
        product, version = _infer_project(self.cwd)
        object.__setattr__(self, "_product", product)
        object.__setattr__(self, "_product_version", version)
        return product

    @property
    def product_version(self) -> str | None:
        cached = self._product_version
        if cached is not ...:
            return cached
        product, version = _infer_project(self.cwd)
        object.__setattr__(self, "_product", product)
        object.__setattr__(self, "_product_version", version)
        return version

    @property
    def cwd(self) -> str:
        """Resolve the current working directory lazily (memoized)."""
        cached = self._cwd_cache
        if cached is not ...:
            return cached
        result = _safe_getcwd()
        object.__setattr__(self, "_cwd_cache", result)
        return result

    @property
    def url(self) -> URL | None:
        """Compute URL (Databricks link or ``local://`` for the cwd) lazily.

        Result is memoized per instance — the resolution probes the
        Databricks runtime / dbutils context and does a path
        normalization, neither of which should fire repeatedly on the
        request hot path.
        """
        cached = self._url_cache
        if cached is not ...:
            return cached
        result = _current_compute_url(hostname=self.hostname, cwd=self.cwd)
        object.__setattr__(self, "_url_cache", result)
        return result

    @property
    def git_url(self) -> URL | None:
        """Compute the git remote URL lazily (walks parents of ``cwd``).

        Memoized — the walk hits the filesystem (HEAD, packed-refs,
        config) and ``UserInfo.git_url`` is read once per request in
        sanitization paths.
        """
        cached = self._git_url_cache
        if cached is not ...:
            return cached
        result = _git_url_from_info(_git_info(self.cwd))
        object.__setattr__(self, "_git_url_cache", result)
        return result

    def with_email(self, email: str | None) -> "UserInfo":
        first_name, last_name = _names_from_email(email)
        # The new email lands in :meth:`_singleton_key`, so this
        # constructor call resolves to a *different* singleton entry
        # than ``self`` — the original instance keeps its old email
        # untouched.
        return UserInfo(
            self.hostname,
            _key=self._key,
            _email=email,
            _first_name=first_name,
            _last_name=last_name,
            _product=self._product,
            _product_version=self._product_version,
        )

    def to_struct_dict(self) -> dict[str, Any]:
        """Flatten into the dict shape that matches :data:`USERINFO_STRUCT`.

        ``cwd`` / ``url`` / ``git_url`` are intentionally excluded —
        they're per-process derived values, not part of the wire
        contract; the receiver re-derives them from its own context.
        Reading the lazy properties here forces resolution and pins
        the result on the cache slots so subsequent reads are free.
        """
        return {
            "hash":            self.hash,
            "key":             self.key,
            "hostname":        self.hostname,
            "email":           self.email,
            "first_name":      self.first_name,
            "last_name":       self.last_name,
            "product":         self.product,
            "product_version": self.product_version,
        }

    @classmethod
    def from_struct_dict(cls, value: Mapping[str, Any]) -> "UserInfo":
        """Inverse of :meth:`to_struct_dict` — drops derived fields.

        Wire values land directly in the lazy cache slots so the
        receiver never re-resolves them. ``hash`` is recomputed by
        the property; ``cwd`` / ``url`` / ``git_url`` are
        per-process and are ignored if present on the input —
        reconstructing them from the struct would defeat the lazy
        contract on the receiving side, which re-derives them
        locally.
        """
        return cls(
            hostname=str(value.get("hostname") or ""),
            _key=str(value.get("key") or ""),
            _email=_or_none(value.get("email")),
            _first_name=_or_none(value.get("first_name")),
            _last_name=_or_none(value.get("last_name")),
            _product=_or_none(value.get("product")),
            _product_version=_or_none(value.get("product_version")),
        )


# ── public API ────────────────────────────────────────────────────────────────


def get_user_info(*, refresh: bool = False) -> UserInfo:
    """Return the cached :class:`UserInfo` for the current process.

    Only ``hostname`` is resolved eagerly — every other field is a
    lazy property on the instance that fires once on first access
    and persists. ``refresh=True`` drops the cached singleton and
    rebuilds, which also abandons any previously memoized lazy
    fields (the new instance starts unresolved).

    The cache is the per-class :attr:`UserInfo._INSTANCES` slot
    inherited from :class:`Singleton` — same primitive every other
    config-keyed singleton in the codebase uses.
    """
    hostname = socket.gethostname()
    if refresh:
        # Pop the matching cache entry so the next construction
        # builds a fresh instance with unresolved lazy slots.
        UserInfo(hostname=hostname).invalidate_singleton()
    return UserInfo(hostname=hostname)


# ── struct / hash helpers ─────────────────────────────────────────────────────


def _names_from_email(email: str | None) -> tuple[str | None, str | None]:
    parsed = parse_name_from_email(email or "")
    if parsed is None:
        return None, None
    first, last, _ = parsed
    return first, last


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
    """URL identifying the compute surface the process is running on.

    Resolution order — first hit wins, so a Databricks notebook
    running inside a Lambda-shaped container (unlikely but possible
    in custom environments) still surfaces as the workspace link
    that's actually meaningful to a human:

    1. Databricks workspace deep-link (DBR runtime).
    2. AWS Lambda console URL (``AWS_LAMBDA_FUNCTION_NAME``).
    3. AWS Batch console URL (``AWS_BATCH_JOB_ID``).
    4. ``local://<hostname>/<cwd>`` fallback.
    """
    dbx = _databricks_current_url(kind="auto")
    if dbx is not None:
        return dbx
    aws = _aws_current_url()
    if aws is not None:
        return aws
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


def _databricks_notebook_user() -> str | None:
    """User email pulled from the notebook ``dbutils`` context.

    The notebook execution context exposes the running user under
    several conventional tag keys (``user`` is the email-shaped
    one on modern DBR; ``userName`` predates it on older runtimes).
    Reading the tag is a pure in-process Java->JSON round trip — no
    SDK call, no network hop — so we prefer it over
    :func:`_get_dbx_user` (which issues an ``iam.users`` request).
    Returns ``None`` outside a notebook or when the tag is empty.
    """
    if not os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return None
    tags = _ctx_tags()
    return _pick(tags.get("user"), tags.get("userName"))


def _aws_lambda_identity() -> dict[str, str] | None:
    """Lambda function context from the runtime-injected env vars.

    ``AWS_LAMBDA_FUNCTION_NAME`` is the canonical detector (reserved
    by the runtime bootstrap, never user-settable). The rest are
    informational — version, region, log group — and may be absent
    in custom runtimes, so the keys are populated opportunistically.
    """
    name = os.getenv("AWS_LAMBDA_FUNCTION_NAME")
    if not name:
        return None
    info: dict[str, str] = {"function_name": name}
    for env_key, dict_key in (
        ("AWS_LAMBDA_FUNCTION_VERSION", "function_version"),
        ("AWS_REGION", "region"),
        ("AWS_DEFAULT_REGION", "region"),
        ("AWS_LAMBDA_LOG_GROUP_NAME", "log_group"),
        ("AWS_LAMBDA_LOG_STREAM_NAME", "log_stream"),
    ):
        value = os.getenv(env_key)
        if value and dict_key not in info:
            info[dict_key] = value
    return info


def _aws_batch_identity() -> dict[str, str] | None:
    """AWS Batch job context from the agent-injected env vars."""
    job_id = os.getenv("AWS_BATCH_JOB_ID")
    if not job_id:
        return None
    info: dict[str, str] = {"job_id": job_id}
    for env_key, dict_key in (
        ("AWS_BATCH_JQ_NAME", "job_queue"),
        ("AWS_BATCH_CE_NAME", "compute_environment"),
        ("AWS_BATCH_JOB_ATTEMPT", "attempt"),
        ("AWS_BATCH_JOB_NODE_INDEX", "node_index"),
        ("AWS_REGION", "region"),
        ("AWS_DEFAULT_REGION", "region"),
    ):
        value = os.getenv(env_key)
        if value and dict_key not in info:
            info[dict_key] = value
    return info


def _aws_current_url() -> URL | None:
    """Deep-link into the AWS console for the current compute surface.

    Lambda and Batch carry enough env-side context to build a
    permalink to the function / job in the AWS console; ECS / bare
    EC2 don't (the metadata URI alone doesn't identify the task in
    the console), so they fall through to the ``local://`` URL.
    """
    lam = _aws_lambda_identity()
    if lam is not None:
        region = lam.get("region") or "us-east-1"
        name = lam["function_name"]
        return URL.from_dict({
            "scheme": "https",
            "host": "console.aws.amazon.com",
            "path": "/lambda/home",
            "query": f"region={region}",
            "fragment": f"/functions/{name}",
        })

    batch = _aws_batch_identity()
    if batch is not None:
        region = batch.get("region") or "us-east-1"
        return URL.from_dict({
            "scheme": "https",
            "host": "console.aws.amazon.com",
            "path": "/batch/home",
            "query": f"region={region}",
            "fragment": f"jobs/detail/{batch['job_id']}",
        })

    return None


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


def _resolve_key() -> str:
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        # Notebook context first — pure in-process JSON, no SDK round
        # trip. The SDK path stays as the fallback for jobs / scripts
        # where ``dbutils`` isn't wired up.
        dbx_user = _databricks_notebook_user() or _get_dbx_user()
        if dbx_user:
            return dbx_user

    # Serverless / managed-container surfaces: ``whoami`` on Lambda
    # returns the sandbox user (``sbx_user1051``) and on Batch returns
    # ``root`` — neither is useful for tagging or attribution. Prefer
    # the service-side identifier the runtime already injected.
    lam = _aws_lambda_identity()
    if lam is not None:
        return f"lambda:{lam['function_name']}"
    batch = _aws_batch_identity()
    if batch is not None:
        return f"batch:{batch['job_id']}"

    whoami = _run_quiet(["whoami"])
    if whoami:
        return whoami.strip()
    return os.getenv("USERNAME") or os.getenv("USER") or os.getenv("LOGNAME") or "unknown"


def _resolve_email() -> str | None:
    return _get_upn_email() or _guess_email_from_env()


def _get_upn_email() -> str | None:
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        # Notebook context first — the ``user`` tag is already an
        # email on modern DBR; fall back to the SDK only if the
        # tag is absent (jobs, init scripts).
        dbx_user = _databricks_notebook_user() or _get_dbx_user()
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