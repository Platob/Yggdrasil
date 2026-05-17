"""Source-attribution metadata captured at deploy time.

When a :class:`Flow` is deployed, the workflow layer collects every
piece of provenance it can derive from the wrapped Python callables —
module path, source file + line, yggdrasil version, and (when the
file lives inside a git checkout) the current commit, branch, and an
HTTPS URL pointing at the exact line on GitHub / GitLab / Azure DevOps.

The metadata lands in three places:

* **Job tags** keyed under ``ygg.*`` — searchable in the Databricks
  UI ("Jobs › Filter by tag") and surfaced on every run.
* **Job description** — a human-readable footer that lists the source
  link, commit, and yggdrasil version.
* **Task description** — same shape, scoped to the task's own
  function. The auto-derived signature description from
  :func:`stage_python_callable` still leads.

All collection is best-effort: missing git, non-git source trees,
detached HEADs, and remotes the URL builder doesn't recognise all
degrade gracefully (the affected fields land as ``None`` / drop out
of the tag set).
"""
from __future__ import annotations

import inspect
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "SourceMetadata",
    "collect_source_metadata",
    "describe_metadata",
    "metadata_tags",
]

LOGGER = logging.getLogger(__name__)

#: Databricks job-tag value cap — the API rejects entries longer than
#: this. Mirrors the existing safe-tag rules in ``DatabricksClient``.
_TAG_VALUE_MAX = 256

#: Tag-value sanitisation: matches what the client-side ``safe_tag_value``
#: helper enforces (alphanumerics, basic punctuation, ``/`` / ``@``).
_TAG_VALUE_SANITIZE = re.compile(r"[^\d \w\+\-=\.:/@]+")


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Provenance bundle for one staged callable.

    Every field is optional — a callable defined inline (or in a REPL)
    can still produce a usable metadata bundle with just ``module``
    and ``qualname`` populated.
    """

    module: Optional[str] = None
    qualname: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    yggdrasil_version: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_remote: Optional[str] = None
    git_root: Optional[str] = None
    source_url: Optional[str] = None
    docstring: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict snapshot of every populated field."""
        out = asdict(self)
        # Drop empty extras to keep the embedded block readable.
        if not out.get("extra"):
            out.pop("extra", None)
        return {k: v for k, v in out.items() if v not in (None, "", {}, [])}


def collect_source_metadata(
    func: Callable[..., Any],
    *,
    extra: Optional[Dict[str, str]] = None,
) -> SourceMetadata:
    """Extract every piece of provenance we can derive from *func*.

    The returned :class:`SourceMetadata` is suitable for splatting into
    job tags via :func:`metadata_tags` and embedding into a job /
    task description via :func:`describe_metadata`.

    *extra* is an optional caller-supplied tag bag — e.g. ``{"team":
    "data"}`` from a CI environment variable — that surfaces alongside
    the auto-derived fields.
    """
    from yggdrasil.version import __version__ as ygg_version

    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)

    source_file: Optional[str] = None
    source_line: Optional[int] = None
    try:
        source_file = inspect.getsourcefile(func) or inspect.getfile(func)
    except (TypeError, OSError):
        pass
    try:
        _, source_line = inspect.getsourcelines(func)
        if source_line == 0:
            source_line = None
    except (TypeError, OSError):
        pass

    docstring = None
    raw_doc = inspect.getdoc(func)
    if raw_doc:
        # First non-empty line is what an operator scanning the
        # Databricks UI cares about — the rest is module-internal.
        first = raw_doc.strip().splitlines()[0]
        docstring = first if first else None

    git_info = _git_info(source_file) if source_file else _GitInfo()

    source_url: Optional[str] = None
    if git_info.remote and git_info.root and source_file:
        try:
            relpath = os.path.relpath(source_file, git_info.root)
        except ValueError:
            relpath = None
        if relpath and not relpath.startswith(".."):
            source_url = _build_source_url(
                remote=git_info.remote,
                ref=git_info.commit or git_info.branch,
                relpath=relpath.replace(os.sep, "/"),
                line=source_line,
            )

    return SourceMetadata(
        module=module,
        qualname=qualname,
        source_file=source_file,
        source_line=source_line,
        yggdrasil_version=str(ygg_version) if ygg_version else None,
        git_commit=git_info.commit,
        git_branch=git_info.branch,
        git_remote=git_info.remote,
        git_root=git_info.root,
        source_url=source_url,
        docstring=docstring,
        extra=dict(extra) if extra else {},
    )


# --------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------- #


def describe_metadata(
    metadata: SourceMetadata,
    *,
    prefix: str = "Source",
) -> str:
    """Render *metadata* as a multi-line human-readable footer.

    Used to append source-attribution to a job / task description so
    the Databricks UI surfaces the link without an operator having to
    crack the staged file open.
    """
    lines: List[str] = []
    if metadata.qualname or metadata.module:
        qual = ".".join(
            x for x in (metadata.module, metadata.qualname) if x
        )
        lines.append(f"{prefix}: {qual}")
    if metadata.source_url:
        lines.append(f"  {metadata.source_url}")
    elif metadata.source_file:
        suffix = f":{metadata.source_line}" if metadata.source_line else ""
        lines.append(f"  {metadata.source_file}{suffix}")
    if metadata.git_commit:
        ref = metadata.git_commit[:12]
        if metadata.git_branch:
            ref += f" ({metadata.git_branch})"
        lines.append(f"  git: {ref}")
    if metadata.yggdrasil_version:
        lines.append(f"  yggdrasil={metadata.yggdrasil_version}")
    if metadata.extra:
        for key, value in metadata.extra.items():
            lines.append(f"  {key}={value}")
    return "\n".join(lines)


def metadata_tags(
    metadata: SourceMetadata,
    *,
    prefix: str = "ygg",
) -> Dict[str, str]:
    """Project *metadata* into a Databricks-job-tag-shaped ``dict``.

    Tag keys use the ``<prefix>.<field>`` convention so operators can
    filter by ``ygg.flow`` / ``ygg.module`` / ``ygg.git_commit`` in
    the Databricks UI. Values are sanitised against the same rules
    :class:`DatabricksClient` applies (alphanumerics, basic
    punctuation) and truncated to 256 characters — the Jobs API rejects
    longer values.
    """
    raw: Dict[str, Optional[str]] = {
        f"{prefix}.module": metadata.module,
        f"{prefix}.qualname": metadata.qualname,
        f"{prefix}.source_file": metadata.source_file,
        f"{prefix}.source_url": metadata.source_url,
        f"{prefix}.git_commit": (
            metadata.git_commit[:12] if metadata.git_commit else None
        ),
        f"{prefix}.git_branch": metadata.git_branch,
        f"{prefix}.version": metadata.yggdrasil_version,
    }
    for key, value in (metadata.extra or {}).items():
        raw[f"{prefix}.{key}"] = value
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if not value:
            continue
        safe = _sanitise_tag_value(value)
        if safe:
            out[key] = safe
    return out


# --------------------------------------------------------------------- #
# Git info — best-effort
# --------------------------------------------------------------------- #


@dataclass(slots=True)
class _GitInfo:
    """Internal tuple of git facts about *one* source file."""

    root: Optional[str] = None
    commit: Optional[str] = None
    branch: Optional[str] = None
    remote: Optional[str] = None


def _git_info(source_file: str) -> _GitInfo:
    """Resolve git metadata for *source_file*'s containing repo.

    Cached per-directory so a flow with N tasks doesn't fire N
    ``git`` subprocesses. Returns an empty :class:`_GitInfo` when
    the file isn't inside a git checkout or ``git`` isn't on
    ``PATH``.
    """
    try:
        directory = str(Path(source_file).resolve().parent)
    except (OSError, ValueError):
        return _GitInfo()
    return _git_info_cached(directory)


@lru_cache(maxsize=64)
def _git_info_cached(directory: str) -> _GitInfo:
    root = _git_rev_parse(directory, "--show-toplevel")
    if not root:
        return _GitInfo()
    commit = _git_rev_parse(directory, "HEAD")
    # Branch: empty string in detached-HEAD state — keep as None.
    branch = _git_rev_parse(directory, "--abbrev-ref", "HEAD")
    if branch == "HEAD" or not branch:
        branch = None
    remote = _git_run(
        directory, "config", "--get", "remote.origin.url",
    ) or None
    return _GitInfo(root=root, commit=commit, branch=branch, remote=remote)


def _git_rev_parse(directory: str, *args: str) -> Optional[str]:
    return _git_run(directory, "rev-parse", *args)


def _git_run(directory: str, *args: str) -> Optional[str]:
    """Run ``git <args>`` in *directory*; return stdout stripped or ``None``."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        LOGGER.debug("git %s failed in %s: %s", args, directory, exc)
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out or None


# --------------------------------------------------------------------- #
# Source URL builder — GitHub, GitLab, Bitbucket, Azure DevOps
# --------------------------------------------------------------------- #


_SSH_REMOTE_RE = re.compile(r"^(?:ssh://)?git@(?P<host>[^:/]+)[:/](?P<path>.+?)(?:\.git)?$")
_HTTPS_REMOTE_RE = re.compile(r"^https?://(?:[^@/]+@)?(?P<host>[^/]+)/(?P<path>.+?)(?:\.git)?$")


def _build_source_url(
    *,
    remote: str,
    ref: Optional[str],
    relpath: str,
    line: Optional[int],
) -> Optional[str]:
    """Best-effort HTTPS link to *relpath* on *ref*.

    Supports the common remote shapes:

    - ``git@github.com:owner/repo.git`` → ``https://github.com/owner/repo/blob/<ref>/<relpath>#L<line>``
    - ``https://github.com/owner/repo.git`` → same
    - ``git@gitlab.com:owner/repo.git`` → ``https://gitlab.com/owner/repo/-/blob/<ref>/<relpath>#L<line>``
    - ``git@bitbucket.org:owner/repo.git`` → ``https://bitbucket.org/owner/repo/src/<ref>/<relpath>#lines-<line>``
    - Azure DevOps SSH / HTTPS — link points at the file browser

    Unknown hosts get a host-rooted but path-only URL with no
    ``blob/`` segment, which is still useful as a "where did this
    flow come from" pointer.
    """
    host, path = _parse_remote(remote)
    if not host or not path:
        return None
    ref_segment = ref or "main"
    host_l = host.lower()
    line_anchor = f"#L{line}" if line else ""
    if host_l == "github.com" or host_l.endswith(".github.com"):
        return f"https://{host}/{path}/blob/{ref_segment}/{relpath}{line_anchor}"
    if "gitlab" in host_l:
        return f"https://{host}/{path}/-/blob/{ref_segment}/{relpath}{line_anchor}"
    if "bitbucket" in host_l:
        bb_anchor = f"#lines-{line}" if line else ""
        return f"https://{host}/{path}/src/{ref_segment}/{relpath}{bb_anchor}"
    if host_l.endswith("azure.com") or host_l.endswith("visualstudio.com"):
        # Azure DevOps uses the ``?path=`` query shape.
        path_q = f"/{relpath}"
        ref_q = f"&version=GC{ref_segment}" if ref else ""
        line_q = f"&line={line}" if line else ""
        return f"https://{host}/{path}?path={path_q}{ref_q}{line_q}"
    return f"https://{host}/{path}"


def _parse_remote(remote: str) -> tuple[Optional[str], Optional[str]]:
    """Pick ``(host, repo_path)`` out of an SSH / HTTPS remote URL."""
    remote = remote.strip()
    m = _SSH_REMOTE_RE.match(remote) or _HTTPS_REMOTE_RE.match(remote)
    if not m:
        return None, None
    return m.group("host"), m.group("path").lstrip("/")


# --------------------------------------------------------------------- #
# Tag sanitisation
# --------------------------------------------------------------------- #


def _sanitise_tag_value(value: str) -> str:
    """Collapse disallowed characters and truncate to the Databricks limit."""
    cleaned = _TAG_VALUE_SANITIZE.sub("-", value)
    # Collapse runs of the replacement char so a long sanitisation
    # doesn't produce ``------``.
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("- ").strip()
    if len(cleaned) > _TAG_VALUE_MAX:
        cleaned = cleaned[: _TAG_VALUE_MAX - 1] + "…"
    return cleaned
