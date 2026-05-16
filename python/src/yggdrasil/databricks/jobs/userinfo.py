"""
UserInfo-derived defaults for Databricks job creation.

The :class:`~yggdrasil.environ.UserInfo` snapshot already knows the
running user's email, git remote / branch / SHA, and project name —
all of which the typical "I just want to spin up a job from this
script" call wants to fill in by hand. These helpers translate that
snapshot into the Databricks SDK shapes
(:class:`JobEmailNotifications`, :class:`GitSource`, etc.) so the
:class:`Jobs` service can layer them under user-supplied overrides
without each caller re-deriving them.

The functions here only read the snapshot — they never mutate it,
never call into the workspace, and never fail loudly if a field is
missing (an absent git remote returns ``None`` rather than raising).
The caller decides which defaults to pull in via the kwargs on
:meth:`Jobs.userinfo_defaults`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from databricks.sdk.service.jobs import (
    GitProvider,
    GitSource,
    JobEmailNotifications,
)

# Hoisted: the private helper is one shared filesystem walk + parse;
# importing it once per module is cheaper than the per-call import
# the integration / repeated-tag paths would otherwise do.
from yggdrasil.environ.userinfo import _git_info

if TYPE_CHECKING:
    from yggdrasil.environ.userinfo import UserInfo


__all__ = [
    "userinfo_job_settings",
    "userinfo_git_source",
    "userinfo_email_notifications",
    "userinfo_tags",
]

LOGGER = logging.getLogger(__name__)


def _safe_git_info(cwd: str) -> Optional[dict[str, str]]:
    """Best-effort wrapper around :func:`_git_info` — swallow failures.

    The git probe walks the parents of *cwd* looking for ``.git``;
    every call hits the filesystem. Wrapping the swallow here lets
    every caller share the same defensive shape and keeps the
    happy-path readable.
    """
    try:
        return _git_info(cwd)
    except Exception:  # noqa: BLE001 — best-effort
        return None


#: Mapping of git remote hostname → :class:`GitProvider`. Covers the
#: providers Databricks Repos understands; anything else surfaces as
#: ``None`` and the git source is skipped (Databricks rejects an
#: unset provider).
_GIT_PROVIDER_BY_HOST: dict[str, GitProvider] = {
    "github.com":         GitProvider.GIT_HUB,
    "gitlab.com":         GitProvider.GIT_LAB,
    "bitbucket.org":      GitProvider.BITBUCKET_CLOUD,
    "dev.azure.com":      GitProvider.AZURE_DEV_OPS_SERVICES,
    "ssh.dev.azure.com":  GitProvider.AZURE_DEV_OPS_SERVICES,
    "visualstudio.com":   GitProvider.AZURE_DEV_OPS_SERVICES,
}


def userinfo_job_settings(
    info: "UserInfo",
    *,
    include_git_source: bool = True,
    include_notifications: bool = True,
    include_tags: bool = True,
    notification_events: tuple[str, ...] = ("on_failure",),
) -> dict[str, Any]:
    """Render *info* as a Databricks ``JobSettings``-shaped dict.

    The returned dict only contains keys for which *info* has a real
    value, so callers can splat it under explicit overrides without
    shadowing intentional ``None`` slots::

        defaults = userinfo_job_settings(UserInfo.current())
        client.jobs.create(name="etl", tasks=[...], **defaults, tags={"Env": "prod"})

    Toggles let callers opt out of any individual derivation without
    needing to re-build the dict — useful in jobs that need
    workspace-pinned notifications (e.g. ``#data-eng`` Slack webhook)
    rather than the running user's email.
    """
    out: dict[str, Any] = {}

    if include_git_source:
        git = userinfo_git_source(info)
        if git is not None:
            out["git_source"] = git

    if include_notifications:
        notifications = userinfo_email_notifications(
            info, events=notification_events,
        )
        if notifications is not None:
            out["email_notifications"] = notifications

    if include_tags:
        tags = userinfo_tags(info)
        if tags:
            out["tags"] = tags

    return out


def userinfo_git_source(info: "UserInfo") -> Optional[GitSource]:
    """Build a :class:`GitSource` for *info*'s local git checkout.

    Returns ``None`` when the snapshot has no resolvable git remote
    or the remote host isn't one of the providers Databricks Repos
    supports. Branch defaults to the working-tree branch; the SHA
    (when known) wins over the branch so triggered runs are
    reproducible against the exact commit the caller staged from.
    """
    git_url = info.git_url
    if git_url is None:
        return None
    host = (git_url.host or "").lower()
    provider = _GIT_PROVIDER_BY_HOST.get(host)
    if provider is None:
        LOGGER.debug(
            "userinfo_git_source: no GitProvider mapping for host %r — "
            "skipping git_source default", host,
        )
        return None

    # Strip any in-URL fragment / query before handing to Databricks —
    # the SDK expects a clean ``https://host/owner/repo`` URL and pulls
    # the commit out of ``git_commit`` instead.
    base_url = git_url.with_fragment(None).with_query(None).to_string()
    sha = git_url.fragment or None

    # ``info.git_url`` carries the SHA in its fragment when available;
    # the branch isn't part of the URL, so peek at the cached git-info
    # for it.
    ginfo = _safe_git_info(info.cwd)
    branch = ginfo.get("git_branch") if ginfo else None

    return GitSource(
        git_url=base_url,
        git_provider=provider,
        git_commit=sha,
        git_branch=branch if not sha else None,
    )


def userinfo_email_notifications(
    info: "UserInfo",
    *,
    events: tuple[str, ...] = ("on_failure",),
) -> Optional[JobEmailNotifications]:
    """Build a :class:`JobEmailNotifications` aimed at *info.email*.

    Returns ``None`` when the snapshot doesn't have an email — the
    notification dict would be empty and Databricks would reject it.
    *events* is a subset of ``{"on_start", "on_success", "on_failure",
    "on_duration_warning_threshold_exceeded"}``; default
    ``("on_failure",)`` mirrors the most common "tell me when it
    breaks" wiring.
    """
    email = info.email
    if not email:
        return None

    # One list-per-event so JobEmailNotifications doesn't alias the
    # same list across slots (Databricks SDK would mutate the shared
    # ref on round-trip).
    return JobEmailNotifications(**{event: [email] for event in events})


def userinfo_tags(info: "UserInfo") -> dict[str, str]:
    """Derive a set of resource tags from *info*'s git + project state.

    Layered on top of :meth:`DatabricksClient.default_tags` — the
    client-level tags carry Owner / Hostname / Product; this set
    adds the per-job source-of-truth tags (git remote, branch, SHA,
    notebook / compute URL) that change between checkouts. Empty
    values are dropped so the merged tag dict stays minimal.
    """
    out: dict[str, str] = {}
    git_url = info.git_url
    if git_url is not None:
        out["GitUrl"] = git_url.with_fragment(None).with_query(None).to_string()
        if git_url.fragment:
            out["GitCommit"] = git_url.fragment

    ginfo = _safe_git_info(info.cwd)
    if ginfo and ginfo.get("git_branch"):
        out["GitBranch"] = ginfo["git_branch"]

    url = info.url
    if url is not None:
        out["StagedFrom"] = url.to_string()

    return out
