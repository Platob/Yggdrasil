"""Declarative resource references for workflow tasks.

A :class:`SecretRef` is a *placeholder* — it stands in for a secret
value at decoration time without ever holding the cleartext. When
the workflow layer stages the task, ``repr(SecretRef("scope", "key"))``
renders a Python expression that resolves the secret at runtime via
:func:`yggdrasil.databricks.workflow.ygg.secret`. The staged ``.py``
on disk reads:

    func(api_key=ygg.secret('scope', 'key'))

— the cluster fetches the cleartext from the Databricks Secrets API
the moment the task body needs it.

Usage:

    from yggdrasil.databricks.workflow import secret, task

    @task
    def call_vendor(payload: dict, api_key: str = secret("vendor", "api-key")):
        requests.post(VENDOR_URL, json=payload, headers={"Authorization": api_key})

To resolve against a workspace *other* than
:meth:`DatabricksClient.current`, pin the target host via
``secret(..., client=...)`` — both a live :class:`DatabricksClient`
and a bare host string work; the staged repr renders
``ygg.secret('scope', 'key', host='https://…')`` so the cluster builds
a fresh client for that workspace at call time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient

__all__ = ["SecretRef", "secret"]


@dataclass(frozen=True, slots=True)
class SecretRef:
    """Late-bound reference to a Databricks secret scope/key pair.

    Construct via :func:`secret` rather than the class directly — the
    function accepts the natural ``("scope", "key")`` shape and the
    one-arg ``"scope/key"`` shortcut, matching :class:`Secrets`
    dict-style access.

    ``host`` (optional) targets a workspace other than the active
    :meth:`DatabricksClient.current`. Stored as a string for pickle /
    repr stability — a live :class:`DatabricksClient` passed via
    ``secret(..., client=client)`` is reduced to its base URL at
    construction time.
    """

    scope: str
    key: str
    host: Optional[str] = None

    def __repr__(self) -> str:
        # The renderer in :mod:`yggdrasil.databricks.jobs.task` injects
        # ``from yggdrasil.databricks.workflow import ygg`` at the top
        # of every staged script, so this literal resolves at
        # task-execution time. Using ``repr(self.scope)`` /
        # ``repr(self.key)`` so embedded quotes / non-ASCII characters
        # round-trip safely.
        if self.host:
            return (
                f"ygg.secret({self.scope!r}, {self.key!r}, "
                f"host={self.host!r})"
            )
        return f"ygg.secret({self.scope!r}, {self.key!r})"


def secret(
    scope: str,
    key: Optional[str] = None,
    /,
    *,
    client: "Union[str, DatabricksClient, None]" = None,
) -> SecretRef:
    """Build a :class:`SecretRef` pointing at ``<scope>/<key>``.

    Accepts either the natural two-arg form (``secret("vendor",
    "api-key")``) or the single-string shortcut (``secret("vendor/api-key")``)
    so it lines up with :class:`Secrets`' dict-style access. Returns
    a frozen :class:`SecretRef` whose ``__repr__`` renders the runtime
    resolution call — pass it through as a parameter default, a keyword
    argument, or anywhere the staged task body expects a string.

    ``client`` targets a workspace other than
    :meth:`DatabricksClient.current` for resolution. Accepts a live
    :class:`DatabricksClient` (the base URL is extracted) or a bare
    host URL string. ``None`` (default) leaves the SecretRef
    workspace-agnostic; whichever ``DatabricksClient`` is current at
    call time resolves it.
    """
    if key is None:
        if "/" in scope:
            scope, key = scope.split("/", 1)
        elif ":" in scope:
            scope, key = scope.split(":", 1)
        else:
            raise ValueError(
                f"secret({scope!r}): single-argument form expects "
                f"'<scope>/<key>' or '<scope>:<key>'; got no separator. "
                "Pass key as a second argument or include '/' in the spec."
            )
    if not scope or not key:
        raise ValueError(
            f"secret(scope={scope!r}, key={key!r}): both scope and key "
            "must be non-empty."
        )
    host: Optional[str] = None
    if client is not None:
        if isinstance(client, str):
            host = client
        else:
            # Live DatabricksClient — extract a stable hostname string.
            host = str(getattr(client, "base_url", None) or "") or None
            if not host:
                raise ValueError(
                    f"secret(..., client={client!r}): could not derive a "
                    "host URL — pass the host string directly."
                )
    return SecretRef(scope=scope, key=key, host=host)
