"""Declarative resource references for workflow tasks.

A :class:`SecretRef` is a *placeholder* — it stands in for a secret
value at decoration time without ever holding the cleartext. When
the workflow layer stages the task, ``repr(SecretRef("scope", "key"))``
renders a Python expression that resolves the secret at runtime via
:func:`yggdrasil.databricks.workflow.runtime.secret`. The staged ``.py``
on disk reads:

    func(api_key=_ygg_runtime.secret('scope', 'key'))

— the cluster fetches the cleartext from the Databricks Secrets API
the moment the task body needs it.

Usage:

    from yggdrasil.databricks.workflow import secret, task

    @task
    def call_vendor(payload: dict, api_key: str = secret("vendor", "api-key")):
        requests.post(VENDOR_URL, json=payload, headers={"Authorization": api_key})
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SecretRef", "secret"]


@dataclass(frozen=True, slots=True)
class SecretRef:
    """Late-bound reference to a Databricks secret scope/key pair.

    Construct via :func:`secret` rather than the class directly — the
    function accepts the natural ``("scope", "key")`` shape and the
    one-arg ``"scope/key"`` shortcut, matching :class:`Secrets`
    dict-style access.
    """

    scope: str
    key: str

    def __repr__(self) -> str:
        # The renderer in :mod:`yggdrasil.databricks.workflow.task` injects
        # ``import yggdrasil.databricks.workflow.runtime as _ygg_runtime`` at
        # the top of every staged script, so this literal resolves at
        # task-execution time. Using ``repr(self.scope)`` / ``repr(self.key)``
        # so embedded quotes / non-ASCII characters round-trip safely.
        return f"_ygg_runtime.secret({self.scope!r}, {self.key!r})"


def secret(scope: str, key: str | None = None, /) -> SecretRef:
    """Build a :class:`SecretRef` pointing at ``<scope>/<key>``.

    Accepts either the natural two-arg form (``secret("vendor",
    "api-key")``) or the single-string shortcut (``secret("vendor/api-key")``)
    so it lines up with :class:`Secrets`' dict-style access. Returns
    a frozen :class:`SecretRef` whose ``__repr__`` renders the runtime
    resolution call — pass it through as a parameter default, a keyword
    argument, or anywhere the staged task body expects a string.
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
    return SecretRef(scope=scope, key=key)
