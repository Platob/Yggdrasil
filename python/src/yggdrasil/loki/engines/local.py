"""Shared base for local (on-workstation) engines.

A local model is free and private but **bounded by the box**, so — unlike a
remote engine, which adapts its model to the *prompt's* cost tier — a local
engine adapts its model to the *machine's* resources: the more CPU/RAM/GPU, the
larger the default model it loads (:mod:`yggdrasil.loki.resources`). Concrete
local engines (``transformers``, ``ollama``) just declare a
:attr:`RESOURCE_MODELS` ladder; sizing, labelling, and the bootstrap model all
fall out of it here.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from ..engine import TokenEngine


class LocalEngine(TokenEngine):
    """A :class:`TokenEngine` that runs on this workstation, sized to it."""

    local = True
    #: size tier (``small``/``medium``/``large``/``xlarge``) → model id. The
    #: machine picks the row (see :func:`yggdrasil.loki.resources.size_tier`).
    RESOURCE_MODELS: ClassVar[dict[str, str]] = {}

    @property
    def bootstrap_model(self) -> str:
        """The default model for this box — the resource-sized row, or the
        engine's :attr:`default_model` fallback."""
        from ..resources import size_tier

        return self.RESOURCE_MODELS.get(size_tier()) or str(self.default_model)

    def resolve_model(
        self,
        *,
        messages: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> Optional[str]:
        """An explicit pin wins; otherwise the model sized to this workstation.

        Local models are resource-bound, so the remote ``fast``/``deep`` cost
        tier doesn't apply — the box, not the prompt, picks the size.
        """
        return self.model or self.bootstrap_model

    @property
    def model_label(self) -> str:
        if self.model:
            return self.model
        from ..resources import size_tier

        return f"{self.bootstrap_model} (resources: {size_tier()})"
