"""Centralized Databricks compute node-type identifiers.

Databricks accepts a free-form ``node_type_id`` string (``"rd-fleet.xlarge"``,
``"m5.4xlarge"``, ``"Standard_D8ds_v5"``, ``"n2-standard-8"``, …). Every
caller used to spell its own default, which made size / cloud changes a
search-and-replace exercise and made it easy to typo a SKU.

:class:`NodeType` pins the convention to one place:

* canonical members for the SKUs Yggdrasil's compute services prefer
  (Fleet sizes for AWS Databricks, common Azure/GCP families);
* :attr:`NodeType.DEFAULT` — the workspace-cloud-agnostic Fleet default;
* semantic aliases (:attr:`SMALL` / :attr:`MEDIUM` / :attr:`LARGE` /
  :attr:`XLARGE`) that map to Fleet sizes;
* :meth:`from_` and :meth:`to_id` for forgiving coercion at API boundaries —
  unknown SKUs pass through as plain strings rather than rejecting valid
  cloud-specific identifiers callers may use.

Members subclass :class:`str`, so a :class:`NodeType` member is interchangeable
with the raw ``node_type_id`` string everywhere the Databricks SDK / REST API
expects one.

Usage::

    from yggdrasil.enums import NodeType

    # Direct use — the member is a str.
    spec = {"node_type_id": NodeType.DEFAULT}

    # Forgiving coercion (returns the original string for unknown SKUs).
    node_type_id = NodeType.to_id(user_provided_value)

    # Membership / known-SKU check.
    if NodeType.is_known(value):
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Iterable, Optional, Union

__all__ = ["NodeType", "NodeSpec"]


# ---------------------------------------------------------------------------
# Alias table — accepted spellings (case-insensitive) that resolve to a
# canonical ``node_type_id`` string. Covers the human-friendly "size" tokens
# Yggdrasil exposes (default / small / medium / large), Databricks' own Fleet
# SKU strings, and the bare SKU names from the major cloud node families.
# ---------------------------------------------------------------------------

_NODETYPE_ALIASES: dict[str, str] = {
    # Semantic defaults — resolved against Fleet for cloud portability.
    "":          "rd-fleet.xlarge",
    "default":   "rd-fleet.xlarge",
    "small":     "rd-fleet.xlarge",
    "medium":    "rd-fleet.2xlarge",
    "large":     "rd-fleet.4xlarge",
    "xlarge":    "rd-fleet.8xlarge",

    # Databricks Fleet (cloud-agnostic, the recommended default on AWS DBR).
    "fleet.xlarge":   "rd-fleet.xlarge",
    "fleet.2xlarge":  "rd-fleet.2xlarge",
    "fleet.4xlarge":  "rd-fleet.4xlarge",
    "fleet.8xlarge":  "rd-fleet.8xlarge",

    # AWS general-purpose
    "m5.large":     "m5.large",
    "m5.xlarge":    "m5.xlarge",
    "m5.2xlarge":   "m5.2xlarge",
    "m5.4xlarge":   "m5.4xlarge",
    "m5.8xlarge":   "m5.8xlarge",

    # AWS memory-optimized
    "r5.xlarge":    "r5.xlarge",
    "r5.2xlarge":   "r5.2xlarge",
    "r5.4xlarge":   "r5.4xlarge",

    # AWS memory-optimized + local NVMe (r5d — fast shuffle/spill)
    "r5d.xlarge":   "r5d.xlarge",
    "r5d.2xlarge":  "r5d.2xlarge",
    "r5d.4xlarge":  "r5d.4xlarge",

    # Azure general-purpose (Dds_v5 family — modern default for Databricks Azure)
    "standard_d4ds_v5":  "Standard_D4ds_v5",
    "standard_d8ds_v5":  "Standard_D8ds_v5",
    "standard_d16ds_v5": "Standard_D16ds_v5",

    # GCP general-purpose
    "n2-standard-4":  "n2-standard-4",
    "n2-standard-8":  "n2-standard-8",
    "n2-standard-16": "n2-standard-16",
}


class NodeType(str, Enum):
    """Canonical Databricks ``node_type_id`` values.

    Member values are the exact strings the Databricks SDK expects. Use
    :attr:`NodeType.DEFAULT` for the codebase-wide default node type, the
    semantic-size aliases (:attr:`SMALL` / :attr:`MEDIUM` / :attr:`LARGE` /
    :attr:`XLARGE`) for intent-driven sizing, or the explicit cloud-specific
    members when you need a precise SKU.

    The enum is intentionally **not exhaustive** — Databricks publishes
    hundreds of SKUs per cloud and most callers want a small, opinionated
    set. Coerce caller-provided strings through :meth:`to_id` /
    :meth:`from_` so unknown SKUs round-trip as plain strings instead of
    raising.
    """

    # --- Databricks Fleet (cloud-agnostic) -----------------------------
    FLEET_XLARGE    = "rd-fleet.xlarge"
    FLEET_2XLARGE   = "rd-fleet.2xlarge"
    FLEET_4XLARGE   = "rd-fleet.4xlarge"
    FLEET_8XLARGE   = "rd-fleet.8xlarge"

    # --- AWS general-purpose -------------------------------------------
    M5_LARGE        = "m5.large"
    M5_XLARGE       = "m5.xlarge"
    M5_2XLARGE      = "m5.2xlarge"
    M5_4XLARGE      = "m5.4xlarge"
    M5_8XLARGE      = "m5.8xlarge"

    # --- AWS memory-optimized ------------------------------------------
    R5_XLARGE       = "r5.xlarge"
    R5_2XLARGE      = "r5.2xlarge"
    R5_4XLARGE      = "r5.4xlarge"

    # --- AWS memory-optimized + local NVMe (r5d) -----------------------
    R5D_XLARGE      = "r5d.xlarge"
    R5D_2XLARGE     = "r5d.2xlarge"
    R5D_4XLARGE     = "r5d.4xlarge"

    # --- Azure general-purpose (Dds_v5) --------------------------------
    AZURE_D4DS_V5   = "Standard_D4ds_v5"
    AZURE_D8DS_V5   = "Standard_D8ds_v5"
    AZURE_D16DS_V5  = "Standard_D16ds_v5"

    # --- GCP general-purpose -------------------------------------------
    GCP_N2_STD_4    = "n2-standard-4"
    GCP_N2_STD_8    = "n2-standard-8"
    GCP_N2_STD_16   = "n2-standard-16"

    # Semantic aliases (collapsed into the matching member at class
    # creation — ``NodeType.DEFAULT is NodeType.FLEET_XLARGE``).
    DEFAULT: ClassVar["NodeType"]
    SMALL: ClassVar["NodeType"]
    MEDIUM: ClassVar["NodeType"]
    LARGE: ClassVar["NodeType"]
    XLARGE: ClassVar["NodeType"]

    # ------------------------------------------------------------------ #
    # Coercion
    # ------------------------------------------------------------------ #
    @classmethod
    def from_(
        cls,
        value: Any,
        *,
        default: Any = ...,
    ) -> "NodeType":
        """Coerce *value* into a :class:`NodeType` member.

        Accepts:

        * :class:`NodeType` (returned as-is);
        * any string in :data:`_NODETYPE_ALIASES`, case-insensitive;
        * a bare member name (``"FLEET_XLARGE"``);
        * ``None`` — returns *default* if supplied, else
          :attr:`NodeType.DEFAULT`.

        Raises :class:`ValueError` for unknown strings unless *default* is
        supplied. To accept arbitrary SKU strings (without rejecting valid
        cloud-specific identifiers), call :meth:`to_id` instead.
        """
        if isinstance(value, cls):
            return value

        if value is None:
            if default is not ...:
                return default
            return cls.DEFAULT

        if isinstance(value, str):
            # Fast path: callers commonly pass an already-canonical
            # token (``"default"`` / ``"rd-fleet.xlarge"`` /
            # ``"FLEET_XLARGE"`` / ``"m5.large"``). A single dict probe
            # resolves any of those without paying ``strip().lower()``.
            hit = _NODETYPE_LOOKUP.get(value)
            if hit is not None:
                return hit

            stripped = value.strip()
            hit = _NODETYPE_LOOKUP.get(stripped)
            if hit is not None:
                return hit
            hit = _NODETYPE_LOOKUP.get(stripped.lower())
            if hit is not None:
                return hit

        if default is not ...:
            return default
        raise ValueError(
            f"Unknown NodeType: {value!r}. "
            f"Use NodeType.to_id() to pass through arbitrary SKUs, or pick a "
            f"known member: {sorted(m.name for m in cls)!r}"
        )

    @classmethod
    def to_id(
        cls,
        value: Union[str, "NodeType", None],
        *,
        default: Optional[Union[str, "NodeType"]] = None,
    ) -> str:
        """Coerce *value* to a Databricks ``node_type_id`` string.

        Unlike :meth:`from_`, this is **forgiving**: arbitrary SKU strings
        round-trip unchanged so callers can pass cloud-specific identifiers
        the enum does not enumerate (``"r5d.metal"``, custom marketplace
        images, …). Aliases are resolved when present.

        Resolution order:

        1. ``NodeType`` member → its string value.
        2. Alias / member name / value match → the canonical string.
        3. Bare string → trimmed and returned as-is.
        4. ``None`` → ``default`` if supplied, else :attr:`NodeType.DEFAULT`.
        """
        if isinstance(value, cls):
            return value.value

        if value is None:
            if default is None:
                return cls.DEFAULT.value
            return cls.to_id(default)

        # Fast path: skip the ``str(value).strip()`` allocation when
        # the caller already handed us a canonical alias / member name
        # / value — the prebuilt lookup covers every common spelling.
        if isinstance(value, str):
            hit = _NODETYPE_LOOKUP.get(value)
            if hit is not None:
                return hit.value

        text = str(value).strip()
        if not text:
            return cls.DEFAULT.value

        hit = _NODETYPE_LOOKUP.get(text)
        if hit is not None:
            return hit.value
        hit = _NODETYPE_LOOKUP.get(text.lower())
        if hit is not None:
            return hit.value

        # Unknown SKU — passthrough so cloud-specific identifiers work.
        return text

    @classmethod
    def is_known(cls, value: Any) -> bool:
        """``True`` when :meth:`from_` would succeed for *value*."""
        try:
            cls.from_(value)
            return True
        except (TypeError, ValueError):
            return False


# Attach the semantic aliases. Python ``Enum`` collapses these into the
# matching member, so ``NodeType.DEFAULT is NodeType.FLEET_XLARGE``.
NodeType.DEFAULT = NodeType.FLEET_XLARGE  # type: ignore[misc]
NodeType.SMALL = NodeType.FLEET_XLARGE  # type: ignore[misc]
NodeType.MEDIUM = NodeType.FLEET_2XLARGE  # type: ignore[misc]
NodeType.LARGE = NodeType.FLEET_4XLARGE  # type: ignore[misc]
NodeType.XLARGE = NodeType.FLEET_8XLARGE  # type: ignore[misc]


def _build_nodetype_lookup() -> dict[str, NodeType]:
    """Pre-compute every accepted spelling → :class:`NodeType` member.

    Folds :data:`_NODETYPE_ALIASES` with member names (``"FLEET_XLARGE"``
    / ``"SMALL"`` / …), their lower-case form, and the canonical
    member values (``"rd-fleet.xlarge"`` / ``"Standard_D8ds_v5"`` /
    ``"m5.large"``) — including a case-folded variant — so both
    :meth:`NodeType.from_` and :meth:`NodeType.to_id` resolve any
    common SKU with a single ``dict.get``.
    """
    out: dict[str, NodeType] = {}
    for alias, canonical_value in _NODETYPE_ALIASES.items():
        member = NodeType(canonical_value)
        out[alias] = member
        upper = alias.upper()
        if upper != alias:
            out[upper] = member
    for member in NodeType:
        out[member.name] = member
        out[member.name.lower()] = member
        out[member.value] = member
        out[member.value.lower()] = member
    return out


_NODETYPE_LOOKUP: dict[str, NodeType] = _build_nodetype_lookup()


# ---------------------------------------------------------------------------
# Hardware specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Hardware specs for a :class:`NodeType` member.

    The numbers reflect the vendor's published per-VM characteristics for
    a single worker (or driver) — not the cluster-wide totals. ``ram_gib``
    is binary GiB (``1024 ** 3`` bytes), matching every cloud's own
    documentation convention.

    Attributes
    ----------
    cpu_cores
        Virtual CPU count exposed to Spark.
    ram_gib
        Memory available to Spark, in IEC GiB.
    gpu_count
        Number of GPUs (``0`` for non-GPU SKUs).
    local_disk_gib
        Local SSD/NVMe storage attached to the instance. ``0`` when the
        SKU has no local disk (relies on remote/elastic storage).
    cloud
        Cloud family identifier (``"fleet"`` / ``"aws"`` / ``"azure"`` /
        ``"gcp"``) used by :meth:`NodeType.from_cpu_and_ram` when
        ``prefer`` is set.
    """

    cpu_cores: int
    ram_gib: float
    gpu_count: int = 0
    local_disk_gib: float = 0.0
    cloud: str = ""


#: Per-:class:`NodeType` hardware spec table. The values come from the
#: vendor pages (Databricks Fleet sizing notes, AWS EC2 m5/r5, Azure
#: ``Dds_v5``, GCP n2-standard) and are the source of truth for
#: :meth:`NodeType.from_cpu_and_ram`. Add new members + entries here
#: together so the lookup keeps producing useful results.
_NODE_SPECS: dict[NodeType, NodeSpec] = {
    # Databricks Fleet — vCPU/RAM matches the m5 family on AWS.
    NodeType.FLEET_XLARGE:  NodeSpec(cpu_cores=4,  ram_gib=16,  cloud="fleet"),
    NodeType.FLEET_2XLARGE: NodeSpec(cpu_cores=8,  ram_gib=32,  cloud="fleet"),
    NodeType.FLEET_4XLARGE: NodeSpec(cpu_cores=16, ram_gib=64,  cloud="fleet"),
    NodeType.FLEET_8XLARGE: NodeSpec(cpu_cores=32, ram_gib=128, cloud="fleet"),

    # AWS general-purpose (m5)
    NodeType.M5_LARGE:      NodeSpec(cpu_cores=2,  ram_gib=8,   cloud="aws"),
    NodeType.M5_XLARGE:     NodeSpec(cpu_cores=4,  ram_gib=16,  cloud="aws"),
    NodeType.M5_2XLARGE:    NodeSpec(cpu_cores=8,  ram_gib=32,  cloud="aws"),
    NodeType.M5_4XLARGE:    NodeSpec(cpu_cores=16, ram_gib=64,  cloud="aws"),
    NodeType.M5_8XLARGE:    NodeSpec(cpu_cores=32, ram_gib=128, cloud="aws"),

    # AWS memory-optimized (r5) — 8 GiB per vCPU.
    NodeType.R5_XLARGE:     NodeSpec(cpu_cores=4,  ram_gib=32,  cloud="aws"),
    NodeType.R5_2XLARGE:    NodeSpec(cpu_cores=8,  ram_gib=64,  cloud="aws"),
    NodeType.R5_4XLARGE:    NodeSpec(cpu_cores=16, ram_gib=128, cloud="aws"),

    # AWS memory-optimized + local NVMe (r5d) — r5 RAM ratio plus a local
    # SSD (~37.5 GiB per vCPU), so shuffle / spill stays off remote storage.
    NodeType.R5D_XLARGE:    NodeSpec(cpu_cores=4,  ram_gib=32,  local_disk_gib=150, cloud="aws"),
    NodeType.R5D_2XLARGE:   NodeSpec(cpu_cores=8,  ram_gib=64,  local_disk_gib=300, cloud="aws"),
    NodeType.R5D_4XLARGE:   NodeSpec(cpu_cores=16, ram_gib=128, local_disk_gib=600, cloud="aws"),

    # Azure Dds_v5 — local NVMe disk scales with vCPU (~37.5 GiB per vCPU).
    NodeType.AZURE_D4DS_V5:  NodeSpec(cpu_cores=4,  ram_gib=16, local_disk_gib=150,  cloud="azure"),
    NodeType.AZURE_D8DS_V5:  NodeSpec(cpu_cores=8,  ram_gib=32, local_disk_gib=300,  cloud="azure"),
    NodeType.AZURE_D16DS_V5: NodeSpec(cpu_cores=16, ram_gib=64, local_disk_gib=600,  cloud="azure"),

    # GCP n2-standard
    NodeType.GCP_N2_STD_4:   NodeSpec(cpu_cores=4,  ram_gib=16, cloud="gcp"),
    NodeType.GCP_N2_STD_8:   NodeSpec(cpu_cores=8,  ram_gib=32, cloud="gcp"),
    NodeType.GCP_N2_STD_16:  NodeSpec(cpu_cores=16, ram_gib=64, cloud="gcp"),
}


def _spec_for(node: "NodeType") -> NodeSpec:
    try:
        return _NODE_SPECS[node]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"NodeType.{node.name} has no NodeSpec entry. "
            f"Add it to _NODE_SPECS in {__name__} to make hardware-aware "
            f"helpers like NodeType.from_cpu_and_ram see this member."
        ) from exc


def _node_spec_property(self: "NodeType") -> NodeSpec:
    return _spec_for(self)


def _node_cpu_cores(self: "NodeType") -> int:
    return _spec_for(self).cpu_cores


def _node_ram_gib(self: "NodeType") -> float:
    return _spec_for(self).ram_gib


def _node_gpu_count(self: "NodeType") -> int:
    return _spec_for(self).gpu_count


def _node_local_disk_gib(self: "NodeType") -> float:
    return _spec_for(self).local_disk_gib


def _node_cloud(self: "NodeType") -> str:
    return _spec_for(self).cloud


# Attach the spec accessors as properties on the enum class. We do this
# after the enum body so the property objects can close over the spec
# table without forward-reference juggling.
NodeType.spec           = property(_node_spec_property)  # type: ignore[attr-defined]
NodeType.cpu_cores      = property(_node_cpu_cores)      # type: ignore[attr-defined]
NodeType.ram_gib        = property(_node_ram_gib)        # type: ignore[attr-defined]
NodeType.gpu_count      = property(_node_gpu_count)      # type: ignore[attr-defined]
NodeType.local_disk_gib = property(_node_local_disk_gib) # type: ignore[attr-defined]
NodeType.cloud          = property(_node_cloud)          # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Best-fit lookup
# ---------------------------------------------------------------------------


_PREFER_ALIASES: dict[str, str] = {
    "fleet":     "fleet",
    "databricks": "fleet",
    "aws":       "aws",
    "amazon":    "aws",
    "ec2":       "aws",
    "azure":     "azure",
    "microsoft": "azure",
    "gcp":       "gcp",
    "google":    "gcp",
}


def _from_cpu_and_ram(
    cls,
    cpu_cores: int,
    ram_gib: Optional[float] = None,
    *,
    gpu: bool = False,
    local_disk_gib: float = 0.0,
    prefer: Optional[str] = "fleet",
    candidates: Optional[Iterable["NodeType"]] = None,
) -> "NodeType":
    """Return the smallest :class:`NodeType` that meets the requirements.

    Filters :data:`_NODE_SPECS` for entries with at least *cpu_cores* vCPU,
    at least *ram_gib* memory, at least *local_disk_gib* local SSD, and
    matching GPU presence (``gpu_count > 0`` when *gpu=True*). Ranks the
    survivors by ``(cpu_cores, ram_gib)`` ascending and picks the first.

    Parameters
    ----------
    cpu_cores
        Minimum vCPU count required.
    ram_gib
        Minimum memory required, in IEC GiB. ``None`` means no constraint.
    gpu
        When ``True``, require a member with ``gpu_count >= 1``. No GPU
        members are enumerated yet — passing ``gpu=True`` raises until one
        is added.
    local_disk_gib
        Minimum local SSD required, in IEC GiB.
    prefer
        Cloud family to prefer (``"fleet"`` / ``"aws"`` / ``"azure"`` /
        ``"gcp"``, also accepts ``"databricks"`` / ``"amazon"`` / ``"google"``).
        If the preferred family has a fit, it wins; otherwise other families
        are considered. Pass ``None`` to consider all members equally.
    candidates
        Optional iterable of :class:`NodeType` members to restrict the
        search to — useful for cloud-locked workspaces.

    Examples
    --------
    >>> NodeType.from_cpu_and_ram(cpu_cores=8, ram_gib=32)
    <NodeType.FLEET_2XLARGE: 'rd-fleet.2xlarge'>
    >>> NodeType.from_cpu_and_ram(cpu_cores=4, ram_gib=32)  # memory-bound
    <NodeType.R5_XLARGE: 'r5.xlarge'>
    >>> NodeType.from_cpu_and_ram(cpu_cores=4, prefer="azure")
    <NodeType.AZURE_D4DS_V5: 'Standard_D4ds_v5'>
    """
    if cpu_cores < 1:
        raise ValueError(f"cpu_cores must be >= 1, got {cpu_cores!r}")
    min_ram = float(ram_gib) if ram_gib is not None else 0.0
    min_disk = float(local_disk_gib)

    preferred_cloud: Optional[str] = None
    if prefer is not None:
        preferred_cloud = _PREFER_ALIASES.get(str(prefer).strip().lower())
        if preferred_cloud is None:
            raise ValueError(
                f"Unknown prefer cloud {prefer!r}. "
                f"Valid: {sorted(set(_PREFER_ALIASES.values()))!r}."
            )

    members: Iterable[NodeType] = candidates if candidates is not None else _NODE_SPECS.keys()

    fits: list[tuple[NodeType, NodeSpec]] = []
    for node in members:
        spec = _NODE_SPECS.get(node)
        if spec is None:
            continue
        if spec.cpu_cores < cpu_cores:
            continue
        if spec.ram_gib < min_ram:
            continue
        if spec.local_disk_gib < min_disk:
            continue
        if gpu and spec.gpu_count < 1:
            continue
        fits.append((node, spec))

    if not fits:
        raise ValueError(
            f"No NodeType satisfies cpu_cores>={cpu_cores} ram_gib>={min_ram} "
            f"local_disk_gib>={min_disk} gpu={gpu}. Add a matching member to "
            f"NodeType / _NODE_SPECS, or pass a cloud-specific node_type_id string."
        )

    def _rank(item: tuple[NodeType, NodeSpec]) -> tuple[int, ...]:
        node, spec = item
        # Prefer the requested cloud family; then smallest cpu, then ram.
        cloud_penalty = 0 if (preferred_cloud is None or spec.cloud == preferred_cloud) else 1
        return (cloud_penalty, spec.cpu_cores, int(spec.ram_gib))

    fits.sort(key=_rank)
    return fits[0][0]


# Bind the lookup as a classmethod on NodeType.
NodeType.from_cpu_and_ram = classmethod(_from_cpu_and_ram)  # type: ignore[attr-defined]
