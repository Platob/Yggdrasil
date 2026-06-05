"""AWS region enum + region-from-text inference.

:class:`AWSRegion` is a ``str``-backed enum of every public AWS region code, so
a region round-trips as a plain string everywhere (URLs, SigV4 scopes, console
hosts) while still being a typed, comparable member.

:meth:`AWSRegion.from_text` recovers a region embedded in arbitrary text —
people routinely bake it into bucket names (``acme-dls3-eu-central-1-p``), so
when no region is configured the S3 layer falls back to sniffing it out of the
bucket name before defaulting to ``us-east-1``.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

__all__ = ["AWSRegion"]


class AWSRegion(str, Enum):
    # --- commercial (aws partition) ---
    US_EAST_1 = "us-east-1"
    US_EAST_2 = "us-east-2"
    US_WEST_1 = "us-west-1"
    US_WEST_2 = "us-west-2"
    AF_SOUTH_1 = "af-south-1"
    AP_EAST_1 = "ap-east-1"
    AP_EAST_2 = "ap-east-2"
    AP_SOUTH_1 = "ap-south-1"
    AP_SOUTH_2 = "ap-south-2"
    AP_NORTHEAST_1 = "ap-northeast-1"
    AP_NORTHEAST_2 = "ap-northeast-2"
    AP_NORTHEAST_3 = "ap-northeast-3"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_SOUTHEAST_2 = "ap-southeast-2"
    AP_SOUTHEAST_3 = "ap-southeast-3"
    AP_SOUTHEAST_4 = "ap-southeast-4"
    AP_SOUTHEAST_5 = "ap-southeast-5"
    AP_SOUTHEAST_7 = "ap-southeast-7"
    CA_CENTRAL_1 = "ca-central-1"
    CA_WEST_1 = "ca-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    EU_CENTRAL_2 = "eu-central-2"
    EU_WEST_1 = "eu-west-1"
    EU_WEST_2 = "eu-west-2"
    EU_WEST_3 = "eu-west-3"
    EU_NORTH_1 = "eu-north-1"
    EU_SOUTH_1 = "eu-south-1"
    EU_SOUTH_2 = "eu-south-2"
    IL_CENTRAL_1 = "il-central-1"
    ME_SOUTH_1 = "me-south-1"
    ME_CENTRAL_1 = "me-central-1"
    MX_CENTRAL_1 = "mx-central-1"
    SA_EAST_1 = "sa-east-1"
    # --- GovCloud (aws-us-gov partition) ---
    US_GOV_EAST_1 = "us-gov-east-1"
    US_GOV_WEST_1 = "us-gov-west-1"
    # --- China (aws-cn partition) ---
    CN_NORTH_1 = "cn-north-1"
    CN_NORTHWEST_1 = "cn-northwest-1"

    def __str__(self) -> str:
        return self.value

    @property
    def partition(self) -> str:
        """The AWS partition this region belongs to (``aws`` / ``aws-us-gov`` /
        ``aws-cn``)."""
        if self.value.startswith("cn-"):
            return "aws-cn"
        if self.value.startswith("us-gov-"):
            return "aws-us-gov"
        return "aws"

    @classmethod
    def from_(cls, value: Any, *, default: Any = ...) -> "Optional[AWSRegion]":
        """Coerce *value* (an :class:`AWSRegion`, a region string, or ``None``)
        to a member. Unknown / ``None`` returns *default* if given, else
        raises."""
        if isinstance(value, cls):
            return value
        if value is None:
            if default is not ...:
                return default
            raise ValueError("AWSRegion cannot be derived from None")
        if isinstance(value, str):
            hit = _LOOKUP.get(value.strip().lower())
            if hit is not None:
                return hit
        if default is not ...:
            return default
        raise ValueError(f"Unknown AWS region: {value!r}")

    @classmethod
    def from_text(cls, text: Any, *, default: Any = None) -> "Optional[AWSRegion]":
        """Find an AWS region code embedded in *text* (e.g. a bucket name like
        ``acme-dls3-eu-central-1-p``), or *default* if none is present.

        Matches a region only when it stands as a whole, delimiter-bounded
        token (so ``eu-central-12`` / ``...-1foo`` don't false-positive), and
        prefers the longest code at a position (``ap-southeast-1`` over
        ``ap-south-1``)."""
        if not text:
            return default
        m = _REGION_RE.search(str(text).lower())
        return cls(m.group(1)) if m else default


_LOOKUP: "dict[str, AWSRegion]" = {r.value: r for r in AWSRegion}

# Region codes, longest-first so the alternation prefers the more specific code
# at a shared position; bounded by non-[a-z0-9] so they only match whole tokens.
_REGION_RE = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(re.escape(r.value) for r in sorted(AWSRegion, key=lambda r: len(r.value), reverse=True))
    + r")(?![a-z0-9])"
)
