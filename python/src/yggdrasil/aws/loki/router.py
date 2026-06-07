"""Route a natural-language AWS request to a specialized ``aws-*`` skill.

In the interactive session, "list my s3 buckets", "show ec2 instances", "what
lambda functions do I have", "who am I on aws" should *do the thing* — dispatch
the right ``aws-*`` skill — not just reason about it. :func:`route` maps the
user's line to ``(skill_name, kwargs)`` for the common, unambiguous **read**
requests, or ``None`` to fall back to reasoning.

Read-only by design: every target is a list/describe skill; nothing here changes
state. Mirrors :mod:`yggdrasil.databricks.loki.router`.
"""
from __future__ import annotations

import re

__all__ = ["route"]

#: Ordered (regex, skill) rules — first match wins. Specific phrases before broad
#: ones so "s3 objects in X" beats the bare "s3 buckets" rule. Data over code:
#: add a row, not a branch.
_RULES: tuple[tuple[str, str], ...] = (
    (r"\b(who am i|caller identity|sts|whoami)\b", "aws-identity"),
    (r"\b(objects?|files?)\b.*\bin\b.*\bbucket\b|\bs3 objects?\b", "aws-s3-objects"),
    (r"\b(s3 buckets?|buckets?)\b", "aws-s3"),
    (r"\b(ec2|instances?)\b", "aws-ec2"),
    (r"\b(lambda functions?|lambdas?|functions?)\b", "aws-lambda"),
    (r"\b(ecr|container repos?|image repos?|repositories)\b", "aws-ecr"),
    (r"\b(ecs|container clusters?)\b", "aws-ecs"),
    (r"\b(iam roles?|roles?)\b", "aws-iam-roles"),
    (r"\b(iam users?|users?)\b", "aws-iam-users"),
    (r"\b(rds|database instances?|databases?)\b", "aws-rds"),
    (r"\b(dynamodb|dynamo)\b", "aws-dynamodb"),
    (r"\b(sqs|queues?)\b", "aws-sqs"),
    (r"\b(sns|topics?)\b", "aws-sns"),
    (r"\b(log groups?|cloudwatch logs?)\b", "aws-logs"),
    (r"\b(alarms?|cloudwatch)\b", "aws-cloudwatch"),
    (r"\b(secrets?|secrets manager)\b", "aws-secrets"),
    (r"\b(glue|glue databases?)\b", "aws-glue"),
    (r"\b(batch job queues?|aws batch|batch queues?)\b", "aws-batch"),
    (r"\b(step functions?|state machines?)\b", "aws-stepfunctions"),
)

#: Only treat a line as an AWS request when it reads like one — a list/show verb
#: or an explicit AWS mention — so "users" alone in a chat doesn't fire it.
_LIST_VERB = re.compile(r"\b(list|show|what|which|how many|get|display|my)\b", re.I)
_AWS_HINT = re.compile(r"\b(aws|amazon|s3|buckets?|ec2|lambda|ecs|ecr|iam|rds|dynamodb|"
                       r"sqs|sns|cloudwatch|glue|sts|step functions?|secrets manager)\b", re.I)


def route(text: str) -> "tuple[str, dict] | None":
    """Map a NL AWS request → ``(skill, kwargs)``, or ``None`` to reason."""
    low = text.lower()
    if not (_AWS_HINT.search(low) and (_LIST_VERB.search(low) or _AWS_HINT.search(low))):
        return None
    for pattern, skill in _RULES:
        if re.search(pattern, low):
            if skill == "aws-s3-objects":
                m = (re.search(r"\bbucket\s+([a-z0-9.\-]{3,})", low)        # "bucket my-data"
                     or re.search(r"\b(?:in|from)\s+([a-z0-9.\-]{3,})\b", low))  # "in my-logs"
                bucket = m.group(1) if m else None
                if bucket in (None, "bucket", "buckets"):
                    return "aws-s3", {}                  # no real bucket named → list buckets
                return skill, {"bucket": bucket}
            return skill, {}
    return None
