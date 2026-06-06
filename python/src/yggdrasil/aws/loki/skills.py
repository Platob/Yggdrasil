"""Specialized AWS Loki behaviors — one agent skill per AWS service.

A data-driven fleet: each :class:`_Spec` becomes an
:class:`AWSServiceSkill` instance that ``requires="aws"`` and drives one
boto3 service call through the project's :class:`~yggdrasil.aws.AWSClient`
(``agent.aws.client(service)``), summarizing the response to the identifying
field(s). All operations are **read/list/describe** — safe to run; nothing
here mutates AWS state.

    ygg loki run aws-s3                       # list buckets
    ygg loki run aws-s3-objects --kwarg Bucket='"my-bucket"'
    ygg loki run aws-ec2                      # instance ids + state
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import LokiSkill, register

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["AWSServiceSkill", "SPECS"]


@dataclasses.dataclass(frozen=True)
class _Spec:
    """One AWS list/describe operation → a Loki behavior."""

    name: str
    service: str
    method: str
    description: str
    #: Key in the response holding the items (``None`` → the whole response).
    list_key: Optional[str] = None
    #: Field(s) to keep per item: a dotted path, a tuple of them, or ``None``
    #: to keep the whole item. Strings in a plain list pass through as-is.
    item: Any = None
    #: For doubly-nested responses (EC2): ``list_key`` → ``[nested]`` → items.
    nested: Optional[str] = None
    #: Parameter names this call accepts (passed from ``run`` kwargs).
    needs: tuple[str, ...] = ()


def _dig(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def _pick(item: Any, fields: Any) -> Any:
    if fields is None or not isinstance(item, dict):
        return item
    if isinstance(fields, str):
        return _dig(item, fields)
    return {f.split(".")[-1]: _dig(item, f) for f in fields}


def _extract(resp: Any, spec: _Spec) -> list[Any]:
    if spec.nested:
        out: list[Any] = []
        for outer in (resp.get(spec.list_key) or []):
            for inner in (outer.get(spec.nested) or []):
                out.append(_pick(inner, spec.item))
        return out[:200]
    values = resp.get(spec.list_key) if spec.list_key else resp
    if isinstance(values, list):
        return [_pick(v, spec.item) for v in values][:200]
    return [_pick(values, spec.item)] if values is not None else []


class AWSServiceSkill(LokiSkill):
    """One AWS service skill, built from a :class:`_Spec`."""

    requires = "aws"
    preprompt = (
        "You are an AWS expert operating through yggdrasil's AWSClient. Prefer "
        "least-privilege IAM, the project's S3/path abstractions (S3Path) and "
        "Arrow/Parquet for data on S3, paginate list calls, and be explicit and "
        "safe with anything destructive or cost-bearing."
    )

    def __init__(self, spec: _Spec) -> None:
        self._spec = spec
        self.name = spec.name            # instance-level (shadows the ClassVar)
        self.description = spec.description

    def run(self, agent: "Loki", **params: Any) -> dict[str, Any]:
        client = agent.aws
        if client is None:  # available() guards; belt-and-suspenders
            raise RuntimeError("no AWS session — configure AWS credentials")
        spec = self._spec
        boto = client.client(spec.service)
        call_kwargs = {k: params[k] for k in spec.needs if k in params}
        resp = getattr(boto, spec.method)(**call_kwargs)
        if isinstance(resp, dict):
            resp = {k: v for k, v in resp.items() if k != "ResponseMetadata"}
        return {"service": spec.service, "method": spec.method,
                "items": _extract(resp, spec)}


#: The fleet — one entry per AWS service surface (read/list/describe only).
SPECS: tuple[_Spec, ...] = (
    _Spec("aws-identity", "sts", "get_caller_identity",
          "Who am I on AWS (STS caller identity).",
          item=("Account", "Arn", "UserId")),
    _Spec("aws-s3", "s3", "list_buckets", "List S3 buckets.",
          list_key="Buckets", item="Name"),
    _Spec("aws-s3-objects", "s3", "list_objects_v2", "List objects in an S3 bucket.",
          list_key="Contents", item="Key", needs=("Bucket", "Prefix")),
    _Spec("aws-ec2", "ec2", "describe_instances", "List EC2 instances (id + state).",
          list_key="Reservations", nested="Instances",
          item=("InstanceId", "InstanceType", "State.Name")),
    _Spec("aws-lambda", "lambda", "list_functions", "List Lambda functions.",
          list_key="Functions", item="FunctionName"),
    _Spec("aws-dynamodb", "dynamodb", "list_tables", "List DynamoDB tables.",
          list_key="TableNames"),
    _Spec("aws-iam-users", "iam", "list_users", "List IAM users.",
          list_key="Users", item="UserName"),
    _Spec("aws-iam-roles", "iam", "list_roles", "List IAM roles.",
          list_key="Roles", item="RoleName"),
    _Spec("aws-sqs", "sqs", "list_queues", "List SQS queues.", list_key="QueueUrls"),
    _Spec("aws-sns", "sns", "list_topics", "List SNS topics.",
          list_key="Topics", item="TopicArn"),
    _Spec("aws-ecr", "ecr", "describe_repositories", "List ECR repositories.",
          list_key="repositories", item="repositoryName"),
    _Spec("aws-logs", "logs", "describe_log_groups", "List CloudWatch log groups.",
          list_key="logGroups", item="logGroupName"),
    _Spec("aws-cloudwatch", "cloudwatch", "describe_alarms", "List CloudWatch alarms.",
          list_key="MetricAlarms", item="AlarmName"),
    _Spec("aws-stepfunctions", "stepfunctions", "list_state_machines",
          "List Step Functions state machines.", list_key="stateMachines", item="name"),
    _Spec("aws-ecs", "ecs", "list_clusters", "List ECS clusters.", list_key="clusterArns"),
    _Spec("aws-rds", "rds", "describe_db_instances", "List RDS database instances.",
          list_key="DBInstances", item="DBInstanceIdentifier"),
    _Spec("aws-glue", "glue", "get_databases", "List Glue catalog databases.",
          list_key="DatabaseList", item="Name"),
    _Spec("aws-secrets", "secretsmanager", "list_secrets",
          "List Secrets Manager secrets (names only).", list_key="SecretList", item="Name"),
    _Spec("aws-batch", "batch", "describe_job_queues", "List AWS Batch job queues.",
          list_key="jobQueues", item="jobQueueName"),
)

for _spec in SPECS:
    register(AWSServiceSkill(_spec))
