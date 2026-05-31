"""AWS Batch runtime — resource + service.

When a job runs under AWS Batch the agent injects a well-known set of env vars
(job id, attempt, queue, compute environment, array / multi-node topology). This
module exposes them the same way the rest of the AWS surface is shaped:

* :class:`BatchService` — the :class:`~yggdrasil.aws.client.AWSService` binding
  (``service_name == "batch"``), so ``BatchService.current()`` and
  ``client.batch`` follow the established singleton pattern instead of a
  free-floating dataclass.
* :class:`AWSBatch` — the :class:`~yggdrasil.aws.client.AWSResource` carrying
  the captured runtime context, a clickable :attr:`explore_url` to the job in
  the Console, and the ``is_*`` flags. Reach it as ``client.batch`` or
  ``AWSBatch.current()``.

:func:`in_aws_environment` is the lightweight, network-free "are we running
inside AWS?" probe (Batch / ECS / Fargate / Lambda — all set a tell-tale env
var); the S3 layer uses it to skip an egress proxy that doesn't apply in-VPC.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from yggdrasil.aws.client import AWSResource, AWSService
from yggdrasil.aws.console import batch_job_url
from yggdrasil.url import URL

__all__ = ["AWSBatch", "BatchService", "in_aws_environment"]

# Env vars any AWS-managed compute sets — used as a network-free "in AWS" probe.
_AWS_ENV_SIGNALS = (
    "AWS_EXECUTION_ENV",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_BATCH_JOB_ID",
    "AWS_LAMBDA_FUNCTION_NAME",
    "ECS_CONTAINER_METADATA_URI",
    "ECS_CONTAINER_METADATA_URI_V4",
)


def in_aws_environment(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when running inside AWS-managed compute (Batch / ECS / Fargate /
    Lambda). Pure env-var probe — no IMDS round trip."""
    e = os.environ if env is None else env
    return any(e.get(k) for k in _AWS_ENV_SIGNALS)


def _int(env: Mapping[str, str], name: str) -> Optional[int]:
    raw = env.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class BatchService(AWSService):
    """AWS Batch service binding (the boto ``batch`` client lives on
    :attr:`boto_client` for future control-plane calls)."""

    @classmethod
    def service_name(cls) -> str:
        return "batch"


class AWSBatch(AWSResource):
    """The AWS Batch runtime context this process is running under.

        >>> AWSClient.current().batch.is_batch
        True
        >>> AWSClient.current().batch
        AWSBatch(URL('https://eu-west-1.console.aws.amazon.com/batch/home?region=eu-west-1#jobs/detail/abc'))

    Off-Batch every field is ``None`` and :attr:`is_batch` is ``False``; the
    captured env is snapshotted at construction so the resource is stable and
    picklable.
    """

    def __init__(
        self,
        service: Optional[AWSService] = None,
        *,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._env: dict[str, str] = dict(os.environ if env is None else env)
        super().__init__(service=service if service is not None else BatchService.current())

    @classmethod
    def current(cls, *, env: Optional[Mapping[str, str]] = None) -> "AWSBatch":
        """The Batch context for the default client + live process env."""
        return cls(service=BatchService.current(), env=env)

    # -- pickling: keep the env snapshot alongside the service ----------
    def __getstate__(self) -> dict:
        return {"service": self.service, "env": self._env}

    def __setstate__(self, state: dict) -> None:
        self.service = state["service"]
        self._env = state.get("env", {})

    # -- captured fields ------------------------------------------------
    @property
    def job_id(self) -> Optional[str]:
        return self._env.get("AWS_BATCH_JOB_ID") or None

    @property
    def job_attempt(self) -> Optional[int]:
        return _int(self._env, "AWS_BATCH_JOB_ATTEMPT")

    @property
    def job_queue(self) -> Optional[str]:
        return self._env.get("AWS_BATCH_JQ_NAME") or None

    @property
    def compute_environment(self) -> Optional[str]:
        return self._env.get("AWS_BATCH_CE_NAME") or None

    @property
    def array_index(self) -> Optional[int]:
        return _int(self._env, "AWS_BATCH_JOB_ARRAY_INDEX")

    @property
    def node_index(self) -> Optional[int]:
        return _int(self._env, "AWS_BATCH_JOB_NODE_INDEX")

    @property
    def main_node_index(self) -> Optional[int]:
        return _int(self._env, "AWS_BATCH_JOB_MAIN_NODE_INDEX")

    @property
    def num_nodes(self) -> Optional[int]:
        return _int(self._env, "AWS_BATCH_JOB_NUM_NODES")

    @property
    def main_node_ip(self) -> Optional[str]:
        return self._env.get("AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS") or None

    @property
    def execution_env(self) -> Optional[str]:
        return self._env.get("AWS_EXECUTION_ENV") or None

    @property
    def region(self) -> Optional[str]:
        return (
            self._env.get("AWS_REGION")
            or self._env.get("AWS_DEFAULT_REGION")
            or self.service.region
        )

    @property
    def container_credentials_uri(self) -> Optional[str]:
        return (
            self._env.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            or self._env.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            or None
        )

    # -- derived flags --------------------------------------------------
    @property
    def is_batch(self) -> bool:
        return self.job_id is not None

    @property
    def is_aws_environment(self) -> bool:
        return in_aws_environment(self._env)

    @property
    def is_array_job(self) -> bool:
        return self.array_index is not None

    @property
    def is_multinode(self) -> bool:
        n = self.num_nodes
        return bool(n and n > 1)

    @property
    def is_main_node(self) -> bool:
        """True for a single-node job, or the main node of a multi-node job."""
        if not self.is_multinode:
            return True
        return self.node_index is not None and self.node_index == self.main_node_index

    @property
    def is_fargate(self) -> bool:
        ee = self.execution_env
        return bool(ee and "FARGATE" in ee.upper())

    @property
    def has_container_credentials(self) -> bool:
        return self.container_credentials_uri is not None

    # -- explore / serialize -------------------------------------------
    @property
    def explore_url(self) -> Optional[URL]:
        """Console deep-link to this Batch job, or ``None`` off-Batch."""
        return batch_job_url(self.job_id, self.region) if self.job_id else None

    def to_dict(self) -> dict:
        return {
            "is_batch": self.is_batch,
            "job_id": self.job_id,
            "job_attempt": self.job_attempt,
            "job_queue": self.job_queue,
            "compute_environment": self.compute_environment,
            "array_index": self.array_index,
            "node_index": self.node_index,
            "main_node_index": self.main_node_index,
            "num_nodes": self.num_nodes,
            "main_node_ip": self.main_node_ip,
            "execution_env": self.execution_env,
            "region": self.region,
            "is_array_job": self.is_array_job,
            "is_multinode": self.is_multinode,
            "is_main_node": self.is_main_node,
            "is_fargate": self.is_fargate,
            "is_aws_environment": self.is_aws_environment,
            "has_container_credentials": self.has_container_credentials,
        }
