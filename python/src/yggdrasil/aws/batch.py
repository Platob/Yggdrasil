"""AWS Batch (and ECS/Fargate) runtime introspection.

When a job runs under AWS Batch, the agent injects a well-known set of
environment variables into the container — the job id, attempt, queue, compute
environment, and (for array / multi-node jobs) the index/topology. This module
captures them into one structured, picklable :class:`AWSBatchEnvironment` so
code can branch on "am I in Batch?", find its array index, point at the job in
the Console, or just log the full runtime context.

Reach it from a client as ``AWSClient.current().batch`` (no network, no
credentials — it's pure ``os.environ``), or directly via
``AWSBatchEnvironment.current()``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

from yggdrasil.aws.console import batch_job_url
from yggdrasil.url import URL

__all__ = ["AWSBatchEnvironment"]


def _int(env: Mapping[str, str], name: str) -> Optional[int]:
    raw = env.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _str(env: Mapping[str, str], name: str) -> Optional[str]:
    v = env.get(name)
    return v or None


@dataclass(frozen=True)
class AWSBatchEnvironment:
    """The AWS Batch runtime context, read from the job container's env.

    Every field is ``None`` off-Batch, so :attr:`is_batch` is the gate before
    trusting the rest. Multi-node fields (``node_*`` / ``num_nodes``) are only
    set for multi-node parallel jobs; ``array_index`` only for array jobs.
    """

    job_id: Optional[str] = None
    job_attempt: Optional[int] = None
    job_queue: Optional[str] = None
    compute_environment: Optional[str] = None
    array_index: Optional[int] = None
    node_index: Optional[int] = None
    main_node_index: Optional[int] = None
    num_nodes: Optional[int] = None
    main_node_ip: Optional[str] = None
    execution_env: Optional[str] = None
    region: Optional[str] = None
    container_credentials_uri: Optional[str] = None
    #: The raw captured ``AWS_BATCH_*`` / runtime vars — handy for logging.
    raw: Mapping[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "AWSBatchEnvironment":
        e = os.environ if env is None else env
        captured = {
            k: e[k] for k in (
                "AWS_BATCH_JOB_ID", "AWS_BATCH_JOB_ATTEMPT", "AWS_BATCH_JQ_NAME",
                "AWS_BATCH_CE_NAME", "AWS_BATCH_JOB_ARRAY_INDEX",
                "AWS_BATCH_JOB_NODE_INDEX", "AWS_BATCH_JOB_MAIN_NODE_INDEX",
                "AWS_BATCH_JOB_NUM_NODES", "AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS",
                "AWS_EXECUTION_ENV", "AWS_REGION", "AWS_DEFAULT_REGION",
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            ) if e.get(k)
        }
        return cls(
            job_id=_str(e, "AWS_BATCH_JOB_ID"),
            job_attempt=_int(e, "AWS_BATCH_JOB_ATTEMPT"),
            job_queue=_str(e, "AWS_BATCH_JQ_NAME"),
            compute_environment=_str(e, "AWS_BATCH_CE_NAME"),
            array_index=_int(e, "AWS_BATCH_JOB_ARRAY_INDEX"),
            node_index=_int(e, "AWS_BATCH_JOB_NODE_INDEX"),
            main_node_index=_int(e, "AWS_BATCH_JOB_MAIN_NODE_INDEX"),
            num_nodes=_int(e, "AWS_BATCH_JOB_NUM_NODES"),
            main_node_ip=_str(e, "AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS"),
            execution_env=_str(e, "AWS_EXECUTION_ENV"),
            region=_str(e, "AWS_REGION") or _str(e, "AWS_DEFAULT_REGION"),
            container_credentials_uri=(
                _str(e, "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
                or _str(e, "AWS_CONTAINER_CREDENTIALS_FULL_URI")
            ),
            raw=captured,
        )

    #: ``current()`` reads the live process environment.
    current = from_env

    # ------------------------------------------------------------------
    @property
    def is_batch(self) -> bool:
        return self.job_id is not None

    @property
    def is_array_job(self) -> bool:
        return self.array_index is not None

    @property
    def is_multinode(self) -> bool:
        return bool(self.num_nodes and self.num_nodes > 1)

    @property
    def is_main_node(self) -> bool:
        """True for a single-node job, or the main node of a multi-node job."""
        if not self.is_multinode:
            return True
        return self.node_index is not None and self.node_index == self.main_node_index

    @property
    def is_fargate(self) -> bool:
        return bool(self.execution_env and "FARGATE" in self.execution_env.upper())

    @property
    def has_container_credentials(self) -> bool:
        """Whether the ECS/Batch task-role credential endpoint is wired up
        (boto3 picks these up automatically — this just reports it)."""
        return self.container_credentials_uri is not None

    @property
    def explore_url(self) -> Optional[URL]:
        """Console deep-link to this Batch job, or ``None`` off-Batch."""
        if not self.job_id:
            return None
        return batch_job_url(self.job_id, self.region)

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
            "has_container_credentials": self.has_container_credentials,
        }

    def _repr_html_(self) -> Optional[str]:
        url = self.explore_url
        if url is None:
            return None
        return f'<a href="{url}" target="_blank">AWS Batch job {self.job_id}</a>'
