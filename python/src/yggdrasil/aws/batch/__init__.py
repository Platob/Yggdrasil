"""AWS Batch runtime resource + service (folder module: resource in
``batch.py``, service in ``service.py``)."""
from __future__ import annotations

from .batch import AWSBatch, in_aws_environment
from .service import BatchService

__all__ = ["AWSBatch", "BatchService", "in_aws_environment"]
