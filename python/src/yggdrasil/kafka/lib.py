"""Optional-dependency guard for :mod:`yggdrasil.kafka`.

Kafka is an optional integration. The runtime client is provided by
``confluent-kafka`` (the librdkafka-backed wrapper); installing
``ygg[kafka]`` pulls it in. Importing this module triggers a single
guarded import — base installs that never touch :mod:`yggdrasil.kafka`
keep working with no extra dependency.
"""

from __future__ import annotations

from types import ModuleType

try:
    import confluent_kafka as _confluent_kafka
except ImportError:  # pragma: no cover - guarded import
    from yggdrasil.environ import runtime_import_module

    _confluent_kafka = runtime_import_module(
        module_name="confluent_kafka",
        pip_name="confluent-kafka",
        install=True,
    )


confluent_kafka: ModuleType = _confluent_kafka

__all__ = ["confluent_kafka"]
