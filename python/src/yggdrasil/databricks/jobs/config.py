"""Widget-backed configuration helpers — alias module.

``NotebookConfig`` historically lived here as a dataclass base whose
subclasses bound Databricks widgets to typed fields. The canonical
surface now lives in :mod:`yggdrasil.environ.parameters` as
:class:`~yggdrasil.environ.parameters.SystemParameters` — a lazy
``Mapping[str, Any]`` that auto-fetches from ``sys.argv``, Databricks
notebook bindings (``dbutils.widgets`` + ``{{job.parameters.*}}``), and
``os.environ``, with type-aware casting through
:func:`yggdrasil.data.cast.convert`.

``NotebookConfig`` is kept as an alias so existing call sites keep
working — drop ``@dataclass`` from subclasses if you still carry it
(``SystemParameters`` is intentionally not a dataclass to stay
flexible). The ``from_environment()`` / ``init_widgets()`` /
``init_job()`` entry points live on :class:`SystemParameters` itself.
"""
from yggdrasil.environ.parameters import (
    ALL_VALUES_TAG,
    SystemParameters,
    WidgetType,
)

#: Backwards-compat alias — :class:`yggdrasil.environ.parameters.SystemParameters`.
NotebookConfig = SystemParameters

__all__ = [
    "ALL_VALUES_TAG",
    "NotebookConfig",
    "SystemParameters",
    "WidgetType",
]
