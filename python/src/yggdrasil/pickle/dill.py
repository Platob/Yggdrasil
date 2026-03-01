try:
    import dill  # type: ignore[import]
except ImportError:
    from yggdrasil.environ import runtime_import_module  # type: ignore[import]

    dill = runtime_import_module(module_name="dill", pip_name="dill", install=True)

from dill import * # type: ignore[import]