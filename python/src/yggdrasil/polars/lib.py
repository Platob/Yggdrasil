try:
    import polars as polars
except ImportError:
    from ..environ import PyEnv

    polars = PyEnv.runtime_import_module(module_name="polars", pip_name="polars", install=True)


__all__ = [
    "polars"
]
