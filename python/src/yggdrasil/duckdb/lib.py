try:
    import duckdb
except ImportError:
    from ..environ import PyEnv

    duckdb = PyEnv.runtime_import_module(module_name="duckdb", pip_name="duckdb", install=True)


__all__ = [
    "duckdb"
]