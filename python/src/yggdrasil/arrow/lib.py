try:
    import pyarrow
except ImportError:
    from yggdrasil.environ import PyEnv

    pyarrow = PyEnv.runtime_import_module(module_name="pyarrow", pip_name="pyarrow", install=True)


__all__ = [
    "pyarrow"
]