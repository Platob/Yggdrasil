try:
    import pandas
except ImportError:
    from ..pyutils.pyenv import PyEnv

    pandas = PyEnv.runtime_import_module(module_name="pandas", pip_name="pandas", install=True)


__all__ = [
    "pandas"
]
