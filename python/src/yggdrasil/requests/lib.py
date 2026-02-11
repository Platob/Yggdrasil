try:
    import requests
except ImportError:
    from ..pyutils.pyenv import PyEnv

    requests = PyEnv.runtime_import_module(module_name="requests", pip_name="requests", install=True)

__all__ = [
    "requests"
]
