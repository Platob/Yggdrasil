try:
    import requests
except ImportError:
    from ..environ import PyEnv

    requests = PyEnv.runtime_import_module(module_name="requests", pip_name="requests", install=True)

__all__ = [
    "requests"
]
