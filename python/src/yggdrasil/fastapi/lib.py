try:
    import fastapi
except ImportError:
    from yggdrasil.pyutils.pyenv import PyEnv

    fastapi = PyEnv.runtime_import_module(module_name="fastapi", pip_name="fastapi", install=True)


__all__ = [
    "fastapi"
]
