try:
    import mongoengine
except ImportError:
    from yggdrasil.environ import PyEnv

    mongoengine = PyEnv.runtime_import_module(
        module_name="mongoengine", pip_name="mongoengine", install=True
    )

from mongoengine import * # type: ignore

__all__ = [
    "mongoengine"
] + mongoengine.__all__
