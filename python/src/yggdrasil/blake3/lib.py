try:
    import blake3
except ImportError:
    from yggdrasil.environ import PyEnv

    blake3 = PyEnv.runtime_import_module(module_name="blake3", pip_name="blake3", install=True)


from blake3 import * # type: ignore
