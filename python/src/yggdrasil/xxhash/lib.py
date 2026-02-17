try:
    import xxhash
except ImportError:  # keep runtime non-blocking
    from ..pyutils.pyenv import PyEnv

    xxhash = PyEnv.runtime_import_module(
        module_name="xxhash", pip_name="xxhash", install=True
    )


__all__ = [
    "xxhash"
]