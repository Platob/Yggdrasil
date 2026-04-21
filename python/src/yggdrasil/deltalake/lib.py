"""Optional-dependency guard for the ``deltalake`` (delta-rs) package.

``deltalake`` is the Python binding of the Rust reference reader/writer
for the Delta Lake transaction log. We use it as the canonical engine
inside :class:`yggdrasil.io.buffer.delta_io.DeltaIO` — the wheel ships
a fully-featured protocol parser (deletion vectors, V2 checkpoints,
column mapping, time travel, CDF) so we don't re-implement any of that
in Python.

Keep imports of ``deltalake`` funneled through this module so that a
base install of ``ygg`` keeps working without it and callers get a
helpful "install extra" error at first use.
"""
try:
    import deltalake
except ImportError:
    from yggdrasil.environ import PyEnv

    deltalake = PyEnv.runtime_import_module(
        module_name="deltalake",
        pip_name="deltalake",
        install=True,
    )


from deltalake import *  # type: ignore # noqa: F401,F403
