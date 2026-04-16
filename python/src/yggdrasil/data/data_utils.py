try:
    from yggdrasil.data.cast.options import CastOptions
except:
    CastOptions = None


__all__ = [
    "get_cast_options_class"
]


def get_cast_options_class():
    global CastOptions
    if CastOptions is None:
        from yggdrasil.data.cast.options import CastOptions
        CastOptions = CastOptions
    return CastOptions
