from .fake_module import make_fake_module

for mod_name in [
    "numpy",
]:
    make_fake_module(mod_name)

import numpy as np


__all__ = [
    "numpy",
]

numpy = np

