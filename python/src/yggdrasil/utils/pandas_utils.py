from .fake_module import make_fake_module
from .numpy_utils import *

for mod_name in [
    "pandas",
]:
    make_fake_module(mod_name)

import pandas as pd


__all__ = [
    "pandas",
    "PandasDataFrame",
    "PandasSeries"
]

pandas = pd
PandasDataFrame = pd.DataFrame
PandasSeries = pd.Series
