from .fake_module import *

for mod_name in [
    "pyspark.sql.types",
    "pandas",
    "polars"
]:
    make_fake_module(mod_name)