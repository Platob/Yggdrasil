import os
import pyarrow

if os.name == "nt":
    try:
        pyarrow.util.download_tzdata_on_windows()
    except:
        pass

__all__ = []
