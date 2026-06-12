from __future__ import annotations
import os

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
DATABRICKS_HOST: str = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN: str = os.getenv("DATABRICKS_TOKEN", "")

# Cache TTLs (seconds)
QUOTE_TTL: float = float(os.getenv("QUOTE_TTL", "15"))
OHLCV_TTL: float = float(os.getenv("OHLCV_TTL", "60"))

# Default paper-trading portfolio cash
INITIAL_CASH: float = float(os.getenv("INITIAL_CASH", "100000"))
