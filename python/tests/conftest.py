import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "python" / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


LOCAL_JHOME = os.path.expanduser(r"~\Downloads\zulu17.62.17-ca-jdk17.0.17-win_x64\zulu17.62.17-ca-jdk17.0.17-win_x64\b")

if os.path.exists(LOCAL_JHOME):
    os.environ["JAVA_HOME"] = LOCAL_JHOME

logger = logging.getLogger("yggdrasil")

def setup_logging(level: int = logging.INFO) -> None:
    # Avoid duplicate handlers if this gets called twice (common in notebooks/tests)
    if logger.handlers:
        return

    logger.setLevel(level)
    logger.propagate = False  # keep it from double-logging via root

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    )
    handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(handler)

# --- usage ---
setup_logging(logging.DEBUG)

os.environ["DATABRICKS_HOST"] = "dbc-e646c5f9-8a44.cloud.databricks.com"