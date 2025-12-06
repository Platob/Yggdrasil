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