from .ai import router as ai_router
from .call import router as call_router
from .cmd import router as cmd_router
from .dag import router as dag_router
from .discovery import router as discovery_router
from .env import router as env_router
from .environment import router as environment_router
from .filesystem import router as filesystem_router
from .function import router as function_router
from .job import router as job_router
from .messenger import router as messenger_router
from .monitor import router as monitor_router
from .python import router as python_router
from .run import router as run_router
from .trading import router as trading_router

__all__ = [
    "ai_router",
    "call_router",
    "cmd_router",
    "dag_router",
    "discovery_router",
    "env_router",
    "environment_router",
    "filesystem_router",
    "function_router",
    "job_router",
    "messenger_router",
    "monitor_router",
    "python_router",
    "run_router",
    "trading_router",
]
