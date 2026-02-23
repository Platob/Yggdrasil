from fastapi import APIRouter

from app.api.v1.endpoints import health

# Add more endpoint modules here as the project grows
# from app.api.v1.endpoints import users, items, ...

router = APIRouter()

router.include_router(health.router)
