from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}

@router.get("/historical_data", tags=["health"])
async def health() -> dict:
    """Liveness probe."""

    return {"status": "ok"}

@router.get("/historical_data_totsa", tags=["health"])
async def health() -> dict:
    """Liveness probe."""

    return {"status": "ok"}