from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_ai_service
from ..schemas.ai import (
    CodeAnalysisRequest,
    CodeAnalysisResponse,
    RunAnalysisRequest,
    RunAnalysisResponse,
)
from ..services.ai import AIService

router = APIRouter(tags=["ai"])


@router.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(
    req: CodeAnalysisRequest,
    service: AIService = Depends(get_ai_service),
) -> CodeAnalysisResponse:
    return service.analyze_code(req)


@router.post("/run/analyze", response_model=RunAnalysisResponse)
async def analyze_run(
    req: RunAnalysisRequest,
    service: AIService = Depends(get_ai_service),
) -> RunAnalysisResponse:
    return service.analyze_run(req)
