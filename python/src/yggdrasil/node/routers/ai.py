from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..deps import get_ai_service
from ..schemas.ai import (
    AIAnalyzeResponse,
    AIChatRequest,
    AIChatResponse,
    AIConversationCreate,
    AIConversationListResponse,
    AIConversationResponse,
    AIPortfolioAnalysis,
    AISuggestionsResponse,
)
from ..services.ai import AIService

router = APIRouter(tags=["ai"])


@router.post("/chat", response_model=AIChatResponse)
async def chat(
    req: AIChatRequest,
    service: AIService = Depends(get_ai_service),
) -> AIChatResponse | StreamingResponse:
    if req.stream:
        async def _gen():
            async for chunk in service.stream_chat(req):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: {\"done\": true}\n\n"
        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await service.chat(req)


# -- conversations --------------------------------------------------------

@router.get("/conversations", response_model=AIConversationListResponse)
async def list_conversations(
    service: AIService = Depends(get_ai_service),
) -> AIConversationListResponse:
    return AIConversationListResponse(conversations=await service.list_conversations())


@router.post("/conversations", response_model=AIConversationResponse)
async def create_conversation(
    req: AIConversationCreate,
    service: AIService = Depends(get_ai_service),
) -> AIConversationResponse:
    return AIConversationResponse(conversation=await service.create_conversation(req))


@router.get("/conversations/{conv_id}", response_model=AIConversationResponse)
async def get_conversation(
    conv_id: int,
    service: AIService = Depends(get_ai_service),
) -> AIConversationResponse:
    return AIConversationResponse(conversation=await service.get_conversation(conv_id))


@router.delete("/conversations/{conv_id}", response_model=AIConversationResponse)
async def delete_conversation(
    conv_id: int,
    service: AIService = Depends(get_ai_service),
) -> AIConversationResponse:
    return AIConversationResponse(conversation=await service.delete_conversation(conv_id))


# -- analysis -------------------------------------------------------------

@router.post("/analyze/{symbol}", response_model=AIAnalyzeResponse)
async def analyze_symbol(
    symbol: str,
    timeframe: str = "1d",
    include_portfolio: bool = False,
    service: AIService = Depends(get_ai_service),
) -> AIAnalyzeResponse:
    return await service.analyze_symbol(symbol, timeframe=timeframe,
                                        include_portfolio=include_portfolio)


@router.post("/analyze/portfolio", response_model=AIPortfolioAnalysis)
async def analyze_portfolio(
    service: AIService = Depends(get_ai_service),
) -> AIPortfolioAnalysis:
    return await service.analyze_portfolio()


# -- suggestions ----------------------------------------------------------

@router.get("/suggestions", response_model=AISuggestionsResponse)
async def get_suggestions(
    service: AIService = Depends(get_ai_service),
) -> AISuggestionsResponse:
    return AISuggestionsResponse(suggestions=await service.suggestions())
