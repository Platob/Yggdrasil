"""AI insight endpoints — file analysis + market brief via Anthropic."""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["ai"])


class AnalyzeRequest(BaseModel):
    path: str
    query: str = "Summarize this dataset and highlight key patterns."


class InsightRequest(BaseModel):
    context: str


@router.post("/analyze")
async def analyze_file(request: Request, body: AnalyzeRequest):
    svc = request.app.state.ai_insight_service
    return await svc.analyze_file(body.path, body.query)


@router.post("/insight")
async def generate_insight(request: Request, body: InsightRequest):
    svc = request.app.state.ai_insight_service
    return await svc.generate_insight(body.context)
