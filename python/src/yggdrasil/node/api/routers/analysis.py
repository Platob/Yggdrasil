from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..deps import get_analysis_service, get_network_service
from ..schemas.analysis import (
    AggregateRequest,
    AggregateResult,
    AiSummaryRequest,
    AiSummaryResult,
    CompareRequest,
    CompareResult,
    DescribeResult,
    ExportRequest,
    FinanceRequest,
    FinanceResult,
    ForecastRequest,
    ForecastResult,
    IndicatorsRequest,
    IndicatorsResult,
    OhlcRequest,
    OhlcResult,
    PivotRequest,
    PivotResult,
    SeriesRequest,
    SeriesResult,
)
from ..services.analysis import AnalysisService
from ..services.network import NetworkService

router = APIRouter(tags=["analysis"])

_MEDIA = {
    "csv": "text/csv", "parquet": "application/vnd.apache.parquet",
    "json": "application/json", "ndjson": "application/x-ndjson",
    "arrow": "application/vnd.apache.arrow.stream", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@router.post("/aggregate", response_model=AggregateResult)
async def aggregate(
    req: AggregateRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> AggregateResult:
    """Group-by / pivot aggregation (sum/mean/min/max/count/median/std/var)."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/aggregate", json_body=req.model_dump())
    return await service.aggregate(req)


@router.post("/pivot", response_model=PivotResult)
async def pivot(
    req: PivotRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> PivotResult:
    """Excel-style pivot: rows × columns (cross-tab) × measures. Streams the
    group-by (bounded memory), shapes a wide table, top-N caps columns."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/pivot", json_body=req.model_dump())
    return await service.pivot(req)


@router.get("/describe", response_model=DescribeResult)
async def describe(
    path: str,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> DescribeResult:
    """Per-column summary statistics (count/mean/std/min/quartiles/max)."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "GET", "/api/v2/analysis/describe", params={"path": path})
    return await service.describe(path)


@router.post("/finance", response_model=FinanceResult)
async def finance(
    req: FinanceRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> FinanceResult:
    """Returns, cumulative return, rolling mean, and rolling volatility of a
    numeric series (optionally ordered by a column)."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/finance", json_body=req.model_dump())
    return await service.finance(req)


@router.post("/series", response_model=SeriesResult)
async def series(
    req: SeriesRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> SeriesResult:
    """Adaptive downsample of a numeric series to ~`points` buckets (mean +
    min/max envelope), with an optional x zoom window pushed into the scan."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/series", json_body=req.model_dump())
    return await service.series(req)


@router.post("/forecast", response_model=ForecastResult)
async def forecast(
    req: ForecastRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> ForecastResult:
    """Forecast a value column over time (optionally per group key): fits
    xgboost → gbr → ridge over trend/lag/seasonal features and projects
    ``horizon`` steps with a confidence band. Runs where the data lives."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/forecast", json_body=req.model_dump())
    return await service.forecast(req)


@router.post("/ohlc", response_model=OhlcResult)
async def ohlc(
    req: OhlcRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> OhlcResult:
    """Resample a price series into `buckets` open/high/low/close bars
    (+ summed volume) for candlestick plotting."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/ohlc", json_body=req.model_dump())
    return await service.ohlc(req)


@router.post("/indicators", response_model=IndicatorsResult)
async def indicators(
    req: IndicatorsRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> IndicatorsResult:
    """RSI, MACD, Bollinger Bands, and ATR for a price series."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/indicators", json_body=req.model_dump())
    return await service.indicators(req)


@router.post("/compare", response_model=CompareResult)
async def compare(
    req: CompareRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> CompareResult:
    """Normalize and compare multiple series; compute cross-series correlation."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/compare", json_body=req.model_dump())
    return await service.compare(req)


@router.post("/ai_summary", response_model=AiSummaryResult)
async def ai_summary(
    req: AiSummaryRequest,
    service: AnalysisService = Depends(get_analysis_service),
) -> AiSummaryResult:
    """Claude-powered natural-language analysis (requires ANTHROPIC_API_KEY)."""
    return await service.ai_summary(req)


@router.post("/export")
async def export(
    req: ExportRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> StreamingResponse:
    """Apply the transform (filters + casts incl. tz→UTC + projection) and
    download the result in any tabular media type (csv/parquet/json/ndjson/
    arrow/xlsx)."""
    if node and node != service.settings.node_id:
        return StreamingResponse(
            network.proxy_post_stream(node, "/api/v2/analysis/export", req.model_dump()),
            media_type=_MEDIA.get(req.fmt, "application/octet-stream"),
        )
    tmp_path, name = await service.export(req)

    async def _stream():
        try:
            with open(tmp_path, "rb") as fh:
                while True:
                    chunk = fh.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk
        finally:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(
        _stream(), media_type=_MEDIA.get(req.fmt, "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
