import logging
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import BASE_DIR, RANKING_TOP_N
from core.akshare_client import AkShareClient
from core.news_fetcher import NewsFetcher
from analysis.stock_analyzer import StockAnalyzer
from analysis.llm_analyzer import LLMAnalyzer
from ranking.scheduler import inflow_ranking, outflow_ranking, last_update
from backtest.engine import BacktestEngine
from backtest.strategies import STRATEGIES

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

_ak = AkShareClient()
_news = NewsFetcher()
_llm = LLMAnalyzer()
_analyzer = StockAnalyzer(client=_ak, news_fetcher=_news, llm=_llm)
_backtest = BacktestEngine(client=_ak)


# ── 页面路由 ──────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request, code: str = ""):
    if not code:
        return templates.TemplateResponse("index.html", {"request": request, "error": "请输入股票代码"})

    report = await _analyzer.analyze(code)

    kline_data = None
    try:
        df = _ak.get_daily_kline(code)
        if not df.empty:
            df = df.tail(60)
            kline_data = {
                "dates": df["日期"].astype(str).tolist(),
                "closes": df["收盘"].astype(float).round(2).tolist(),
            }
    except Exception as e:
        logger.warning("K-line fetch failed for %s: %s", code, e)

    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "report": report,
        "kline": kline_data,
    })


@router.get("/ranking", response_class=HTMLResponse)
async def ranking_page(request: Request):
    return templates.TemplateResponse("ranking.html", {
        "request": request,
        "inflow": inflow_ranking,
        "outflow": outflow_ranking,
        "last_update": last_update,
    })


@router.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {
        "request": request,
        "strategies": STRATEGIES,
    })


# ── API 路由 ───────────────────────────────────

@router.get("/api/analyze/{code}")
async def api_analyze(code: str):
    return await _analyzer.analyze(code)


@router.get("/api/ranking/inflow")
async def api_ranking_inflow():
    return {"data": inflow_ranking, "updated": str(last_update)}


@router.get("/api/ranking/outflow")
async def api_ranking_outflow():
    return {"data": outflow_ranking, "updated": str(last_update)}


class BacktestRequest(BaseModel):
    code: str
    strategy: str = "ma_cross"
    start: str = "2024-01-01"
    end: str = "2025-12-31"
    cash: float = 100_000
    fast: int = 5
    slow: int = 20
    lookback: int = 20
    holding: int = 10
    period: int = 20
    threshold: float = 0.05


@router.post("/api/backtest/run")
async def api_backtest_run(req: BacktestRequest):
    info = STRATEGIES.get(req.strategy)
    if not info:
        raise HTTPException(status_code=400, detail=f"未知策略: {req.strategy}")

    kwargs = {}
    for p in info["params"]:
        kwargs[p] = getattr(req, p, None)

    try:
        return _backtest.run(
            code=req.code,
            strategy_cls=info["cls"],
            start=req.start,
            end=req.end,
            cash=req.cash,
            **kwargs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
