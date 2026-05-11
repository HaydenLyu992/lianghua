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
from ranking.scheduler import inflow_ranking, outflow_ranking, northbound_ranking, last_update
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
        "northbound": northbound_ranking,
        "last_update": last_update,
        # Initial table render variables
        "rows": inflow_ranking,
        "tab": "inflow",
        "col_label": "主力净流入（万元）",
        "money_class": "money-in",
    })


@router.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {
        "request": request,
        "strategies": STRATEGIES,
    })


# ── API 路由 ───────────────────────────────────

@router.get("/api/suggest")
async def api_suggest(q: str = ""):
    """搜索建议：模糊匹配股票代码/名称"""
    if len(q) < 1:
        return ""
    try:
        df = _ak.get_spot_df()
        if df.empty:
            return "<div class='suggest-empty'>暂无数据</div>"
    except Exception:
        return "<div class='suggest-empty'>搜索服务暂不可用</div>"

    q_lower = q.strip().lower()
    matches = df[
        df["代码"].astype(str).str.contains(q_lower, na=False)
        | df["名称"].astype(str).str.contains(q_lower, na=False)
    ].head(8)

    if matches.empty:
        return "<div class='suggest-empty'>无匹配结果</div>"

    items = "".join(
        f"<div class='suggest-item' data-code='{r['代码']}'>{r['代码']} — {r['名称']}</div>"
        for _, r in matches.iterrows()
    )
    return items

@router.get("/api/analyze/{code}")
async def api_analyze(code: str):
    return await _analyzer.analyze(code)


@router.get("/api/ranking/inflow")
async def api_ranking_inflow():
    return {"data": inflow_ranking, "updated": str(last_update)}


@router.get("/api/ranking/outflow")
async def api_ranking_outflow():
    return {"data": outflow_ranking, "updated": str(last_update)}


@router.get("/api/ranking/northbound")
async def api_ranking_northbound():
    return {"data": northbound_ranking, "updated": str(last_update)}


@router.get("/api/ranking/inflow-html", response_class=HTMLResponse)
async def ranking_inflow_html(request: Request, q: str = ""):
    """HTMX 局部刷新：主力净流入表格"""
    data = _filter_ranking(inflow_ranking, q)
    return templates.TemplateResponse("_ranking_table.html", {
        "request": request,
        "rows": data,
        "tab": "inflow",
        "col_label": "主力净流入（万元）",
        "money_class": "money-in",
    })


@router.get("/api/ranking/outflow-html", response_class=HTMLResponse)
async def ranking_outflow_html(request: Request, q: str = ""):
    data = _filter_ranking(outflow_ranking, q)
    return templates.TemplateResponse("_ranking_table.html", {
        "request": request,
        "rows": data,
        "tab": "outflow",
        "col_label": "主力净流出（万元）",
        "money_class": "money-out",
    })


@router.get("/api/ranking/northbound-html", response_class=HTMLResponse)
async def ranking_northbound_html(request: Request, q: str = ""):
    data = _filter_ranking(northbound_ranking, q)
    return templates.TemplateResponse("_ranking_table.html", {
        "request": request,
        "rows": data,
        "tab": "northbound",
        "col_label": "北向净买入（万元）",
        "money_class": "money-in",
    })


def _filter_ranking(data: list[dict], q: str) -> list[dict]:
    if not q:
        return data
    q = q.strip()
    return [r for r in data if q in r.get("code", "") or q in r.get("name", "")]


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
    oversold: int = 30
    overbought: int = 70
    devfactor: float = 2.0


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


@router.get("/api/news/{code}")
async def api_news(code: str):
    news = _news.fetch_stock_news(code, limit=30)
    return {"code": code, "news": news}


@router.get("/api/history/{code}")
async def api_history(code: str):
    """查询最近20条分析历史记录"""
    from core.database import AsyncSession, AnalysisHistory
    from sqlalchemy import select
    try:
        async with AsyncSession() as session:
            stmt = (
                select(AnalysisHistory)
                .where(AnalysisHistory.stock_code == code)
                .order_by(AnalysisHistory.created_at.desc())
                .limit(20)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return {
                "code": code,
                "history": [
                    {
                        "score": r.score_total,
                        "signal": r.signal,
                        "created_at": r.created_at.isoformat(),
                    }
                    for r in rows
                ],
            }
    except Exception as e:
        logger.warning("Failed to fetch history for %s: %s", code, e)
        return {"code": code, "history": [], "error": str(e)}


@router.get("/api/limit/{code}")
async def api_limit(code: str):
    from analysis.limit_analyzer import LimitAnalyzer
    la = LimitAnalyzer(_ak)
    return await la.analyze(code)
