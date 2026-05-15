import asyncio
import logging
import uuid
from datetime import datetime
import pandas as pd
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import BASE_DIR, RANKING_TOP_N
from core.akshare_client import AkShareClient
from core.news_fetcher import NewsFetcher
from analysis.stock_analyzer import StockAnalyzer
from analysis.llm_analyzer import LLMAnalyzer
from analysis.progress import ProgressTracker
from ranking.scheduler import inflow_ranking, surge_ranking, rising_trend, falling_trend, last_update, refresh_rankings
from backtest.engine import BacktestEngine
from backtest.strategies import STRATEGIES
from sector.sector_analyzer import SectorAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

_ak = AkShareClient()
_news = NewsFetcher()
_llm = LLMAnalyzer()
_analyzer = StockAnalyzer(client=_ak, news_fetcher=_news, llm=_llm)
_backtest = BacktestEngine(client=_ak)
_sector = SectorAnalyzer()

_trackers: dict[str, ProgressTracker] = {}


# ── 页面路由 ──────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _resolve_code(user_input: str) -> str:
    """将用户输入解析为标准股票代码。中文名或含前缀代码均可识别。"""
    inp = user_input.strip()
    if not inp:
        return ""
    # 已经是纯数字代码
    cleaned = inp.replace("sz", "").replace("sh", "").replace("bj", "").replace("SZ", "").replace("SH", "").replace("BJ", "").strip()
    if cleaned.isdigit():
        return cleaned
    # 可能是中文名，查股票列表
    try:
        df = _ak.get_stock_list()
        if not df.empty:
            mask = df["名称"].astype(str).str.contains(inp, na=False)
            match = df[mask]
            if not match.empty:
                return str(match.iloc[0]["代码"])
    except Exception:
        pass
    return inp

@router.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request, code: str = ""):
    if not code:
        return templates.TemplateResponse("index.html", {"request": request, "error": "请输入股票代码"})

    # 如果是纯数字代码，同步解析（瞬时完成）；中文名则交给后台解析
    raw_input = code.strip()
    is_digit = raw_input.replace("sz", "").replace("sh", "").replace("bj", "").replace("SZ", "").replace("SH", "").replace("BJ", "").strip().isdigit()
    resolved = raw_input if is_digit else raw_input  # 中文名先原样传入，后台解析

    tracker = ProgressTracker(code=resolved)
    tracker_id = str(uuid.uuid4())[:8]
    _trackers[tracker_id] = tracker

    async def _run_analysis():
        try:
            nonlocal resolved
            if not is_digit:
                resolved = _resolve_code(raw_input)
                if not resolved.isdigit():
                    tracker.mark_error(f"未找到「{raw_input}」对应的股票，请尝试输入代码或更准确的名称")
                    return
                tracker.code = resolved
            report = await asyncio.wait_for(
                _analyzer.analyze(resolved, tracker=tracker), timeout=300
            )
            tracker.mark_complete(report)
        except asyncio.TimeoutError:
            logger.warning("Analysis timeout for %s", resolved)
            tracker.mark_error(f"分析「{raw_input}」超时（300秒），可能数据源响应慢，请稍后重试")
        except Exception as e:
            logger.error("Analysis failed for %s: %s", resolved, e)
            tracker.mark_error(f"分析出错: {e}")

    asyncio.create_task(_run_analysis())

    return templates.TemplateResponse("analysis_progress.html", {
        "request": request,
        "tracker_id": tracker_id,
        "code": resolved,
    })


@router.get("/api/analyze/progress/{tracker_id}")
async def api_analysis_progress(tracker_id: str):
    tracker = _trackers.get(tracker_id)
    if not tracker:
        return {"error": "not_found"}
    return tracker.to_dict()


@router.get("/api/analyze/progress-html/{tracker_id}", response_class=HTMLResponse)
async def api_analysis_progress_html(request: Request, tracker_id: str):
    tracker = _trackers.get(tracker_id)
    if not tracker:
        return HTMLResponse("<div>not found</div>")
    data = tracker.to_dict()
    return templates.TemplateResponse("_progress_panel.html", {
        "request": request,
        **data,
    })


@router.get("/analyze/result/{tracker_id}", response_class=HTMLResponse)
async def analysis_result(request: Request, tracker_id: str):
    tracker = _trackers.get(tracker_id)
    if not tracker or not tracker.result:
        return RedirectResponse("/")

    report = tracker.result
    resolved = tracker.code

    kline_data = None
    try:
        df = await asyncio.to_thread(_ak.get_daily_kline, resolved)
        if not df.empty:
            date_col = next((c for c in df.columns if c in ("date", "日期", "时间")), df.columns[0])
            close_col = next((c for c in df.columns if c in ("close", "收盘")), df.columns[3])
            df = df.tail(60)
            kline_data = {
                "dates": df[date_col].astype(str).tolist(),
                "closes": pd.to_numeric(df[close_col], errors="coerce").round(2).tolist(),
            }
    except Exception as e:
        logger.warning("K-line fetch failed for %s: %s", resolved, e)

    # cleanup tracker
    _trackers.pop(tracker_id, None)

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
        "surge": surge_ranking,
        "last_update": last_update["value"],
        "rows": inflow_ranking,
        "tab": "inflow",
        "col_label": "主力净流入（万元）",
        "value_class": "money-in",
    })


@router.get("/sector", response_class=HTMLResponse)
async def sector_page(request: Request):
    return templates.TemplateResponse("sector.html", {
        "request": request,
    })


@router.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {
        "request": request,
        "strategies": STRATEGIES,
    })


# ── API 路由 ───────────────────────────────────

@router.get("/api/suggest")
async def api_suggest(code: str = "", q: str = ""):
    """搜索建议：模糊匹配股票代码/名称"""
    keyword = (code or q).strip()
    if len(keyword) < 1:
        return HTMLResponse("")
    try:
        df = _ak.get_stock_list()
        if df.empty:
            return HTMLResponse("<div class='suggest-empty'>数据源暂不可用，请直接输入完整代码搜索</div>")
    except Exception:
        return HTMLResponse("<div class='suggest-empty'>搜索服务暂不可用</div>")

    kw = keyword.lower()
    code_col = df["代码"].astype(str)
    name_col = df["名称"].astype(str)

    # 纯数字 → 优先代码前缀匹配
    if kw.isdigit():
        prefix = code_col[code_col.str.startswith(kw)]
        fuzzy = code_col[code_col.str.contains(kw, na=False) & ~code_col.str.startswith(kw)]
        name_match = name_col[name_col.str.contains(kw, na=False)]
        by_code = pd.concat([df.loc[prefix.index], df.loc[fuzzy.index]], ignore_index=True)
        by_name = df.loc[name_match.index]
        matches = pd.concat([by_code, by_name], ignore_index=True).head(8)
    else:
        mask = name_col.str.contains(kw, na=False)
        matches = df[mask].head(8)

    if matches.empty:
        return HTMLResponse("<div class='suggest-spinner'><span class='spinner-dot'></span> 匹配中...</div>")

    items = "".join(
        f"<div class='suggest-item' data-code='{r['代码']}'>{r['代码']} — {r['名称']}</div>"
        for _, r in matches.iterrows()
    )
    return HTMLResponse(items)

@router.get("/api/analyze/{code}")
async def api_analyze(code: str):
    return await _analyzer.analyze(code)


@router.get("/api/ranking/inflow")
async def api_ranking_inflow():
    return {"data": inflow_ranking, "updated": str(last_update["value"])}


@router.get("/api/ranking/surge")
async def api_ranking_surge():
    return {"data": surge_ranking, "updated": str(last_update["value"])}


@router.get("/api/ranking/refresh-inflow-html", response_class=HTMLResponse)
async def ranking_refresh_inflow_html(request: Request):
    """手动刷新：先拉取最新数据，再返回资金流入表格 HTML"""
    await refresh_rankings(force=True)
    return templates.TemplateResponse("_ranking_table.html", {
        "request": request,
        "rows": inflow_ranking,
        "tab": "inflow",
        "col_label": "主力净流入（万元）",
        "value_class": "money-in",
    })


@router.get("/api/ranking/refresh-surge-html", response_class=HTMLResponse)
async def ranking_refresh_surge_html(request: Request):
    """手动刷新：先拉取最新数据，再返回飙升榜表格 HTML"""
    await refresh_rankings(force=True)
    return templates.TemplateResponse("_ranking_table.html", {
        "request": request,
        "rows": surge_ranking,
        "tab": "surge",
        "col_label": "涨跌幅",
        "value_class": "money-in",
    })


@router.get("/api/ranking/inflow-html", response_class=HTMLResponse)
async def ranking_inflow_html(request: Request, q: str = ""):
    """HTMX 局部刷新：资金流入表格，无数据时自动拉取"""
    if not inflow_ranking:
        await refresh_rankings(force=True)
    data = _filter_ranking(inflow_ranking, q)
    return templates.TemplateResponse("_ranking_table.html", {
        "request": request,
        "rows": data,
        "tab": "inflow",
        "col_label": "主力净流入（万元）",
        "value_class": "money-in",
    })


@router.get("/api/ranking/surge-html", response_class=HTMLResponse)
async def ranking_surge_html(request: Request, q: str = ""):
    if not surge_ranking:
        await refresh_rankings(force=True)
    data = _filter_ranking(surge_ranking, q)
    return templates.TemplateResponse("_ranking_table.html", {
        "request": request,
        "rows": data,
        "tab": "surge",
        "col_label": "涨跌幅",
        "value_class": "money-in",
    })


@router.get("/api/ranking/trend-rising-html", response_class=HTMLResponse)
async def ranking_trend_rising_html(request: Request, q: str = ""):
    if not rising_trend:
        await refresh_rankings(force=True)
    data = _filter_ranking(rising_trend, q)
    return templates.TemplateResponse("_trend_table.html", {
        "request": request,
        "rows": data,
        "tab": "trend-rising",
        "col_label": "上升趋势信号",
        "value_class": "money-in",
    })


@router.get("/api/ranking/trend-falling-html", response_class=HTMLResponse)
async def ranking_trend_falling_html(request: Request, q: str = ""):
    if not falling_trend:
        await refresh_rankings(force=True)
    data = _filter_ranking(falling_trend, q)
    return templates.TemplateResponse("_trend_table.html", {
        "request": request,
        "rows": data,
        "tab": "trend-falling",
        "col_label": "下跌趋势信号",
        "value_class": "money-out",
    })


@router.get("/api/ranking/refresh-trend-html", response_class=HTMLResponse)
async def ranking_refresh_trend_html(request: Request):
    await refresh_rankings(force=True)
    # Return both tables wrapped together
    rising_html = templates.get_template("_trend_table.html").render({
        "request": request, "rows": rising_trend,
        "tab": "trend-rising", "col_label": "上升趋势信号", "value_class": "money-in",
    })
    falling_html = templates.get_template("_trend_table.html").render({
        "request": request, "rows": falling_trend,
        "tab": "trend-falling", "col_label": "下跌趋势信号", "value_class": "money-out",
    })
    return HTMLResponse(f'<div id="trend-rising-area">{rising_html}</div><div id="trend-falling-area">{falling_html}</div>')


@router.get("/api/ranking/trend-analysis", response_class=HTMLResponse)
async def ranking_trend_analysis():
    """LLM 趋势分析（按需触发）"""
    from ranking.trend_ranking import TrendRanking
    from core.akshare_client import AkShareClient
    from openai import AsyncOpenAI
    from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

    if not rising_trend and not falling_trend:
        await refresh_rankings(force=True)

    try:
        trend = TrendRanking(AkShareClient())
        prompt = trend.build_llm_prompt(rising_trend[:15], falling_trend[:15])
        llm = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        response = await llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位A股趋势分析专家，擅长从资金流和价格数据中识别趋势信号。"
                               "你只基于提供的数据分析，绝不编造。输出规范Markdown。"
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=2048,
        )
        raw = (response.choices[0].message.content or "").strip()
        import markdown
        md = markdown.Markdown(extensions=["extra", "sane_lists"])
        html = md.convert(raw)
        return HTMLResponse(f'<div class="trend-analysis markdown-body">{html}</div>')
    except Exception as e:
        logger.warning("Trend LLM analysis failed: %s", e)
        return HTMLResponse(f'<div class="error-banner">AI趋势分析暂时不可用：{e}</div>')


def _filter_ranking(data: list[dict], q: str) -> list[dict]:
    if not q:
        return data
    q = q.strip()
    return [r for r in data if q in r.get("code", "") or q in r.get("name", "")]


@router.get("/api/sector/analyze", response_class=HTMLResponse)
async def api_sector_analyze():
    """HTMX 局部刷新：返回板块分析 HTML 片段"""
    result = await _sector.analyze_hot_sectors()
    html = result.get("analysis_html", "")
    generated = result.get("generated_at", "")
    if not html:
        return HTMLResponse("<div class='error-banner'>分析失败，请稍后重试</div>")
    return HTMLResponse(
        f"<div class='data-note'><span class='note-dot'></span> 分析生成时间：{generated}，数据来源：AkShare 实时数据 + DeepSeek AI 分析</div>"
        f"<div class='markdown-body'>{html}</div>"
    )


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
