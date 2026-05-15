import asyncio
import json
import logging
import math

import numpy as np

from core.akshare_client import AkShareClient
from core.news_fetcher import NewsFetcher
from core.database import AsyncSession, AnalysisHistory
from factors.fundamental import FundamentalFactor
from factors.industry import IndustryFactor
from factors.macro import MacroFactor
from factors.fund_flow import FundFlowFactor
from factors.sentiment import SentimentFactor
from factors.geo_external import GeoExternalFactor
from analysis.limit_analyzer import LimitAnalyzer
from analysis.llm_analyzer import LLMAnalyzer

logger = logging.getLogger(__name__)


class StockAnalyzer:
    """综合分析引擎：并行调用 6 因子 + 涨跌停 + LLM，加权汇总生成报告。"""

    @staticmethod
    def _sanitize(obj):
        """递归转换 numpy 类型为 Python 原生类型，确保 JSON 可序列化。"""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return None if math.isnan(val) or math.isinf(val) else val
        if isinstance(obj, np.ndarray):
            return StockAnalyzer._sanitize(obj.tolist())
        if isinstance(obj, dict):
            return {k: StockAnalyzer._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [StockAnalyzer._sanitize(v) for v in obj]
        return obj

    def __init__(
        self,
        client: AkShareClient,
        news_fetcher: NewsFetcher,
        llm: LLMAnalyzer,
    ):
        self.client = client
        self.news_fetcher = news_fetcher
        self.llm = llm

        self.factors = {
            "fundamental": FundamentalFactor(client),
            "industry": IndustryFactor(client),
            "macro": MacroFactor(client),
            "fund_flow": FundFlowFactor(client),
            "sentiment": SentimentFactor(client),
            "geo_external": GeoExternalFactor(client, news_fetcher),
        }
        self.limit_analyzer = LimitAnalyzer(client)

    async def analyze(self, code: str, name: str = "", tracker=None) -> dict:
        """综合分析。若传入 ProgressTracker 则实时更新进度。"""
        # 0) 解析股票名称
        if not name:
            stock_name = await asyncio.to_thread(self._get_stock_name, code)
        else:
            stock_name = name

        # 1) 并行跑 6 个因子 + 涨跌停（线程池）—— 仅收集数据，不打分
        async def _run_factor(key, factor):
            r = await self._safe_factor(key, factor, code, stock_name)
            if tracker:
                tracker.mark_stage(key, "done")
                fr = r[1] if isinstance(r, tuple) else r
                tracker.mark_stage_report(key, self._factor_report_text(key, fr))
            return r

        async def _run_limit():
            r = await self._safe_limit(code)
            if tracker:
                tracker.mark_stage("limit", "done")
                tracker.mark_stage_report("limit", self._limit_report_text(r))
            return r

        if tracker:
            for s in ["fundamental", "industry", "macro", "fund_flow", "sentiment", "geo_external", "limit"]:
                tracker.mark_stage(s, "active")

        tasks = [_run_factor(k, v) for k, v in self.factors.items()]
        tasks.append(_run_limit())
        results = await asyncio.gather(*tasks)

        factor_results = {}
        limit_result = None
        for r in results:
            if isinstance(r, tuple):
                key, fr = r
                factor_results[key] = fr
            elif isinstance(r, dict) and "zt_prob" in r:
                limit_result = r

        # 2) 组装因子数据列表（纯原始数据，无评分）
        factors_brief = []
        for key, fr in factor_results.items():
            if not getattr(fr, "has_data", True):
                continue
            label = self._factor_label(key)
            factors_brief.append({
                "key": key,
                "name": label,
                "detail": fr.detail,
                "events": fr.events,
            })

        # 3) 并行：LLM 新闻分析 + LLM 因子深度分析
        if tracker:
            tracker.mark_stage("llm_news", "active")
            tracker.mark_stage("llm_factors", "active")

        llm_news, factor_descs = await asyncio.gather(
            self.llm.analyze_news(code, tracker=tracker),
            self.llm.analyze_factors(code, stock_name, factors_brief, tracker=tracker),
        )

        if tracker:
            tracker.mark_stage("llm_news", "done")
            tracker.mark_stage_report("llm_news", self._news_report_text(llm_news))
            tracker.mark_stage("llm_factors", "done")
            tracker.mark_stage_report("llm_factors", self._factors_llm_report_text(factor_descs))

        # 4) 组装带LLM分析描述的因子列表
        factors_with_desc = []
        for f in factors_brief:
            desc = factor_descs.get(f["key"], "")
            factors_with_desc.append({**f, "description": desc})

        # 5) LLM 辩论式综合分析 —— 代入所有原始数据 + LLM分析结论
        if tracker:
            tracker.mark_stage("llm_debate", "active")
        summary = await self.llm.comprehensive_analysis(
            code, stock_name, factors_with_desc,
            limit_result, llm_news,
            factor_descs=factor_descs,
            tracker=tracker,
        )
        if tracker:
            tracker.mark_stage("llm_debate", "done")
            tracker.mark_stage_report("llm_debate", self._summary_preview(summary))

        report = {
            "code": code,
            "name": stock_name,
            "factors": factors_with_desc,
            "limit": limit_result,
            "news": llm_news,
            "summary": summary,
        }

        # 递归清理 numpy 类型，确保 JSON 可序列化
        clean_report = self._sanitize(report)

        # 6) 持久化
        await self._save_history(code, factor_results, clean_report)

        return clean_report

    async def _save_history(self, code, factor_results, report):
        try:
            async with AsyncSession() as session:
                record = AnalysisHistory(
                    stock_code=code,
                    score_total=0,
                    score_fund=0,
                    score_ind=0,
                    score_macro=0,
                    score_flow=0,
                    score_sent=0,
                    score_geo=0,
                    signal="llm",
                    report_json=report,
                )
                session.add(record)
                await session.commit()
        except Exception as e:
            logger.warning("Failed to save analysis history: %s", e)

    async def _safe_factor(self, key: str, factor, code: str, name: str):
        try:
            # 因子 analyze() 内部全是同步 AkShare 调用，放入线程池以免阻塞事件循环
            def _run():
                return asyncio.run(factor.analyze(code, name))
            fr = await asyncio.to_thread(_run)
            return (key, fr)
        except Exception as e:
            logger.error("Factor %s failed: %s", key, e)
            from factors.base import FactorResult
            return (key, FactorResult(
                factor_name=key, score=0, signal="no_data",
                detail={"错误": str(e)}, has_data=False,
            ))

    async def _safe_limit(self, code: str):
        try:
            def _run():
                return asyncio.run(self.limit_analyzer.analyze(code))
            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.error("Limit analyzer failed: %s", e)
            return {"zt_prob": 0, "dt_prob": 0, "seal": 0, "streak": 0}

    def _factor_label(self, key: str) -> str:
        return {
            "fundamental": "基本面",
            "industry": "行业",
            "macro": "宏观",
            "fund_flow": "资金流向",
            "sentiment": "情绪技术",
            "geo_external": "地缘",
        }.get(key, key)

    def _get_stock_name(self, code: str) -> str:
        try:
            df = self.client.get_stock_list()
            if df.empty:
                return code
            clean = code.replace("sz", "").replace("sh", "").replace("bj", "")
            row = df[df["代码"].astype(str).str.contains(clean, na=False)]
            if not row.empty:
                return str(row.iloc[0].get("名称", code))
            return code
        except Exception:
            return code

    def _factor_report_text(self, key: str, fr) -> str:
        """Build a human-readable summary for a completed factor stage."""
        label = self._factor_label(key)
        lines = [f"**{label}**"]

        detail = getattr(fr, "detail", {}) or {}
        for k, v in detail.items():
            if k.startswith("_") or k == "数据状态":
                continue
            if isinstance(v, (int, float)):
                lines.append(f"- {k}: {v}")
            elif isinstance(v, str) and v:
                lines.append(f"- {k}: {v}")

        events = getattr(fr, "events", []) or []
        if events:
            lines.append("- 关键事件:")
            for ev in events[:5]:
                sent = ev.get("sentiment", "neutral")
                emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(sent, "⚪")
                lines.append(f"  {emoji} {ev.get('title', '')}")

        if not detail and not events:
            lines.append("— 暂无数据")

        return "\n".join(lines)

    def _limit_report_text(self, r: dict) -> str:
        if not r:
            return "**涨跌停分析** — 无数据"
        zt_p = r.get("zt_prob", 0)
        dt_p = r.get("dt_prob", 0)
        seal = r.get("seal", 0)
        streak = r.get("streak", 0)
        is_zt = r.get("is_zt", False)
        is_dt = r.get("is_dt", False)

        lines = ["**涨跌停分析**"]
        if is_zt:
            lines.append(f"🔥 今日涨停，连板{streak}天，封单{seal}亿")
        elif is_dt:
            lines.append(f"💀 今日跌停")
        else:
            lines.append("今日未涨跌停")
        lines.append(f"- 涨停概率估值: {zt_p}%")
        lines.append(f"- 跌停概率估值: {dt_p}%")
        return "\n".join(lines)

    def _news_report_text(self, news: list[dict]) -> str:
        if not news:
            return "**LLM 新闻分析** — 暂无相关新闻"
        lines = [f"**LLM 新闻分析** — 共 {len(news)} 条"]
        for n in news[:8]:
            sent = n.get("sentiment", "neutral")
            emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(sent, "⚪")
            impact = n.get("impact", "-")
            title = n.get("title", "")
            summary = n.get("summary", "")
            lines.append(f"{emoji} [{impact}/5] {title}")
            if summary:
                lines.append(f"  _{summary}_")
        return "\n".join(lines)

    def _factors_llm_report_text(self, factor_descs: dict[str, str]) -> str:
        if not factor_descs:
            return "**LLM 因子分析** — 分析结果为空"
        lines = ["**LLM 因子深度分析**"]
        label_map = {
            "fundamental": "基本面", "industry": "行业", "macro": "宏观",
            "fund_flow": "资金流向", "sentiment": "情绪技术", "geo_external": "地缘",
        }
        for key, desc in factor_descs.items():
            label = label_map.get(key, key)
            # Truncate each description to ~120 chars for preview
            preview = desc[:150] + "..." if len(desc) > 150 else desc
            lines.append(f"\n**{label}**\n{preview}")
        return "\n".join(lines)

    def _summary_preview(self, summary: str) -> str:
        """Extract a preview from the debate summary HTML."""
        import re
        # Strip HTML tags for preview
        clean = re.sub(r"<[^>]+>", "", summary)
        # Get ~400 chars
        preview = clean[:400]
        if len(clean) > 400:
            preview += "..."
        return preview
