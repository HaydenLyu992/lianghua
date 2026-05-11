import asyncio
import json
import logging
from config import FACTOR_WEIGHTS
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

    async def analyze(self, code: str, name: str = "") -> dict:
        # 1) 并行跑 6 个因子 + 涨跌停
        tasks = [
            self._safe_factor(k, v, code, name)
            for k, v in self.factors.items()
        ]
        tasks.append(self._safe_limit(code))
        results = await asyncio.gather(*tasks)

        factor_results = {}
        limit_result = None
        for r in results:
            if isinstance(r, tuple):
                key, fr = r
                factor_results[key] = fr
            elif isinstance(r, dict) and "zt_prob" in r:
                limit_result = r

        # 2) 加权汇总
        total_score = 0
        total_weight = 0
        for key, fr in factor_results.items():
            w = FACTOR_WEIGHTS.get(key, 10)
            total_score += fr.score * w
            total_weight += w
        total_score = int(total_score / total_weight)

        # 3) LLM 新闻分析
        llm_news = await self.llm.analyze_news(code)

        # 4) 组装报告
        signal = (
            "强烈看多" if total_score >= 80 else
            "中性偏多" if total_score >= 60 else
            "中性" if total_score >= 40 else
            "中性偏空" if total_score >= 20 else
            "强烈看空"
        )

        report = {
            "code": code,
            "name": name or self._get_stock_name(code),
            "score": total_score,
            "signal": signal,
            "factors": [
                {
                    "key": key,
                    "name": self._factor_label(key),
                    "score": fr.score,
                    "signal": fr.signal,
                    "weight": FACTOR_WEIGHTS.get(key, 10),
                    "detail": fr.detail,
                    "events": fr.events,
                }
                for key, fr in factor_results.items()
            ],
            "limit": limit_result,
            "news": llm_news,
        }

        # 5) 持久化到 analysis_history
        await self._save_history(code, total_score, factor_results, signal, report)

        return report

    async def _save_history(self, code, total_score, factor_results, signal, report):
        try:
            scores = {k: fr.score for k, fr in factor_results.items()}
            async with AsyncSession() as session:
                record = AnalysisHistory(
                    stock_code=code,
                    score_total=total_score,
                    score_fund=scores.get("fundamental", 0),
                    score_ind=scores.get("industry", 0),
                    score_macro=scores.get("macro", 0),
                    score_flow=scores.get("fund_flow", 0),
                    score_sent=scores.get("sentiment", 0),
                    score_geo=scores.get("geo_external", 0),
                    signal=signal,
                    report_json=report,
                )
                session.add(record)
                await session.commit()
        except Exception as e:
            logger.warning("Failed to save analysis history: %s", e)

    async def _safe_factor(self, key: str, factor, code: str, name: str):
        try:
            fr = await factor.analyze(code, name)
            return (key, fr)
        except Exception as e:
            logger.error("Factor %s failed: %s", key, e)
            from factors.base import FactorResult
            return (key, FactorResult(factor_name=key, score=50, signal="neutral"))

    async def _safe_limit(self, code: str):
        try:
            return await self.limit_analyzer.analyze(code)
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
            df = self.client.get_realtime_quote(code)
            return df.get("名称", code)
        except Exception:
            return code
