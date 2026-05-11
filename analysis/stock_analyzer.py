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

        factors_with_desc = []
        for key, fr in factor_results.items():
            label = self._factor_label(key)
            weight = FACTOR_WEIGHTS.get(key, 10)
            factors_with_desc.append({
                "key": key,
                "name": label,
                "score": fr.score,
                "signal": fr.signal,
                "weight": weight,
                "detail": fr.detail,
                "events": fr.events,
                "description": self._factor_desc(key, fr.score, fr.signal, fr.detail, weight),
            })

        stock_name = name or self._get_stock_name(code)

        # 5) LLM 综合分析摘要
        summary = await self.llm.comprehensive_analysis(
            code, stock_name, total_score, signal, factors_with_desc,
            limit_result, llm_news,
        )

        report = {
            "code": code,
            "name": stock_name,
            "score": total_score,
            "signal": signal,
            "factors": factors_with_desc,
            "limit": limit_result,
            "news": llm_news,
            "summary": summary,
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

    def _factor_desc(self, key: str, score: int, signal: str, detail: dict,
                     weight: int) -> str:
        """根据因子类型和得分生成通俗易懂的中文解释"""
        level = "优秀" if score >= 80 else "良好" if score >= 65 else "一般" if score >= 45 else "偏弱" if score >= 30 else "较差"
        weight_desc = f"该因子占综合评分的{weight}%权重"
        signal_cn = {"bullish": "看多信号", "bearish": "看空信号", "neutral": "中性信号"}
        signal_str = signal_cn.get(signal, "中性")

        base = f"得分{score}分（{level}），发出{signal_str}。{weight_desc}。"

        if key == "fundamental":
            roe = detail.get("ROE", "N/A") if detail else "N/A"
            return f"{base} 反映公司盈利能力与成长性。当前ROE为{roe}。得分越高说明财报表现越好，利润增长、高ROE、机构调研活跃均为加分项。"

        elif key == "industry":
            change = detail.get("行业涨跌幅", "未知") if detail else "未知"
            return f"{base} 反映所属行业板块的整体热度。当前板块涨跌幅{change}。行业上涨时个股更容易获得板块效应加持。"

        elif key == "macro":
            pmi = detail.get("PMI", "N/A") if detail else "N/A"
            return f"{base} 反映宏观经济环境对股市的影响。当前PMI为{pmi}。PMI>50经济扩张利好股市，CPI温和(1-3%)为佳，M2增速和LPR利率也纳入考量。"

        elif key == "fund_flow":
            return f"{base} 追踪主力资金动向，包括北向资金、融资融券、龙虎榜等。主力持续净流入是强势信号，资金出逃则需警惕。"

        elif key == "sentiment":
            return f"{base} 综合涨停/跌停板情绪、市场热度排名及均线/RSI/MACD等技术指标。高分意味着市场情绪偏多、技术面走强。"

        elif key == "geo_external":
            return f"{base} 监测地缘政治风险事件（战争、制裁、贸易摩擦等）对市场的冲击。得分越高说明当前宏观新闻面偏平静，外部风险较低。"

        return base

    def _get_stock_name(self, code: str) -> str:
        try:
            df = self.client.get_realtime_quote(code)
            return df.get("名称", code)
        except Exception:
            return code
