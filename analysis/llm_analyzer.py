import json
import logging
import re
from typing import Optional
from openai import AsyncOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from core.news_fetcher import NewsFetcher

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个A股新闻分析助手。对给定的新闻列表，分析每条新闻对对应股票的影响：

1. sentiment: "positive"(利好) / "negative"(利空) / "neutral"(中性)
2. impact: 1-5 的影响级别（5=重大影响）
3. summary: 10字以内的简述

以JSON数组格式返回，每个元素包含 title, sentiment, impact, summary。
只返回JSON，不要其他内容。"""


class LLMAnalyzer:
    """LLM 新闻分析器 — 使用 DeepSeek (OpenAI 兼容协议)，强制启用。"""

    def __init__(self):
        self.news_fetcher = NewsFetcher()
        self.client = AsyncOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )

    async def analyze_news(self, code: str) -> list[dict]:
        """拉取个股新闻并用 LLM 做情感分析"""
        raw_news = self.news_fetcher.fetch_stock_news(code, limit=20)
        if not raw_news:
            return []

        titles = [item["title"] for item in raw_news if item.get("title")]
        if not titles:
            return []

        response = await self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(titles, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or "[]"
        content = content.strip()
        if content.startswith("```"):
            parts = content.split("\n", 1)
            content = parts[1] if len(parts) > 1 else content
            content = content.rsplit("\n```", 1)[0] if content.endswith("```") else content.rstrip("`")
        return self._safe_json_parse(content)

    async def comprehensive_analysis(self, code: str, name: str, total_score: int,
                                      signal: str, factors: list[dict],
                                      limit_info: dict, news: list[dict]) -> str:
        """综合所有因子结果，生成一段直截了当的买卖建议"""
        factor_text = "\n".join(
            f"- {f['name']}(权重{f['weight']}%): {f['score']}分 {f['signal']}"
            for f in factors
        )
        news_text = "\n".join(
            f"- [{n.get('sentiment','neutral')}] {n.get('title','')}"
            for n in (news or [])[:10]
        ) or "无相关新闻"

        limit_text = ""
        if limit_info:
            if limit_info.get("is_zt"):
                limit_text = f"该股今日涨停，连板{limit_info.get('streak',0)}天，封单{limit_info.get('seal',0)}亿。"
            elif limit_info.get("is_dt"):
                limit_text = "该股今日跌停，风险较高。"

        prompt = f"""你是一位资深A股分析师。请根据以下多因子分析结果，给出一段150字以内的综合研判：

股票: {code} {name}
综合得分: {total_score}/100 ({signal})

各因子详情:
{factor_text}

涨跌停情况: {limit_text or '无异常'}

近期新闻:
{news_text}

请直接给出：1)涨跌趋势判断 2)今日是否适合买入 3)建议买入价格区间 4)建议卖出/止损价格。
要求语言简明扼要，不模棱两可。以纯文本返回，不要markdown格式。"""

        try:
            response = await self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一位A股量化分析师。给出直截了当、简洁明确的买卖建议。不模棱两可。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("Comprehensive analysis failed: %s", e)
            return self._fallback_summary(total_score, signal, factors, limit_info)

    def _fallback_summary(self, total_score: int, signal: str, factors: list[dict],
                          limit_info: dict) -> str:
        """LLM不可用时的规则兜底"""
        bullish = sum(1 for f in factors if f["signal"] == "bullish")
        bearish = sum(1 for f in factors if f["signal"] == "bearish")

        if total_score >= 80:
            action = "强势看多，可考虑逢低买入。建议买入区间为近期均价的-3%~-5%，止损设在买入价下方5%。"
        elif total_score >= 60:
            action = "中性偏多，可轻仓试探。建议回调至均线附近买入，止损设为买入价下方3%。"
        elif total_score >= 40:
            action = "方向不明，建议观望等待更明确信号。若已持仓可继续持有，不建议新开仓。"
        elif total_score >= 20:
            action = "中性偏空，不宜买入。若已持仓建议逢高减仓，止损不宜过远。"
        else:
            action = "强烈看空，不建议买入。持仓者应考虑止损离场。"

        limit_note = ""
        if limit_info.get("is_zt"):
            limit_note = "该股今日涨停，追高风险较大。"
        elif limit_info.get("is_dt"):
            limit_note = "该股今日跌停，短期仍有下行风险。"

        return f"综合{len(factors)}个维度分析：{signal}（{total_score}分）。{bullish}个因子看多，{bearish}个因子看空。{action}{limit_note}"

    def _safe_json_parse(self, content: str) -> list[dict]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning("LLM returned unparseable content: %s", content[:200])
        return []
