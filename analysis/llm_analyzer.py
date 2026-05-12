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
        """综合所有因子结果，生成深度研判分析"""
        factor_lines = []
        for f in factors:
            detail_str = ""
            if f.get("detail"):
                items = [
                    f"{k}: {v}" for k, v in f["detail"].items()
                    if not k.startswith("_") and k != "数据状态"
                ]
                if items:
                    detail_str = f"（{', '.join(items[:5])}）"
            factor_lines.append(
                f"- {f['name']}(权重{f['weight']}%): {f['score']}分 {f['signal']} {detail_str}"
            )
        factor_text = "\n".join(factor_lines)

        news_text = "\n".join(
            f"- [{n.get('sentiment','neutral')}] {n.get('title','')} — {n.get('summary','')}"
            for n in (news or [])[:15]
        ) or "无相关新闻"

        limit_text = ""
        if limit_info:
            if limit_info.get("is_zt"):
                limit_text = (
                    f"该股今日涨停，连板{limit_info.get('streak',0)}天，"
                    f"封单{limit_info.get('seal',0)}亿。"
                    f"涨停概率估值{limit_info.get('zt_prob',0)}%，"
                    f"跌停概率估值{limit_info.get('dt_prob',0)}%。"
                )
            elif limit_info.get("is_dt"):
                limit_text = (
                    f"该股今日跌停。"
                    f"涨停概率估值{limit_info.get('zt_prob',0)}%，"
                    f"跌停概率估值{limit_info.get('dt_prob',0)}%。"
                )
            else:
                limit_text = (
                    f"该股今日未涨跌停。"
                    f"涨停概率估值{limit_info.get('zt_prob',0)}%，"
                    f"跌停概率估值{limit_info.get('dt_prob',0)}%。"
                )

        prompt = f"""你是一位资深A股量化分析师，拥有20年实战经验。请基于以下多维数据，对该股进行一次深入、全面的研判分析。要求分析详细、有数据支撑，避免空洞的结论。

**股票信息**
代码: {code}
名称: {name}
综合得分: {total_score}/100
综合信号: {signal}

**各因子详细评分**
{factor_text}

**涨跌停分析**
{limit_text or '无异常'}

**近期新闻（已标注情感）**
{news_text}

请从以下几个维度展开深度分析（总篇幅800-1200字）：

1. **趋势研判**（200-300字）
   - 结合各因子得分，判断该股当前处于什么阶段（上升趋势/下降趋势/震荡整理）
   - 哪些因子在支撑趋势，哪些因子在发出反向信号
   - 未来1-2周最可能的走势推演

2. **多空博弈分析**（200-300字）
   - 多方的核心逻辑和支撑依据是什么
   - 空方的核心逻辑和风险点在哪里
   - 当前市场环境下多空力量对比的评估
   - 结合资金面（北向、主力、融资融券）分析资金博弈格局

3. **买入建议**（150-200字）
   - 今日是否适合买入（明确回答：适合/观望/不适合）
   - 如果适合，建议的买入价格区间及依据（参考均线支撑位、近期低点等）
   - 如果不适合，需要等待什么条件出现才可以考虑买入

4. **风险控制**（150-200字）
   - 建议的止损价位及计算逻辑
   - 建议的止盈目标价位
   - 当前仓位建议（轻仓/中等/重仓）
   - 需要重点监控的風險因素

5. **关键观察点**（100-150字）
   - 未来1-3个交易日内最值得关注的信号或事件
   - 哪些数据变化可能导致研判结论需要修正

要求：语言专业但不晦涩，有数据支撑每一个判断，不模棱两可，给出明确的操作建议。以纯文本返回，适当分段。"""

        try:
            response = await self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一位资深A股量化分析师，擅长多因子模型分析和资金面研判。"
                            "你的分析风格：数据驱动、逻辑严密、敢于给出明确判断，同时充分揭示风险。"
                            "你从不模棱两可，每个结论都有具体的数据支撑。"
                            "你对技术面、基本面、资金面、宏观面都有深刻理解，能够综合多维度信息。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
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
