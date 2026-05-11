import json
import logging
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
            content = content.split("\n", 1)[1]
            content = content.rsplit("\n```", 1)[0]
        return json.loads(content)
