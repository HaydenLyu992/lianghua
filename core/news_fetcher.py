import logging
from datetime import datetime
from typing import Optional

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_MAP = {
    "cls": "财联社",
    "eastmoney": "东方财富",
    "xueqiu": "雪球",
    "baidu": "百度经济",
}


class NewsFetcher:
    """多源新闻聚合器。从财联社、东方财富、百度等来源拉取新闻。"""

    def fetch_stock_news(self, code: str, limit: int = 50) -> list[dict]:
        """拉取个股关联新闻"""
        results: list[dict] = []

        sources = [
            self._fetch_cls,
            self._fetch_eastmoney,
            self._fetch_baidu,
        ]

        for source_fn in sources:
            try:
                items = source_fn(code, limit)
                results.extend(items)
            except Exception as e:
                logger.warning("News source %s failed for %s: %s", source_fn.__name__, code, e)

        results.sort(key=lambda x: x.get("pub_time", ""), reverse=True)
        return results[:limit]

    def fetch_macro_news(self, limit: int = 30) -> list[dict]:
        """拉取宏观/政策新闻"""
        try:
            df = ak.news_economic_baidu()
            items = []
            for _, row in df.head(limit).iterrows():
                items.append({
                    "title": row.get("title", ""),
                    "content": row.get("content", ""),
                    "source": "baidu",
                    "pub_time": str(row.get("date", "")),
                })
            return items
        except Exception as e:
            logger.warning("Failed to fetch macro news: %s", e)
            return []

    # ---- private ----

    def _fetch_cls(self, code: str, limit: int) -> list[dict]:
        try:
            df = ak.stock_news_em(symbol=code)
            items = []
            for _, row in df.head(limit).iterrows():
                items.append({
                    "title": row.get("title", ""),
                    "content": row.get("content", ""),
                    "source": "eastmoney",
                    "pub_time": str(row.get("pub_time", "")),
                })
            return items
        except Exception:
            return []

    def _fetch_eastmoney(self, code: str, limit: int) -> list[dict]:
        try:
            df = ak.stock_info_global_em(symbol=code)
            items = []
            for _, row in df.head(limit).iterrows():
                items.append({
                    "title": row.get("title", ""),
                    "content": "",
                    "source": "eastmoney",
                    "pub_time": str(row.get("pub_time", datetime.now())),
                })
            return items
        except Exception:
            return []

    def _fetch_baidu(self, code: str, limit: int) -> list[dict]:
        try:
            df = ak.news_report_em(symbol=code)
            items = []
            for _, row in df.head(limit).iterrows():
                items.append({
                    "title": row.get("title", ""),
                    "content": "",
                    "source": "eastmoney",
                    "pub_time": str(row.get("pub_time", "")),
                })
            return items
        except Exception:
            return []
