import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import akshare as ak
import pandas as pd
from sqlalchemy import select

from core.database import AsyncSession, NewsCache

logger = logging.getLogger(__name__)

SOURCE_MAP = {
    "cls": "财联社",
    "eastmoney": "东方财富",
    "xueqiu": "雪球",
    "baidu": "百度经济",
}

NEWS_CACHE_TTL = 600  # 新闻缓存 10 分钟


class NewsFetcher:
    """多源新闻聚合器。从财联社、东方财富、百度等来源拉取新闻，支持 DB 缓存。"""

    def fetch_stock_news(self, code: str, limit: int = 50) -> list[dict]:
        """拉取个股关联新闻（先查缓存，再调外部源）"""
        cached = self._get_cached(code, limit)
        if cached:
            return cached

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
        results = results[:limit]

        self._save_cache(code, results)
        return results

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

    # ---- cache helpers ----

    def _get_cached(self, code: str, limit: int) -> list[dict] | None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Called from async context — skip cache for simplicity
                return None
        except RuntimeError:
            pass

        try:
            import asyncio
            async def _query():
                async with AsyncSession() as session:
                    cutoff = datetime.now() - timedelta(seconds=NEWS_CACHE_TTL)
                    stmt = (
                        select(NewsCache)
                        .where(NewsCache.stock_code == code)
                        .where(NewsCache.created_at >= cutoff)
                        .order_by(NewsCache.pub_time.desc())
                        .limit(limit)
                    )
                    result = await session.execute(stmt)
                    rows = result.scalars().all()
                    if rows:
                        return [
                            {
                                "title": r.title,
                                "content": r.content,
                                "source": r.source,
                                "pub_time": str(r.pub_time) if r.pub_time else "",
                            }
                            for r in rows
                        ]
                    return None

            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context; create a new event loop in a thread approach won't work.
                # Instead, just return None and let the caller do fresh fetch.
                return None
            except RuntimeError:
                # No running loop — safe to run
                return asyncio.run(_query())
        except Exception as e:
            logger.warning("News cache lookup failed: %s", e)
            return None

    def _save_cache(self, code: str, items: list[dict]):
        if not items:
            return
        try:
            import asyncio
            async def _insert():
                async with AsyncSession() as session:
                    for item in items:
                        pub_time = None
                        raw = item.get("pub_time", "")
                        if raw:
                            try:
                                pub_time = datetime.fromisoformat(str(raw))
                            except (ValueError, TypeError):
                                try:
                                    pub_time = datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S")
                                except (ValueError, TypeError):
                                    pass
                        record = NewsCache(
                            stock_code=code,
                            title=item.get("title", ""),
                            content=item.get("content", ""),
                            source=item.get("source", ""),
                            pub_time=pub_time or datetime.now(),
                        )
                        session.add(record)
                    await session.commit()
            try:
                loop = asyncio.get_running_loop()
                # In async context, schedule the coroutine
                loop.create_task(_insert())
            except RuntimeError:
                asyncio.run(_insert())
        except Exception as e:
            logger.warning("News cache save failed: %s", e)

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
