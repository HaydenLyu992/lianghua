import logging
import numpy as np
from factors.base import FactorBase, FactorResult
from core.akshare_client import AkShareClient
from core.news_fetcher import NewsFetcher

logger = logging.getLogger(__name__)

GEO_KEYWORDS = [
    "战争", "冲突", "制裁", "地缘", "台湾", "南海", "中美", "贸易战",
    "关税", "封锁", "军事", "导弹", "北约", "俄罗斯", "乌克兰",
    "以色列", "伊朗", "朝鲜", "原油", "大宗商品", "供应链中断",
    "芯片禁令", "实体清单", "出口管制", "OPEC", "美联储加息",
    "美股暴跌", "汇率波动", "资本外流",
]


class GeoExternalFactor(FactorBase):
    name = "geo_external"

    def __init__(self, client: AkShareClient, news_fetcher: NewsFetcher):
        self.client = client
        self.news_fetcher = news_fetcher

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        events: list[dict] = []
        score = 50
        try:
            macro_news = self.news_fetcher.fetch_macro_news(limit=30)
            hits = 0
            for item in macro_news:
                title = item.get("title", "")
                if any(kw in title for kw in GEO_KEYWORDS):
                    hits += 1
                    events.append({
                        "title": title,
                        "sentiment": "negative",
                        "impact": 2,
                    })
            if hits == 0:
                score = 65  # no geopolitical noise
            elif hits <= 3:
                score = 45
            else:
                score = 25

        except Exception as e:
            logger.warning("Geo-external factor error for %s: %s", code, e)

        score = int(np.clip(score, 0, 100))
        signal = "bullish" if score >= 65 else ("bearish" if score < 35 else "neutral")
        return FactorResult(
            factor_name=self.name, score=score, signal=signal,
            detail={"地缘事件数": len(events)},
            events=events,
        )
