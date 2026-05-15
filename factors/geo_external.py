import logging
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

POLICY_KEYWORDS = [
    "降准", "降息", "LPR", "MLF", "逆回购", "社融", "M2",
    "证监会", "银保监", "央行", "国务院", "发改委", "工信部",
    "新质生产力", "半导体", "新能源补贴", "房地产调控", "平台经济",
    "专项债", "减税降费", "自贸区", "一带一路", "碳中和",
    "注册制", "退市", "减持新规", "再融资", "IPO",
    "数据要素", "算力", "低空经济", "人形机器人",
]


class GeoExternalFactor(FactorBase):
    name = "geo_external"

    def __init__(self, client: AkShareClient, news_fetcher: NewsFetcher):
        self.client = client
        self.news_fetcher = news_fetcher

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        events: list[dict] = []
        geo_hits = 0
        policy_hits = 0
        geo_titles: list[str] = []
        policy_titles: list[str] = []
        try:
            macro_news = self.news_fetcher.fetch_macro_news(limit=30)
            for item in macro_news:
                title = item.get("title", "")
                if any(kw in title for kw in GEO_KEYWORDS):
                    geo_hits += 1
                    geo_titles.append(title)
                    events.append({"title": title, "sentiment": "negative", "impact": 2})
                if any(kw in title for kw in POLICY_KEYWORDS):
                    policy_hits += 1
                    policy_titles.append(title)
                    bullish_kw = ["降准", "降息", "减税", "补贴", "新质生产力", "一带一路",
                                  "碳中和", "数据要素", "算力", "低空经济", "注册制"]
                    bearish_kw = ["调控", "退市", "减持", "罚款", "立案", "警示"]
                    is_bullish = any(k in title for k in bullish_kw)
                    is_bearish = any(k in title for k in bearish_kw)
                    if is_bullish and not is_bearish:
                        events.append({"title": title, "sentiment": "positive", "impact": 2})
                    elif is_bearish:
                        events.append({"title": title, "sentiment": "negative", "impact": 2})
                    else:
                        events.append({"title": title, "sentiment": "neutral", "impact": 1})

        except Exception as e:
            logger.warning("Geo-external factor error for %s: %s", code, e)
            return FactorResult(
                factor_name=self.name, has_data=False,
                detail={"错误": str(e)},
            )

        data = {
            "地缘风险事件数": str(geo_hits),
            "政策相关事件数": str(policy_hits),
        }
        if geo_titles:
            data["地缘事件摘要"] = "；".join(geo_titles[:5])
        if policy_titles:
            data["政策事件摘要"] = "；".join(policy_titles[:5])

        return FactorResult(
            factor_name=self.name, detail=data, events=events,
        )
