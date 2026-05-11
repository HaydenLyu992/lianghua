import logging
import time
import numpy as np
from factors.base import FactorBase, FactorResult
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)

# 行业指数 TTL 缓存 — 非个股相关，5分钟内复用
_ind_cache: dict = {"time": 0, "data": None}
_IND_CACHE_TTL = 300


class IndustryFactor(FactorBase):
    name = "industry"

    def __init__(self, client: AkShareClient):
        self.client = client

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        try:
            df = self._get_cached_index()
            if df.empty:
                return FactorResult(factor_name=self.name, score=50, signal="neutral")

            latest = df.iloc[0]
            change = float(latest.get("涨跌幅", 0) or 0)
            score = int(np.clip(50 + change * 5, 0, 100))

            signal = "bullish" if score >= 65 else ("bearish" if score < 35 else "neutral")
            return FactorResult(
                factor_name=self.name, score=score, signal=signal,
                detail={
                    "行业涨跌幅": f"{change:.2f}%",
                    "领涨行业": self._top_sectors(df, 3),
                    "领跌行业": self._top_sectors(df, -3),
                },
            )
        except Exception as e:
            logger.warning("Industry factor error for %s: %s", code, e)
            return FactorResult(factor_name=self.name, score=50, signal="neutral")

    def _get_cached_index(self):
        global _ind_cache
        now = time.time()
        if _ind_cache["data"] is not None and (now - _ind_cache["time"]) < _IND_CACHE_TTL:
            return _ind_cache["data"]
        df = self.client.get_industry_index()
        _ind_cache = {"time": now, "data": df}
        return df

    def _top_sectors(self, df, n: int) -> list[str]:
        sorted_df = df.sort_values("涨跌幅", ascending=n < 0)
        return sorted_df["板块名称"].head(abs(n)).tolist()
