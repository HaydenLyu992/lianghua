import logging
import time
from factors.base import FactorBase, FactorResult
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)

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
                return FactorResult(
                    factor_name=self.name, has_data=False,
                    detail={"数据状态": "行业指数数据不可用"},
                )

            latest = df.iloc[0]
            change = float(latest.get("涨跌幅", 0) or 0)
            data = {
                "当日行业涨跌幅": f"{change:.2f}%",
                "领涨行业(TOP3)": "、".join(self._top_sectors(df, 3)),
                "领跌行业(TOP3)": "、".join(self._top_sectors(df, -3)),
            }

            # 全行业排名概况
            name_col = next((c for c in df.columns if c in ("板块", "板块名称")), "板块")
            if name_col in df.columns:
                data["行业总数"] = str(len(df))

            return FactorResult(factor_name=self.name, detail=data)
        except Exception as e:
            logger.warning("Industry factor error for %s: %s", code, e)
            return FactorResult(
                factor_name=self.name, has_data=False,
                detail={"错误": str(e)},
            )

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
        name_col = next((c for c in df.columns if c in ("板块", "板块名称")), "板块")
        return sorted_df[name_col].head(abs(n)).tolist()
