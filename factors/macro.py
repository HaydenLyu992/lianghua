import logging
import time
from factors.base import FactorBase, FactorResult
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)

_macro_cache: dict = {"time": 0, "pmi": None, "cpi": None, "m2": None, "lpr": None}
_MACRO_CACHE_TTL = 3600


class MacroFactor(FactorBase):
    name = "macro"

    def __init__(self, client: AkShareClient):
        self.client = client

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        data: dict[str, str] = {}
        try:
            pmi = self._cached("pmi", self.client.get_pmi)
            if pmi is not None and not pmi.empty:
                latest = float(pmi.iloc[0].get("制造业", 50))
                data["PMI(制造业)"] = str(latest)

            cpi = self._cached("cpi", self.client.get_cpi)
            if cpi is not None and not cpi.empty:
                vals = cpi.select_dtypes(include=[float]).iloc[0]
                if len(vals) > 0:
                    data["CPI(最新)"] = str(vals.iloc[0])

            m2 = self._cached("m2", self.client.get_money_supply)
            if m2 is not None and not m2.empty:
                latest_m2 = float(m2.iloc[0].get("同比", "N/A"))
                if str(latest_m2) != "nan":
                    data["M2同比增速"] = f"{latest_m2}%"

            lpr = self._cached("lpr", self.client.get_lpr)
            if lpr is not None and not lpr.empty:
                lpr_1y = float(lpr.iloc[0].get("1年期", "N/A"))
                lpr_5y = float(lpr.iloc[0].get("5年期", "N/A"))
                if str(lpr_1y) != "nan":
                    data["LPR(1年期)"] = f"{lpr_1y}%"
                if str(lpr_5y) != "nan":
                    data["LPR(5年期)"] = f"{lpr_5y}%"

        except Exception as e:
            logger.warning("Macro factor error for %s: %s", code, e)

        if not data:
            return FactorResult(
                factor_name=self.name, has_data=False,
                detail={"数据状态": "所有宏观数据源均不可用"},
            )

        return FactorResult(factor_name=self.name, detail=data)

    def _cached(self, key: str, fetcher):
        global _macro_cache
        now = time.time()
        if _macro_cache[key] is not None and (now - _macro_cache["time"]) < _MACRO_CACHE_TTL:
            return _macro_cache[key]
        data = fetcher()
        _macro_cache[key] = data
        _macro_cache["time"] = now
        return data
