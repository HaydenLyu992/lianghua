import logging
import numpy as np
from factors.base import FactorBase, FactorResult
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)


class MacroFactor(FactorBase):
    name = "macro"

    def __init__(self, client: AkShareClient):
        self.client = client

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        checks: dict[str, int] = {}
        try:
            pmi = self.client.get_pmi()
            if not pmi.empty:
                checks["PMI"] = self._score_pmi(pmi)

            cpi = self.client.get_cpi()
            if not cpi.empty:
                checks["CPI"] = self._score_cpi(cpi)

            m2 = self.client.get_money_supply()
            if not m2.empty:
                checks["货币供应"] = self._score_m2(m2)

            lpr = self.client.get_lpr()
            if not lpr.empty:
                checks["LPR"] = self._score_lpr(lpr)

        except Exception as e:
            logger.warning("Macro factor error for %s: %s", code, e)

        if not checks:
            return FactorResult(factor_name=self.name, score=50, signal="neutral")

        raw = np.mean(list(checks.values()))
        score = int(np.clip(raw, 0, 100))
        signal = "bullish" if score >= 65 else ("bearish" if score < 35 else "neutral")
        return FactorResult(
            factor_name=self.name, score=score, signal=signal,
            detail=checks,
        )

    def _score_pmi(self, df) -> int:
        latest = float(df.iloc[0].get("制造业", 50))
        return int(np.clip(latest * 2 - 50, 0, 100))

    def _score_cpi(self, df) -> int:
        vals = df.select_dtypes(include=[float]).iloc[0]
        latest = float(vals.iloc[0] if len(vals) > 0 else 2)
        return 70 if 1 <= latest <= 3 else (50 if 0 < latest < 5 else 30)

    def _score_m2(self, df) -> int:
        latest = float(df.iloc[0].get("同比", 8))
        return int(np.clip(latest * 10, 0, 100))

    def _score_lpr(self, df) -> int:
        latest = float(df.iloc[0].get("1年期", 3.5))
        return int(np.clip(100 - latest * 20, 0, 100))
