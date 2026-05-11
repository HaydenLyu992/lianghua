import logging
import numpy as np
import pandas as pd
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)


class LimitAnalyzer:
    """涨跌停专项分析：封板强度、连板高度、涨跌停概率估算。"""

    def __init__(self, client: AkShareClient):
        self.client = client

    async def analyze(self, code: str) -> dict:
        try:
            zt = self.client.get_limit_up_pool()
            dt = self.client.get_limit_down_pool()
        except Exception:
            return {"zt_prob": 5, "dt_prob": 3, "seal": 0, "streak": 0, "is_zt": False, "is_dt": False}

        zt_mask = zt["代码"].astype(str).str.contains(code) if not zt.empty else pd.Series()
        dt_mask = dt["代码"].astype(str).str.contains(code) if not dt.empty else pd.Series()

        is_zt = not zt.empty and zt[zt_mask].shape[0] > 0
        is_dt = not dt.empty and dt[dt_mask].shape[0] > 0

        seal = 0
        streak = 0
        if is_zt:
            row = zt[zt_mask].iloc[0]
            seal = float(row.get("封单金额", 0) or 0) / 1e8
            streak = int(row.get("连板数", 1) or 1)

        # Probability estimation
        zt_prob = self._estimate_zt(is_zt, seal, streak)
        dt_prob = self._estimate_dt(is_dt)

        return {
            "zt_prob": zt_prob,
            "dt_prob": dt_prob,
            "seal": round(seal, 2),
            "streak": streak,
            "is_zt": is_zt,
            "is_dt": is_dt,
        }

    def _estimate_zt(self, is_zt: bool, seal: float, streak: int) -> int:
        if is_zt:
            return int(np.clip(80 + seal * 5 + streak * 3, 0, 100))
        return int(np.clip(5 + seal * 2, 0, 30))

    def _estimate_dt(self, is_dt: bool) -> int:
        return 85 if is_dt else 5


