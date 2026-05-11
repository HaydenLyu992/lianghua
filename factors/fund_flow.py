import logging
import numpy as np
from factors.base import FactorBase, FactorResult
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)


class FundFlowFactor(FactorBase):
    name = "fund_flow"

    def __init__(self, client: AkShareClient):
        self.client = client

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        checks: dict[str, int] = {}
        events: list[dict] = []
        try:
            nf = self.client.get_northbound_flow()
            if not nf.empty:
                checks["北向资金"] = self._score_northbound(nf)

            margin = self.client.get_margin_detail()
            if not margin.empty:
                checks["融资融券"] = self._score_margin(margin, code)

            ff = self.client.get_fund_flow_individual(code)
            if not ff.empty:
                sc, ev = self._score_fund_flow(ff)
                checks["主力资金"] = sc
                events.extend(ev)

            dt = self.client.get_dragon_tiger()
            if not dt.empty:
                checks["龙虎榜"] = self._score_dt(dt, code)

        except Exception as e:
            logger.warning("Fund flow factor error for %s: %s", code, e)

        if not checks:
            return FactorResult(factor_name=self.name, score=50, signal="neutral")

        raw = np.mean(list(checks.values()))
        score = int(np.clip(raw, 0, 100))
        signal = "bullish" if score >= 65 else ("bearish" if score < 35 else "neutral")
        return FactorResult(
            factor_name=self.name, score=score, signal=signal,
            detail=checks, events=events,
        )

    def _score_northbound(self, df) -> int:
        recent = df.head(5)
        net = recent["当日资金净流入"].astype(float).sum() if "当日资金净流入" in df.columns else 0
        return int(np.clip(50 + net / 1e9, 0, 100))

    def _score_margin(self, df, code: str) -> int:
        try:
            mask = df["标的代码"].astype(str).str.contains(code)
            row = df[mask]
            if row.empty:
                return 50
            bal = float(row.iloc[0].get("融资余额", 0))
            return int(np.clip(bal / 1e7, 0, 100))
        except Exception:
            return 50

    def _score_fund_flow(self, df) -> tuple[int, list[dict]]:
        events = []
        df_cols = {c.lower(): c for c in df.columns}
        main_col = df_cols.get("主力净流入", df_cols.get("主力净流入-净额"))
        if not main_col:
            return 50, events

        recent = df.head(5)[main_col].astype(float)
        net = recent.sum()
        score = int(np.clip(50 + net / 5e7, 0, 100))
        if net > 1e8:
            events.append({"title": f"近5日主力净流入 {net/1e8:.1f}亿", "sentiment": "positive", "impact": 3})
        elif net < -1e8:
            events.append({"title": f"近5日主力净流出 {abs(net)/1e8:.1f}亿", "sentiment": "negative", "impact": 3})
        return score, events

    def _score_dt(self, df, code: str) -> int:
        try:
            mask = df["代码"].astype(str).str.contains(code)
            if df[mask].empty:
                return 50
            return 75
        except Exception:
            return 50
