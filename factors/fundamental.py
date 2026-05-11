import logging
import numpy as np
import pandas as pd
from factors.base import FactorBase, FactorResult
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)


class FundamentalFactor(FactorBase):
    name = "fundamental"

    def __init__(self, client: AkShareClient):
        self.client = client

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        events: list[dict] = []
        checks: dict[str, int] = {}
        try:
            fin = self.client.get_financial_summary(code)
            if not fin.empty:
                checks.update(self._score_financial(fin))

            notices = self.client.get_notices(code)
            if not notices.empty:
                ev, sc = self._score_notices(notices)
                events.extend(ev)
                checks["公告"] = sc

            restricted = self.client.get_restricted_release()
            if not restricted.empty:
                checks["限售解禁"] = self._score_restricted(restricted, code)

            visits = self.client.get_institutional_visits(code)
            if not visits.empty:
                checks["机构调研"] = self._score_visits(visits)

        except Exception as e:
            logger.warning("Fundamental factor error for %s: %s", code, e)

        if not checks:
            return FactorResult(factor_name=self.name, score=50, signal="neutral")

        raw = np.mean(list(checks.values()))
        score = int(np.clip(raw, 0, 100))
        signal = "bullish" if score >= 65 else ("bearish" if score < 35 else "neutral")
        return FactorResult(
            factor_name=self.name, score=score, signal=signal,
            detail=checks, events=events,
        )

    def _score_financial(self, df) -> dict[str, int]:
        checks = {}
        cols = set(df.columns)
        # AkShare 可能返回不同列名，做模糊匹配
        net_profit_col = next((c for c in cols if "净利润" in c), None)
        revenue_col = next((c for c in cols if "营收" in c or "收入" in c), None)
        roe_col = next((c for c in cols if "ROE" in c.upper() or "净资产收益率" in c), None)

        if net_profit_col:
            vals = pd.to_numeric(df[net_profit_col], errors="coerce").dropna()
            if len(vals) >= 2 and vals.iloc[1] != 0:
                growth = float(vals.iloc[0] / vals.iloc[1] - 1)
                checks["净利润增速"] = int(np.clip(50 + growth * 200, 0, 100))

        if revenue_col:
            vals = pd.to_numeric(df[revenue_col], errors="coerce").dropna()
            if len(vals) >= 2 and vals.iloc[1] != 0:
                growth = float(vals.iloc[0] / vals.iloc[1] - 1)
                checks["营收增速"] = int(np.clip(50 + growth * 150, 0, 100))

        if roe_col:
            vals = pd.to_numeric(df[roe_col], errors="coerce").dropna()
            roe = float(vals.iloc[0]) if len(vals) > 0 else 0
            checks["ROE"] = int(np.clip(roe * 5, 0, 100))
        return checks

    def _score_notices(self, df) -> tuple[list[dict], int]:
        events = []
        score = 50
        positive_keywords = ["回购", "增持", "分红", "中标", "战略合作", "减值", "预增"]
        negative_keywords = ["减持", "亏损", "立案", "警示", "关注函", "退市", "预亏"]

        for _, row in df.head(10).iterrows():
            title = str(row.get("title", "") or row.get("name", ""))
            sentiment = "neutral"
            if any(kw in title for kw in positive_keywords):
                sentiment = "positive"
                score += 3
            if any(kw in title for kw in negative_keywords):
                sentiment = "negative"
                score -= 5
            events.append({"title": title, "sentiment": sentiment, "impact": 1})
        score = int(np.clip(score, 0, 100))
        return events, score

    def _score_restricted(self, df, code: str) -> int:
        mask = df["代码"].astype(str).str.contains(code.replace("sz", "").replace("sh", ""))
        upcoming = df[mask]
        if upcoming.empty:
            return 75  # 无解禁 = 中性偏好
        return 30  # 有解禁 = 偏空

    def _score_visits(self, df) -> int:
        count = len(df)
        if count > 20:
            return 80
        elif count > 5:
            return 60
        return 50
