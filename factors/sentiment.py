import logging
import numpy as np
import pandas as pd

from factors.base import FactorBase, FactorResult
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)


class SentimentFactor(FactorBase):
    name = "sentiment"

    def __init__(self, client: AkShareClient):
        self.client = client

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        checks: dict[str, int] = {}
        events: list[dict] = []
        try:
            zt = self.client.get_limit_up_pool()
            if not zt.empty:
                checks["涨停情绪"] = self._score_limit_up(zt, code, events)

            dt = self.client.get_limit_down_pool()
            if not dt.empty:
                checks["跌停风险"] = 100 - self._score_limit_down(dt, code)

            hot = self.client.get_hot_rank()
            if not hot.empty:
                checks["市场热度"] = self._score_hot(hot, code)

            kline = self.client.get_daily_kline(code)
            if not kline.empty:
                tech = self._score_technical(kline)
                checks.update(tech)

        except Exception as e:
            logger.warning("Sentiment factor error for %s: %s", code, e)

        if not checks:
            return FactorResult(factor_name=self.name, score=50, signal="neutral")

        raw = np.mean(list(checks.values()))
        score = int(np.clip(raw, 0, 100))
        signal = "bullish" if score >= 65 else ("bearish" if score < 35 else "neutral")
        return FactorResult(
            factor_name=self.name, score=score, signal=signal,
            detail=checks, events=events,
        )

    def _score_limit_up(self, df, code: str, events: list) -> int:
        total = len(df)
        mask = df["代码"].astype(str).str.contains(code)
        matched = df[mask]
        if not matched.empty:
            events.append({"title": "今日涨停", "sentiment": "positive", "impact": 5})
            row = matched.iloc[0]
            seal = float(row.get("封单金额", row.get("封板强度", 50)) or 0) / 1e8
            return int(np.clip(min(seal * 3, 100), 60, 100))
        return min(int(np.clip(total / 2, 30, 100)), 70)

    def _score_limit_down(self, df, code: str) -> int:
        mask = df["代码"].astype(str).str.contains(code)
        if not df[mask].empty:
            return 90  # high downside risk (penalty in parent)
        return 5

    def _score_hot(self, df, code: str) -> int:
        total = len(df)
        mask = df["代码"].astype(str).str.contains(code)
        if df[mask].empty:
            return 40  # not on hot list
        rank = int(df[mask].iloc[0].get("排名", total // 2))
        return int(np.clip(100 - (rank / total) * 100, 0, 100))

    def _score_technical(self, df: pd.DataFrame) -> dict[str, int]:
        checks = {}
        close = df["收盘"].astype(float)

        # MA trend
        if len(close) >= 20:
            ma5 = close.rolling(5).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            if ma5 > ma20:
                checks["均线多头"] = 70
            else:
                checks["均线空头"] = 30

        # RSI
        if len(close) >= 14:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
            loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
            rs = gain / loss if loss != 0 else 100
            rsi = 100 - (100 / (1 + rs))
            if 30 <= rsi <= 70:
                checks["RSI中性"] = 55
            elif rsi < 30:
                checks["RSI超卖"] = 30
            else:
                checks["RSI超买"] = 70

        # MACD
        if len(close) >= 26:
            ema12 = close.ewm(span=12).mean().iloc[-1]
            ema26 = close.ewm(span=26).mean().iloc[-1]
            macd = ema12 - ema26
            if macd > 0:
                checks["MACD金叉"] = 70
            else:
                checks["MACD死叉"] = 30

        return checks
