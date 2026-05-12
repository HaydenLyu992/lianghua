import logging
import numpy as np
import pandas as pd
import talib

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
            return 90
        return 5

    def _score_hot(self, df, code: str) -> int:
        total = len(df)
        mask = df["代码"].astype(str).str.contains(code)
        if df[mask].empty:
            return 40
        rank = int(df[mask].iloc[0].get("排名", total // 2))
        return int(np.clip(100 - (rank / total) * 100, 0, 100))

    def _score_technical(self, df: pd.DataFrame) -> dict[str, int]:
        checks = {}
        close_col = next((c for c in df.columns if c in ("close", "收盘")), None)
        if close_col is None:
            return checks
        close = df[close_col].astype(float).values

        high_col = next((c for c in df.columns if c in ("high", "最高")), None)
        low_col = next((c for c in df.columns if c in ("low", "最低")), None)
        vol_col = next((c for c in df.columns if c in ("volume", "成交量")), None)

        high = df[high_col].astype(float).values if high_col else None
        low = df[low_col].astype(float).values if low_col else None
        volume = df[vol_col].astype(float).values if vol_col else None

        # === 1. 均线系统 ===
        if len(close) >= 60:
            ma5 = talib.SMA(close, timeperiod=5)
            ma10 = talib.SMA(close, timeperiod=10)
            ma20 = talib.SMA(close, timeperiod=20)
            ma60 = talib.SMA(close, timeperiod=60)
            if not any(np.isnan(x[-1]) for x in [ma5, ma10, ma20, ma60]):
                bull_align = ma5[-1] > ma10[-1] > ma20[-1] > ma60[-1]
                bear_align = ma5[-1] < ma10[-1] < ma20[-1] < ma60[-1]
                if bull_align:
                    checks["均线多头排列"] = 85
                elif bear_align:
                    checks["均线空头排列"] = 15
                elif ma5[-1] > ma20[-1]:
                    checks["均线偏多"] = 65
                else:
                    checks["均线偏空"] = 35
                # 价格相对均线位置
                above_ma = sum([close[-1] > ma5[-1], close[-1] > ma20[-1], close[-1] > ma60[-1]])
                checks["均线上方数"] = above_ma * 30 + 10  # 10/40/70/100
        elif len(close) >= 20:
            ma5 = talib.SMA(close, timeperiod=5)
            ma20 = talib.SMA(close, timeperiod=20)
            if not np.isnan(ma5[-1]) and not np.isnan(ma20[-1]):
                checks["均线多头"] = 70 if ma5[-1] > ma20[-1] else 30

        # === 2. MACD 深度分析 ===
        if len(close) >= 26:
            macd, signal_line, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            if not np.isnan(macd[-1]):
                # 金叉/死叉状态
                checks["MACD金叉"] = 70 if macd[-1] > signal_line[-1] else 30
                # 柱状图趋势（最近3根的变化方向）
                if len(hist) >= 3:
                    hist_recent = hist[-3:]
                    if all(hist_recent[i] > hist_recent[i-1] for i in range(1, 3)):
                        checks["MACD动能增强"] = 75
                    elif all(hist_recent[i] < hist_recent[i-1] for i in range(1, 3)):
                        checks["MACD动能衰减"] = 25
                # 零轴位置
                checks["MACD零轴"] = 65 if macd[-1] > 0 else 35

        # === 3. RSI 结构分析 ===
        if len(close) >= 14:
            rsi_vals = talib.RSI(close, timeperiod=14)
            rsi = rsi_vals[-1]
            if not np.isnan(rsi):
                if rsi < 25:
                    checks["RSI深度超卖"] = 25
                elif rsi < 35:
                    checks["RSI超卖"] = 40
                elif rsi <= 65:
                    checks["RSI中性"] = 55
                elif rsi <= 75:
                    checks["RSI偏强"] = 65
                else:
                    checks["RSI超买"] = 75
                # RSI 趋势方向（近5日斜率）
                if len(rsi_vals) >= 6:
                    rsi_slope = rsi_vals[-1] - rsi_vals[-6]
                    if rsi_slope > 8:
                        checks["RSI快速走强"] = 70
                    elif rsi_slope < -8:
                        checks["RSI快速走弱"] = 30

        # === 4. 布林带 ===
        if len(close) >= 20:
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            if not any(np.isnan(x[-1]) for x in [upper, middle, lower]):
                bb_width_pct = (upper[-1] - lower[-1]) / middle[-1] * 100
                price_in_bb = (close[-1] - lower[-1]) / (upper[-1] - lower[-1])  # 0-1
                checks["布林带宽%"] = int(np.clip(bb_width_pct * 20, 10, 90))  # 宽=高波动
                if price_in_bb > 0.9:
                    checks["布林上轨区域"] = 75  # 可能超买
                elif price_in_bb < 0.1:
                    checks["布林下轨区域"] = 25  # 可能超卖
                elif price_in_bb > 0.6:
                    checks["布林偏强区域"] = 60
                elif price_in_bb < 0.4:
                    checks["布林偏弱区域"] = 40
                else:
                    checks["布林中轨"] = 50
                # 带宽趋势（收窄可能变盘）
                if len(upper) >= 5:
                    bb_width_prev = (upper[-6] - lower[-6]) / middle[-6] * 100
                    if bb_width_pct < bb_width_prev * 0.9:
                        checks["布林收窄变盘"] = 50  # 信号待方向确认
                    elif bb_width_pct > bb_width_prev * 1.15:
                        checks["布林扩张趋势"] = 65  # 趋势延续

        # === 5. ATR 波动率 ===
        if high is not None and low is not None and len(close) >= 14:
            atr = talib.ATR(high, low, close, timeperiod=14)
            if not np.isnan(atr[-1]):
                atr_pct = atr[-1] / close[-1] * 100  # ATR as % of price
                if atr_pct > 5:
                    checks["ATR极高波动"] = 30  # 高风险
                elif atr_pct > 3:
                    checks["ATR偏高波动"] = 40
                elif atr_pct > 1.5:
                    checks["ATR正常波动"] = 55
                else:
                    checks["ATR低波动"] = 45  # 可能酝酿突破

        # === 6. 成交量分析 ===
        if volume is not None and len(volume) >= 20:
            vol_ma20 = talib.SMA(volume, timeperiod=20)
            if not np.isnan(vol_ma20[-1]) and vol_ma20[-1] > 0:
                vol_ratio = volume[-1] / vol_ma20[-1]
                if vol_ratio > 2.5:
                    checks["巨量成交"] = 75 if close[-1] > close[-2] else 25
                elif vol_ratio > 1.5:
                    checks["放量"] = 65 if close[-1] > close[-2] else 35
                elif vol_ratio < 0.5:
                    checks["极度缩量"] = 50
                elif vol_ratio < 0.8:
                    checks["缩量"] = 45
                else:
                    checks["量能正常"] = 55
            # 量价配合
            if len(close) >= 5 and volume is not None:
                price_up = close[-1] > close[-6]
                vol_up = volume[-5:].mean() > volume[-10:-5].mean()
                if price_up and vol_up:
                    checks["量价配合上涨"] = 75
                elif not price_up and vol_up:
                    checks["放量下跌"] = 25
                elif price_up and not vol_up:
                    checks["缩量上涨"] = 55
                elif not price_up and not vol_up:
                    checks["缩量下跌"] = 45

        # === 7. 支撑/阻力位 ===
        if len(close) >= 20:
            recent_high = np.max(close[-20:])
            recent_low = np.min(close[-20:])
            support_dist = (close[-1] - recent_low) / recent_low * 100
            resist_dist = (recent_high - close[-1]) / close[-1] * 100
            if support_dist < 3:
                checks["靠近支撑位"] = 65
            elif support_dist > 15:
                checks["远离支撑位"] = 35
            if resist_dist < 3:
                checks["靠近阻力位"] = 35
            elif resist_dist > 15:
                checks["远离阻力位"] = 60

            # 多周期高低点
            if len(close) >= 60:
                high_60d = np.max(close[-60:])
                low_60d = np.min(close[-60:])
                if close[-1] >= high_60d * 0.95:
                    checks["接近60日新高"] = 75
                elif close[-1] <= low_60d * 1.05:
                    checks["接近60日新低"] = 25

        return checks
