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
        data: dict[str, str] = {}
        events: list[dict] = []
        try:
            # 涨停池
            zt = self.client.get_limit_up_pool()
            if not zt.empty:
                raw = code.replace("sz", "").replace("sh", "").replace("bj", "")
                mask = zt["代码"].astype(str).str.contains(raw)
                if not zt[mask].empty:
                    row = zt[mask].iloc[0]
                    seal = float(row.get("封单金额", 0) or 0) / 1e8
                    streak = int(row.get("连板数", 1) or 1)
                    data["涨停状态"] = f"已涨停(连板{streak}天, 封单{seal:.1f}亿)"
                    events.append({"title": "今日涨停", "sentiment": "positive", "impact": 5})
                data["全市场涨停家数"] = str(len(zt))

            # 跌停池
            dt = self.client.get_limit_down_pool()
            if not dt.empty:
                raw = code.replace("sz", "").replace("sh", "").replace("bj", "")
                mask = dt["代码"].astype(str).str.contains(raw)
                if not dt[mask].empty:
                    data["跌停状态"] = "已跌停"
                    events.append({"title": "今日跌停", "sentiment": "negative", "impact": 5})
                data["全市场跌停家数"] = str(len(dt))

            # 热门排名
            hot = self.client.get_hot_rank()
            if not hot.empty:
                raw = code.replace("sz", "").replace("sh", "").replace("bj", "")
                mask = hot["代码"].astype(str).str.contains(raw)
                if not hot[mask].empty:
                    rank = int(hot[mask].iloc[0].get("排名", "N/A"))
                    data["市场热度排名"] = str(rank)
                else:
                    data["市场热度排名"] = f"未进前{len(hot)}"

            # K线技术指标
            kline = self.client.get_daily_kline(code)
            if not kline.empty:
                tech_data = self._collect_technical(kline)
                data.update(tech_data)

        except Exception as e:
            logger.warning("Sentiment factor error for %s: %s", code, e)

        if not data:
            return FactorResult(
                factor_name=self.name, has_data=False,
                detail={"数据状态": "所有情绪技术数据源均不可用"},
            )

        return FactorResult(
            factor_name=self.name, detail=data, events=events,
        )

    def _collect_technical(self, df: pd.DataFrame) -> dict[str, str]:
        """收集技术指标原始数据，不评分"""
        data = {}
        close_col = next((c for c in df.columns if c in ("close", "收盘")), None)
        if close_col is None:
            return data
        close = df[close_col].astype(float).values

        high_col = next((c for c in df.columns if c in ("high", "最高")), None)
        low_col = next((c for c in df.columns if c in ("low", "最低")), None)
        vol_col = next((c for c in df.columns if c in ("volume", "成交量")), None)

        high = df[high_col].astype(float).values if high_col else None
        low = df[low_col].astype(float).values if low_col else None
        volume = df[vol_col].astype(float).values if vol_col else None

        # 1. 均线系统
        if len(close) >= 60:
            ma5 = talib.SMA(close, timeperiod=5)
            ma10 = talib.SMA(close, timeperiod=10)
            ma20 = talib.SMA(close, timeperiod=20)
            ma60 = talib.SMA(close, timeperiod=60)
            if not any(np.isnan(x[-1]) for x in [ma5, ma10, ma20, ma60]):
                data["MA5"] = f"{ma5[-1]:.2f}"
                data["MA10"] = f"{ma10[-1]:.2f}"
                data["MA20"] = f"{ma20[-1]:.2f}"
                data["MA60"] = f"{ma60[-1]:.2f}"
                data["当前价格"] = f"{close[-1]:.2f}"
                # 排列形态
                if ma5[-1] > ma10[-1] > ma20[-1] > ma60[-1]:
                    data["均线排列"] = "多头排列(MA5>MA10>MA20>MA60)"
                elif ma5[-1] < ma10[-1] < ma20[-1] < ma60[-1]:
                    data["均线排列"] = "空头排列(MA5<MA10<MA20<MA60)"
                elif ma5[-1] > ma20[-1]:
                    data["均线排列"] = "短期偏多(MA5>MA20)"
                else:
                    data["均线排列"] = "短期偏空(MA5<MA20)"
                above = sum([close[-1] > ma5[-1], close[-1] > ma20[-1], close[-1] > ma60[-1]])
                data["价格站上均线数"] = f"{above}/3"

        # 2. MACD
        if len(close) >= 26:
            macd, signal_line, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            if not np.isnan(macd[-1]):
                data["MACD_DIF"] = f"{macd[-1]:.4f}"
                data["MACD_DEA"] = f"{signal_line[-1]:.4f}"
                data["MACD_柱"] = f"{hist[-1]:.4f}"
                data["MACD状态"] = "金叉(DIF>DEA)" if macd[-1] > signal_line[-1] else "死叉(DIF<DEA)"
                data["MACD零轴"] = "零轴上方" if macd[-1] > 0 else "零轴下方"
                if len(hist) >= 3:
                    hist_recent = hist[-3:]
                    if all(hist_recent[i] > hist_recent[i-1] for i in range(1, 3)):
                        data["MACD动能"] = "增强(绿柱缩短/红柱增长)"
                    elif all(hist_recent[i] < hist_recent[i-1] for i in range(1, 3)):
                        data["MACD动能"] = "衰减(红柱缩短/绿柱增长)"

        # 3. RSI
        if len(close) >= 14:
            rsi_vals = talib.RSI(close, timeperiod=14)
            rsi = rsi_vals[-1]
            if not np.isnan(rsi):
                data["RSI(14)"] = f"{rsi:.1f}"
                if rsi < 25:
                    data["RSI区域"] = "深度超卖(<25)"
                elif rsi < 35:
                    data["RSI区域"] = "超卖(25-35)"
                elif rsi <= 65:
                    data["RSI区域"] = "中性(35-65)"
                elif rsi <= 75:
                    data["RSI区域"] = "偏强(65-75)"
                else:
                    data["RSI区域"] = "超买(>75)"

        # 4. 布林带
        if len(close) >= 20:
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            if not any(np.isnan(x[-1]) for x in [upper, middle, lower]):
                data["布林上轨"] = f"{upper[-1]:.2f}"
                data["布林中轨"] = f"{middle[-1]:.2f}"
                data["布林下轨"] = f"{lower[-1]:.2f}"
                bb_width = (upper[-1] - lower[-1]) / middle[-1] * 100
                data["布林带宽"] = f"{bb_width:.1f}%"
                price_in_bb = (close[-1] - lower[-1]) / (upper[-1] - lower[-1])
                if price_in_bb > 0.9:
                    data["布林位置"] = "上轨附近(偏强)"
                elif price_in_bb < 0.1:
                    data["布林位置"] = "下轨附近(偏弱)"
                elif price_in_bb > 0.6:
                    data["布林位置"] = "中轨上方(偏强)"
                elif price_in_bb < 0.4:
                    data["布林位置"] = "中轨下方(偏弱)"
                else:
                    data["布林位置"] = "中轨附近"

        # 5. ATR
        if high is not None and low is not None and len(close) >= 14:
            atr = talib.ATR(high, low, close, timeperiod=14)
            if not np.isnan(atr[-1]):
                atr_pct = atr[-1] / close[-1] * 100
                data["ATR"] = f"{atr[-1]:.2f}"
                data["ATR(价格%)"] = f"{atr_pct:.1f}%"

        # 6. 成交量分析
        if volume is not None and len(volume) >= 20:
            vol_ma20 = talib.SMA(volume, timeperiod=20)
            if not np.isnan(vol_ma20[-1]) and vol_ma20[-1] > 0:
                vol_ratio = volume[-1] / vol_ma20[-1]
                data["成交量(相对20日均量)"] = f"{vol_ratio:.1f}倍"
            if len(close) >= 5:
                price_up = close[-1] > close[-6]
                vol_up = volume[-5:].mean() > volume[-10:-5].mean()
                if price_up and vol_up:
                    data["量价关系"] = "价涨量增(健康)"
                elif not price_up and vol_up:
                    data["量价关系"] = "价跌量增(偏空)"
                elif price_up and not vol_up:
                    data["量价关系"] = "价涨量缩(动能不足)"
                elif not price_up and not vol_up:
                    data["量价关系"] = "价跌量缩(偏弱)"

        # 7. 支撑/阻力位
        if len(close) >= 20:
            recent_high = float(np.max(close[-20:]))
            recent_low = float(np.min(close[-20:]))
            data["20日最高价"] = f"{recent_high:.2f}"
            data["20日最低价"] = f"{recent_low:.2f}"
            support_dist = (close[-1] - recent_low) / recent_low * 100
            resist_dist = (recent_high - close[-1]) / close[-1] * 100
            data["距20日低点"] = f"{support_dist:.1f}%"
            data["距20日高点"] = f"{resist_dist:.1f}%"

            if len(close) >= 60:
                high_60d = float(np.max(close[-60:]))
                low_60d = float(np.min(close[-60:]))
                if close[-1] >= high_60d * 0.95:
                    data["60日位置"] = "接近60日新高"
                elif close[-1] <= low_60d * 1.05:
                    data["60日位置"] = "接近60日新低"
                else:
                    pct_60 = (close[-1] - low_60d) / (high_60d - low_60d) * 100
                    data["60日位置"] = f"区间{pct_60:.0f}%分位"

        return data
