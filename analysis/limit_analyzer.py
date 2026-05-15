import logging
import numpy as np
import pandas as pd
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)


class LimitAnalyzer:
    """涨跌停专项分析：封板强度、连板高度、涨跌停概率估算。

    概率不是固定值，而是基于：
    - 当日涨跌幅（距涨跌停还有多少空间）
    - 近期波动率（高波动 → 触及涨跌停可能性更大）
    - 市场情绪（今日涨停/跌停家数占比）
    - 该股近期是否有涨停/跌停历史
    - 技术面（RSI极端值 → 反向修正）
    """

    def __init__(self, client: AkShareClient):
        self.client = client

    async def analyze(self, code: str) -> dict:
        raw = code.replace("sz", "").replace("sh", "").replace("bj", "").strip()

        # 拉涨停/跌停池
        try:
            zt = self.client.get_limit_up_pool()
        except Exception:
            zt = pd.DataFrame()
        try:
            dt = self.client.get_limit_down_pool()
        except Exception:
            dt = pd.DataFrame()

        zt_mask = zt["代码"].astype(str).str.contains(raw) if not zt.empty else pd.Series(dtype=bool)
        dt_mask = dt["代码"].astype(str).str.contains(raw) if not dt.empty else pd.Series(dtype=bool)

        is_zt = not zt.empty and zt[zt_mask].shape[0] > 0
        is_dt = not dt.empty and dt[dt_mask].shape[0] > 0

        seal = 0.0
        streak = 0
        if is_zt:
            row = zt[zt_mask].iloc[0]
            seal = float(row.get("封单金额", 0) or 0) / 1e8
            streak = int(row.get("连板数", 1) or 1)

        # 拉实时行情 — 优先腾讯快照（极快），回退 Sina spot
        pct_chg = 0.0
        try:
            txq = self.client.get_tencent_quote(raw)
            if txq:
                pct_chg = txq.get("change_pct", 0)
            else:
                spot = self.client.get_spot_df()
                if not spot.empty:
                    spot_row = spot[spot["代码"].astype(str).str.contains(raw)]
                    if not spot_row.empty:
                        pct_col = next(
                            (c for c in spot_row.columns if "涨跌幅" in c or "pct" in c.lower()),
                            None,
                        )
                        if pct_col:
                            pct_chg = float(spot_row.iloc[0][pct_col])
        except Exception:
            pass

        # 拉K线算波动率
        volatility = 0.0
        recent_limit_hits = 0
        try:
            kline = self.client.get_daily_kline(code)
            if not kline.empty and len(kline) >= 20:
                close_col = next(
                    (c for c in kline.columns if c in ("close", "收盘")),
                    None,
                )
                if close_col:
                    closes = kline[close_col].astype(float)
                    returns = closes.pct_change().dropna()
                    if len(returns) > 0:
                        volatility = float(returns.std() * np.sqrt(252))

                    # 近20日触及涨跌停次数
                    daily_chg = closes.pct_change().dropna()
                    recent_limit_hits = int(
                        ((daily_chg.abs() > 0.09) & (daily_chg.abs() < 0.11)).sum()
                    )
        except Exception as e:
            logger.warning("K-line fetch failed in limit analyzer: %s", e)

        # 市场情绪：涨跌停家数占比
        market_zt_ratio = 0.0
        market_dt_ratio = 0.0
        try:
            spot = self.client.get_spot_df()
            if not spot.empty:
                total = len(spot)
                if total > 0:
                    market_zt_ratio = (len(zt) / total) if not zt.empty else 0.0
                    market_dt_ratio = (len(dt) / total) if not dt.empty else 0.0
        except Exception:
            pass

        zt_prob = self._estimate_zt(
            is_zt, seal, streak, pct_chg, volatility,
            recent_limit_hits, market_zt_ratio,
        )
        dt_prob = self._estimate_dt(
            is_dt, pct_chg, volatility, recent_limit_hits, market_dt_ratio,
        )

        return {
            "zt_prob": zt_prob,
            "dt_prob": dt_prob,
            "seal": round(seal, 2),
            "streak": streak,
            "is_zt": is_zt,
            "is_dt": is_dt,
        }

    def _estimate_zt(
        self,
        is_zt: bool,
        seal: float,
        streak: int,
        pct_chg: float,
        volatility: float,
        recent_hits: int,
        market_ratio: float,
    ) -> int:
        # 已涨停：高概率（次日继续涨停概率 = 连板概率）
        if is_zt:
            base = 80 + seal * 3 + streak * 2
            return int(np.clip(base, 60, 98))

        # 未涨停：基于多维因素估算
        prob = 5.0  # 基础概率

        # 距涨停空间（A股主板±10%，科创/创业±20%）
        remaining = 10.0 - pct_chg  # 默认按10%涨停
        if remaining <= 0:
            remaining = 20.0 - pct_chg  # 可能是20%涨跌停板
        if remaining <= 0:
            prob += 90
        elif remaining < 2:
            prob += 40 + (2 - remaining) * 25  # 距涨停<2%时快速上升
        elif remaining < 5:
            prob += 10 + (5 - remaining) * 3   # 距涨停2-5%温和上升
        elif remaining < 7:
            prob += remaining * 1.5             # 距涨停5-7%小幅上升
        else:
            prob += min(remaining * 0.5, 5)     # 距涨停>7%微升

        # 波动率因子：高波动 → 更容易触板
        if volatility > 0:
            # 年化波动率20%以下 = 低波，60%以上 = 高波
            vol_score = np.clip((volatility - 0.15) * 50, -5, 15)
            prob += vol_score

        # 近期触板历史：近20日有涨停记录的加分
        prob += min(recent_hits * 4, 16)

        # 市场情绪：今日全市场涨停率
        prob += market_ratio * 200  # 涨停率1%→+2分

        return int(np.clip(prob, 1, 99))

    def _estimate_dt(
        self,
        is_dt: bool,
        pct_chg: float,
        volatility: float,
        recent_hits: int,
        market_ratio: float,
    ) -> int:
        # 已跌停：高概率
        if is_dt:
            return int(np.clip(75 + abs(pct_chg + 10) * 5, 55, 98))

        prob = 3.0

        # 距跌停空间
        remaining = 10.0 + pct_chg  # pct_chg为负时，距-10%的空间
        if remaining <= 0:
            remaining = 20.0 + pct_chg
        if remaining <= 0:
            prob += 90
        elif remaining < 2:
            prob += 40 + (2 - remaining) * 25
        elif remaining < 5:
            prob += 8 + (5 - remaining) * 3
        elif remaining < 7:
            prob += remaining * 1.2
        else:
            prob += min(remaining * 0.3, 4)

        # 波动率因子
        if volatility > 0:
            prob += np.clip((volatility - 0.15) * 40, -5, 12)

        # 近期触板
        prob += min(recent_hits * 3, 12)

        # 市场情绪
        prob += market_ratio * 250

        return int(np.clip(prob, 1, 99))
