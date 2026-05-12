import logging
import numpy as np
import pandas as pd
from factors.base import FactorBase, FactorResult
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)


class FundFlowFactor(FactorBase):
    name = "fund_flow"

    def __init__(self, client: AkShareClient):
        self.client = client

    @staticmethod
    def _raw_code(code: str) -> str:
        return code.replace("sz", "").replace("sh", "").replace("bj", "").strip()

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        checks: dict[str, int] = {}
        events: list[dict] = []
        diag: list[str] = []

        raw = self._raw_code(code)

        # 1) 北向资金
        try:
            nf = self.client.get_northbound_flow()
            if not nf.empty:
                checks["北向资金"] = self._score_northbound(nf)
                diag.append("北向✅")
            else:
                diag.append("北向❌无数据")
        except Exception as e:
            logger.warning("northbound error %s: %s", code, e)
            diag.append("北向❌异常")

        # 2) 融资融券
        try:
            margin = self.client.get_margin_detail()
            if not margin.empty:
                checks["融资融券"] = self._score_margin(margin, raw)
                diag.append("融资融券✅")
            else:
                diag.append("融资融券❌无数据")
        except Exception as e:
            logger.warning("margin error %s: %s", code, e)
            diag.append("融资融券❌异常")

        # 3) 个股资金流向
        try:
            ff = self.client.get_fund_flow_individual(raw)
            if not ff.empty:
                sc, ev = self._score_fund_flow(ff)
                checks["主力资金"] = sc
                events.extend(ev)
                diag.append("主力✅")
            else:
                diag.append("主力❌无数据")
                # 尝试从全市场排名中提取
                all_ff = self.client.get_all_fund_flow()
                if not all_ff.empty:
                    sc2, ev2 = self._score_all_fund_flow(all_ff, raw)
                    if sc2 != 50:
                        checks["主力资金"] = sc2
                        events.extend(ev2)
                        diag[-1] = "主力✅(全市场排名)"
        except Exception as e:
            logger.warning("fund_flow error %s: %s", code, e)
            diag.append("主力❌异常")

        # 4) 龙虎榜
        try:
            dt = self.client.get_dragon_tiger()
            if not dt.empty:
                checks["龙虎榜"] = self._score_dt(dt, raw)
                diag.append("龙虎榜✅")
            else:
                diag.append("龙虎榜❌无数据")
        except Exception as e:
            logger.warning("dragon_tiger error %s: %s", code, e)
            diag.append("龙虎榜❌异常")

        # 5) 北向个股流向
        try:
            nbi = self.client.get_northbound_individual()
            if not nbi.empty:
                nb_score = self._score_northbound_individual(nbi, raw)
                if nb_score is not None:
                    checks["北向个股"] = nb_score
                    diag.append("北向个股✅")
        except Exception as e:
            logger.warning("northbound_ind error %s: %s", code, e)

        if not checks:
            return FactorResult(
                factor_name=self.name, score=50, signal="neutral",
                detail={"数据状态": "、".join(diag)},
            )

        raw_score = np.mean(list(checks.values()))
        score = int(np.clip(raw_score, 0, 100))
        signal = "bullish" if score >= 65 else ("bearish" if score < 35 else "neutral")

        detail = dict(checks)
        detail["数据状态"] = "、".join(diag)
        return FactorResult(
            factor_name=self.name, score=score, signal=signal,
            detail=detail, events=events,
        )

    def _score_northbound(self, df) -> int:
        col = next(
            (c for c in df.columns if "净流入" in c or "净买入" in c),
            None,
        )
        if col is None:
            return 50
        recent = df.head(5)
        net = recent[col].astype(float).sum()
        return int(np.clip(50 + net / 1e9, 0, 100))

    def _score_margin(self, df, code: str) -> int:
        try:
            code_col = next((c for c in df.columns if "代码" in c or "标的" in c), None)
            if code_col is None:
                return 50
            mask = df[code_col].astype(str).str.contains(code)
            row = df[mask]
            if row.empty:
                return 50
            bal_col = next((c for c in df.columns if "余额" in c), None)
            if bal_col is None:
                return 50
            bal = float(row.iloc[0][bal_col])
            return int(np.clip(bal / 1e7, 0, 100))
        except Exception:
            return 50

    def _score_fund_flow(self, df) -> tuple[int, list[dict]]:
        events = []
        df_cols = {c.lower(): c for c in df.columns}
        main_col = (
            df_cols.get("主力净流入")
            or df_cols.get("主力净流入-净额")
            or df_cols.get("主力资金净流入")
            or next((c for c in df.columns if "主力" in c and ("净流入" in c or "净额" in c)), None)
        )
        if not main_col:
            return 50, events

        recent = df.head(5)[main_col].astype(float)
        net = recent.sum()
        score = int(np.clip(50 + net / 5e7, 0, 100))
        if net > 1e8:
            events.append({"title": f"近5日主力净流入 {net/1e8:.1f}亿", "sentiment": "positive", "impact": 3})
        elif net < -1e8:
            events.append({"title": f"近5日主力净流出 {abs(net)/1e8:.1f}亿", "sentiment": "negative", "impact": 3})

        # 超大单/大单净流入
        big_col = next((c for c in df.columns if "超大单" in c and "净" in c), None)
        if big_col:
            big_net = df.head(5)[big_col].astype(float).sum()
            if abs(big_net) > 5e7:
                direction = "流入" if big_net > 0 else "流出"
                events.append({"title": f"近5日超大单{direction} {abs(big_net)/1e8:.1f}亿", "sentiment": "positive" if big_net > 0 else "negative", "impact": 2})

        return score, events

    def _score_all_fund_flow(self, df, code: str) -> tuple[int, list[dict]]:
        """从全市场资金流向排名中提取个股数据"""
        try:
            code_col = next((c for c in df.columns if "代码" in c), None)
            if code_col is None:
                return 50, []
            mask = df[code_col].astype(str).str.contains(code)
            matched = df[mask]
            if matched.empty:
                return 50, []
            row = matched.iloc[0]
            flow_col = next(
                (c for c in df.columns if "净流入" in c or "净额" in c),
                None,
            )
            if flow_col is None:
                return 50, []
            net = float(row[flow_col])
            score = int(np.clip(50 + net / 1e7, 0, 100))
            events = []
            if abs(net) > 5e7:
                direction = "流入" if net > 0 else "流出"
                events.append({"title": f"当日主力{direction} {abs(net)/1e8:.1f}亿", "sentiment": "positive" if net > 0 else "negative", "impact": 2})
            return score, events
        except Exception:
            return 50, []

    def _score_dt(self, df, code: str) -> int:
        try:
            code_col = next((c for c in df.columns if "代码" in c), None)
            if code_col is None:
                return 50
            mask = df[code_col].astype(str).str.contains(code)
            if df[mask].empty:
                return 50
            # 在龙虎榜上：看净买入额
            net_col = next((c for c in df.columns if "净买" in c or "净额" in c), None)
            if net_col:
                net = float(df[mask].iloc[0][net_col])
                if net > 1e8:
                    return 85
                elif net > 0:
                    return 75
                elif net > -1e8:
                    return 55
                return 40
            return 75
        except Exception:
            return 50

    def _score_northbound_individual(self, df, code: str) -> int | None:
        """北向资金个股流向评分"""
        try:
            code_col = next((c for c in df.columns if "代码" in c), None)
            if code_col is None:
                return None
            mask = df[code_col].astype(str).str.contains(code)
            if df[mask].empty:
                return None
            row = df[mask].iloc[0]
            net_col = next((c for c in df.columns if "净买入" in c or "净流入" in c), None)
            if net_col is None:
                return None
            net = float(row[net_col])
            return int(np.clip(50 + net / 5e7, 0, 100))
        except Exception:
            return None
