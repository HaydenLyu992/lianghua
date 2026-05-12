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

    @staticmethod
    def _raw_code(code: str) -> str:
        return code.replace("sz", "").replace("sh", "").replace("bj", "").strip()

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        events: list[dict] = []
        checks: dict[str, int] = {}
        diag: list[str] = []

        raw = self._raw_code(code)

        # 1) 财务数据
        try:
            fin = self.client.get_financial_summary(raw)
            if not fin.empty:
                checks.update(self._score_financial(fin))
                diag.append("财务✅")
            else:
                diag.append("财务❌无数据")
        except Exception as e:
            logger.warning("financial error %s: %s", code, e)
            diag.append("财务❌异常")

        # 2) 公告
        try:
            notices = self.client.get_notices(raw)
            if not notices.empty:
                ev, sc = self._score_notices(notices)
                events.extend(ev)
                checks["公告"] = sc
                diag.append("公告✅")
            else:
                diag.append("公告❌无数据")
        except Exception as e:
            logger.warning("notices error %s: %s", code, e)
            diag.append("公告❌异常")

        # 3) 限售解禁
        try:
            restricted = self.client.get_restricted_release()
            if not restricted.empty:
                checks["限售解禁"] = self._score_restricted(restricted, raw)
                diag.append("限售解禁✅")
            else:
                diag.append("限售解禁❌无数据")
        except Exception as e:
            logger.warning("restricted error %s: %s", code, e)
            diag.append("限售解禁❌异常")

        # 4) 机构调研
        try:
            visits = self.client.get_institutional_visits(raw)
            if not visits.empty:
                checks["机构调研"] = self._score_visits(visits)
                diag.append("机构调研✅")
            else:
                diag.append("机构调研❌无数据")
        except Exception as e:
            logger.warning("visits error %s: %s", code, e)
            diag.append("机构调研❌异常")

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

    def _score_financial(self, df) -> dict[str, int]:
        checks = {}
        cols = set(df.columns)

        net_profit_col = next(
            (c for c in cols if "净利润" in c or "净利" in c or "net_profit" in c.lower()),
            None,
        )
        revenue_col = next(
            (c for c in cols if "营收" in c or "收入" in c or "营业" in c or "revenue" in c.lower()),
            None,
        )
        roe_col = next(
            (c for c in cols if "ROE" in c.upper() or "净资产收益率" in c or "净资产" in c),
            None,
        )

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

        # PE 估值（如有）
        pe_col = next((c for c in cols if "市盈" in c or "PE" in c.upper()), None)
        if pe_col:
            vals = pd.to_numeric(df[pe_col], errors="coerce").dropna()
            pe = float(vals.iloc[0]) if len(vals) > 0 else 0
            if pe > 0:
                # PE 15-25 = 70分；过高或负值扣分
                pe_score = int(np.clip(100 - abs(pe - 20) * 2, 10, 100))
                checks["市盈率"] = pe_score

        return checks

    def _score_notices(self, df) -> tuple[list[dict], int]:
        events = []
        score = 50
        positive_keywords = [
            "回购", "增持", "分红", "中标", "战略合作", "预增", "扭亏",
            "股权激励", "定向增发", "资产注入", "重组", "高送转",
        ]
        negative_keywords = [
            "减持", "亏损", "立案", "警示", "关注函", "退市", "预亏",
            "诉讼", "冻结", "质押", "爆雷", "商誉减值", "业绩变脸",
        ]

        for _, row in df.head(15).iterrows():
            title = str(row.get("title", "") or row.get("name", "") or row.get("notice_title", ""))
            if not title or title == "nan":
                continue
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
        raw = self._raw_code(code)
        mask = df["代码"].astype(str).str.contains(raw)
        upcoming = df[mask]
        if upcoming.empty:
            return 75
        # 有解禁：看解禁规模，解禁股数越多越偏空
        try:
            ratio_col = next((c for c in df.columns if "比例" in c or "占比" in c), None)
            if ratio_col:
                ratio = float(upcoming.iloc[0][ratio_col])
                if ratio < 1:
                    return 60
                elif ratio < 3:
                    return 40
                return 25
        except (ValueError, TypeError):
            pass
        return 30

    def _score_visits(self, df) -> int:
        count = len(df)
        if count > 20:
            return 85
        elif count > 10:
            return 70
        elif count > 5:
            return 60
        elif count > 0:
            return 50
        return 40
