import logging
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

    @staticmethod
    def _parse_pct_col(series) -> pd.Series:
        """解析含 %/万/亿 后缀的财务数据列，转为浮点数"""
        s = series.astype(str).str.replace("%", "", regex=False)
        s = s.str.replace("万", "e4", regex=False)
        s = s.str.replace("亿", "e8", regex=False)
        return pd.to_numeric(s, errors="coerce").dropna()

    async def analyze(self, code: str, name: str = "") -> FactorResult:
        data: dict[str, str] = {}
        events: list[dict] = []
        diag: list[str] = []

        raw = self._raw_code(code)

        # 1) 腾讯财经实时快照 — PE/PB/市值/换手率/涨跌停价
        try:
            txq = self.client.get_tencent_quote(raw)
            if txq:
                if txq.get("pe_ttm", 0) > 0:
                    data["PE(TTM)"] = str(txq["pe_ttm"])
                if txq.get("pe_static", 0) > 0:
                    data["PE(静态)"] = str(txq["pe_static"])
                if txq.get("pb", 0) > 0:
                    data["PB"] = str(txq["pb"])
                if txq.get("mcap_yi", 0) > 0:
                    data["总市值(亿)"] = str(txq["mcap_yi"])
                if txq.get("float_mcap_yi", 0) > 0:
                    data["流通市值(亿)"] = str(txq["float_mcap_yi"])
                if txq.get("turnover_pct", 0) > 0:
                    data["换手率"] = f"{txq['turnover_pct']}%"
                if txq.get("amount_yi", 0) > 0:
                    data["成交额(亿)"] = str(txq["amount_yi"])
                if txq.get("change_pct", 0) != 0:
                    data["涨跌幅"] = f"{txq['change_pct']}%"
                if txq.get("limit_up", 0) > 0:
                    data["涨停价"] = str(txq["limit_up"])
                if txq.get("limit_down", 0) > 0:
                    data["跌停价"] = str(txq["limit_down"])
                diag.append("实时估值✅")
            else:
                diag.append("实时估值❌无数据")
        except Exception as e:
            logger.warning("tencent quote error %s: %s", code, e)
            diag.append("实时估值❌异常")

        # 2) 财务摘要 — 净利润增速/营收增速/ROE/资产负债率/每股收益等
        try:
            fin = self.client.get_financial_summary(raw)
            if not fin.empty:
                cols = list(fin.columns)

                # 净利润同比增长率
                profit_col = next((c for c in cols if "净利润同比增长率" in c), None)
                if profit_col:
                    vals = self._parse_pct_col(fin[profit_col])
                    if len(vals) > 0:
                        data["净利润同比增速(最新)"] = f"{vals.iloc[0]:.2f}%"
                        if len(vals) >= 2:
                            data["净利润同比增速(上期)"] = f"{vals.iloc[1]:.2f}%"

                # 营业总收入同比增长率
                rev_col = next((c for c in cols if "营业总收入同比增长率" in c), None)
                if rev_col:
                    vals = self._parse_pct_col(fin[rev_col])
                    if len(vals) > 0:
                        data["营收同比增速(最新)"] = f"{vals.iloc[0]:.2f}%"
                        if len(vals) >= 2:
                            data["营收同比增速(上期)"] = f"{vals.iloc[1]:.2f}%"

                # ROE
                roe_col = next((c for c in cols if c in ("净资产收益率", "净资产收益率-摊薄")), None)
                if roe_col:
                    vals = self._parse_pct_col(fin[roe_col])
                    if len(vals) > 0 and vals.iloc[0] > 0:
                        data["ROE(最新)"] = f"{vals.iloc[0]:.2f}%"
                        if len(vals) >= 2 and vals.iloc[1] > 0:
                            data["ROE(上期)"] = f"{vals.iloc[1]:.2f}%"

                # 资产负债率
                debt_col = next((c for c in cols if "资产负债率" in c), None)
                if debt_col:
                    vals = self._parse_pct_col(fin[debt_col])
                    if len(vals) > 0:
                        data["资产负债率"] = f"{vals.iloc[0]:.2f}%"

                # 基本每股收益
                eps_col = next((c for c in cols if c == "基本每股收益"), None)
                if eps_col:
                    s = fin[eps_col].astype(str).str.replace("元", "", regex=False)
                    eps_vals = pd.to_numeric(s, errors="coerce").dropna()
                    if len(eps_vals) > 0:
                        data["基本每股收益"] = str(eps_vals.iloc[0])

                # 每股净资产
                bvps_col = next((c for c in cols if c == "每股净资产"), None)
                if bvps_col:
                    s = fin[bvps_col].astype(str).str.replace("元", "", regex=False)
                    bvps_vals = pd.to_numeric(s, errors="coerce").dropna()
                    if len(bvps_vals) > 0:
                        data["每股净资产"] = str(bvps_vals.iloc[0])

                # 每股经营现金流
                cf_col = next((c for c in cols if "每股经营现金流" in c), None)
                if cf_col:
                    s = fin[cf_col].astype(str).str.replace("元", "", regex=False)
                    cf_vals = pd.to_numeric(s, errors="coerce").dropna()
                    if len(cf_vals) > 0:
                        data["每股经营现金流"] = str(cf_vals.iloc[0])

                # 销售净利率
                margin_col = next((c for c in cols if c == "销售净利率"), None)
                if margin_col:
                    vals = self._parse_pct_col(fin[margin_col])
                    if len(vals) > 0:
                        data["销售净利率"] = f"{vals.iloc[0]:.2f}%"

                # 流动比率
                curr_col = next((c for c in cols if c == "流动比率"), None)
                if curr_col:
                    s = fin[curr_col].astype(str).str.replace("倍", "", regex=False)
                    curr_vals = pd.to_numeric(s, errors="coerce").dropna()
                    if len(curr_vals) > 0:
                        data["流动比率"] = str(curr_vals.iloc[0])

                # 速动比率
                quick_col = next((c for c in cols if c == "速动比率"), None)
                if quick_col:
                    s = fin[quick_col].astype(str).str.replace("倍", "", regex=False)
                    quick_vals = pd.to_numeric(s, errors="coerce").dropna()
                    if len(quick_vals) > 0:
                        data["速动比率"] = str(quick_vals.iloc[0])

                diag.append("财务✅")
            else:
                diag.append("财务❌无数据")
        except Exception as e:
            logger.warning("financial error %s: %s", code, e)
            diag.append("财务❌异常")

        # 3) 公告
        try:
            notices = self.client.get_notices(raw)
            if not notices.empty:
                for _, row in notices.head(15).iterrows():
                    title = str(row.get("title", "") or row.get("name", "") or row.get("notice_title", ""))
                    if not title or title == "nan":
                        continue
                    pos_kw = ["回购", "增持", "分红", "中标", "战略合作", "预增", "扭亏",
                              "股权激励", "定向增发", "资产注入", "重组", "高送转"]
                    neg_kw = ["减持", "亏损", "立案", "警示", "关注函", "退市", "预亏",
                              "诉讼", "冻结", "质押", "爆雷", "商誉减值", "业绩变脸"]
                    sentiment = "neutral"
                    if any(k in title for k in pos_kw):
                        sentiment = "positive"
                    if any(k in title for k in neg_kw):
                        sentiment = "negative"
                    events.append({"title": title, "sentiment": sentiment, "impact": 1})
                diag.append(f"公告✅({len(events)}条)")
            else:
                diag.append("公告❌无数据")
        except Exception as e:
            logger.warning("notices error %s: %s", code, e)
            diag.append("公告❌异常")

        # 4) 限售解禁
        try:
            restricted = self.client.get_restricted_release()
            if not restricted.empty:
                mask = restricted["代码"].astype(str).str.contains(raw)
                upcoming = restricted[mask]
                if not upcoming.empty:
                    row = upcoming.iloc[0]
                    ratio_col = next((c for c in restricted.columns if "比例" in c or "占比" in c), None)
                    qty_col = next((c for c in restricted.columns if "数量" in c), None)
                    date_col = next((c for c in restricted.columns if "时间" in c or "日期" in c), None)
                    if ratio_col:
                        data["近期解禁占比"] = str(row.get(ratio_col, ""))
                    if qty_col:
                        data["近期解禁数量"] = str(row.get(qty_col, ""))
                    if date_col:
                        data["解禁日期"] = str(row.get(date_col, ""))
                    diag.append("限售解禁✅(有)")
                else:
                    data["限售解禁"] = "近期无解禁"
                    diag.append("限售解禁✅(无)")
            else:
                diag.append("限售解禁❌无数据")
        except Exception as e:
            logger.warning("restricted error %s: %s", code, e)
            diag.append("限售解禁❌异常")

        # 5) 机构调研
        try:
            visits = self.client.get_institutional_visits(raw)
            if not visits.empty:
                data["近一年机构调研次数"] = str(len(visits))
                diag.append("机构调研✅")
            else:
                diag.append("机构调研❌无数据")
        except Exception as e:
            logger.warning("visits error %s: %s", code, e)
            diag.append("机构调研❌异常")

        if not data and not events:
            return FactorResult(
                factor_name=self.name, has_data=False,
                detail={"数据状态": "、".join(diag)},
            )

        data["数据状态"] = "、".join(diag)
        return FactorResult(
            factor_name=self.name, detail=data, events=events,
        )
