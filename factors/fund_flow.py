import logging
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
        data: dict[str, str] = {}
        events: list[dict] = []
        diag: list[str] = []

        raw = self._raw_code(code)

        # 1) 北向资金整体
        try:
            nf = self.client.get_northbound_flow()
            if not nf.empty:
                col = next((c for c in nf.columns if "净流入" in c or "净买入" in c), None)
                if col is not None:
                    recent = nf.head(5)[col].astype(float)
                    net = recent.sum()
                    data["北向资金近5日净额(亿)"] = f"{net / 1e8:.2f}"
                    diag.append("北向✅")
                else:
                    diag.append("北向❌缺字段")
            else:
                diag.append("北向❌无数据")
        except Exception as e:
            logger.warning("northbound error %s: %s", code, e)
            diag.append("北向❌异常")

        # 2) 融资融券
        try:
            margin = self.client.get_margin_detail()
            if not margin.empty:
                code_col = next((c for c in margin.columns if "代码" in c or "标的" in c), None)
                if code_col is not None:
                    mask = margin[code_col].astype(str).str.contains(raw)
                    row = margin[mask]
                    if not row.empty:
                        bal_col = next((c for c in margin.columns if "余额" in c), None)
                        if bal_col is not None:
                            data["融资融券余额"] = str(row.iloc[0][bal_col])
                            diag.append("融资融券✅")
                        else:
                            diag.append("融资融券❌缺字段")
                    else:
                        diag.append("融资融券❌无个股数据")
                else:
                    diag.append("融资融券❌缺代码列")
            else:
                diag.append("融资融券❌无数据")
        except Exception as e:
            logger.warning("margin error %s: %s", code, e)
            diag.append("融资融券❌异常")

        # 3) 个股资金流向
        try:
            ff = self.client.get_fund_flow_individual(raw)
            if not ff.empty:
                df_cols = {c.lower(): c for c in ff.columns}
                main_col = (
                    df_cols.get("主力净流入")
                    or df_cols.get("主力净流入-净额")
                    or next((c for c in ff.columns if "主力" in c and ("净流入" in c or "净额" in c)), None)
                )
                if main_col is not None:
                    recent = ff.head(5)[main_col].astype(float)
                    net = recent.sum()
                    data["近5日主力净流入(亿)"] = f"{net / 1e8:.2f}"
                    if net > 1e8:
                        events.append({"title": f"近5日主力净流入 {net/1e8:.1f}亿", "sentiment": "positive", "impact": 3})
                    elif net < -1e8:
                        events.append({"title": f"近5日主力净流出 {abs(net)/1e8:.1f}亿", "sentiment": "negative", "impact": 3})

                    big_col = next((c for c in ff.columns if "超大单" in c and "净" in c), None)
                    if big_col is not None:
                        big_net = ff.head(5)[big_col].astype(float).sum()
                        data["近5日超大单净额(亿)"] = f"{big_net / 1e8:.2f}"
                    diag.append("主力✅")
                else:
                    diag.append("主力❌缺字段")
            else:
                diag.append("主力❌无数据")
                all_ff = self.client.get_all_fund_flow()
                if not all_ff.empty:
                    code_col = next((c for c in all_ff.columns if "代码" in c), None)
                    if code_col is not None:
                        mask = all_ff[code_col].astype(str).str.contains(raw)
                        matched = all_ff[mask]
                        if not matched.empty:
                            flow_col = next((c for c in all_ff.columns if "净流入" in c or "净额" in c), None)
                            if flow_col is not None:
                                data["当日主力净流入(全市场排名)"] = str(matched.iloc[0][flow_col])
                                diag[-1] = "主力✅(全市场排名)"
        except Exception as e:
            logger.warning("fund_flow error %s: %s", code, e)
            diag.append("主力❌异常")

        # 4) 龙虎榜
        try:
            dt = self.client.get_dragon_tiger()
            if not dt.empty:
                code_col = next((c for c in dt.columns if "代码" in c), None)
                if code_col is not None:
                    mask = dt[code_col].astype(str).str.contains(raw)
                    if not dt[mask].empty:
                        row = dt[mask].iloc[0]
                        net_col = next((c for c in dt.columns if "净买" in c or "净额" in c), None)
                        if net_col is not None:
                            data["龙虎榜净买入(万)"] = str(row[net_col])
                        reason_col = next((c for c in dt.columns if "原因" in c), None)
                        if reason_col is not None:
                            data["龙虎榜上榜原因"] = str(row.get(reason_col, ""))
                        diag.append("龙虎榜✅(上榜)")
                    else:
                        diag.append("龙虎榜✅(未上榜)")
                else:
                    diag.append("龙虎榜❌缺代码列")
            else:
                diag.append("龙虎榜❌无数据")
        except Exception as e:
            logger.warning("dragon_tiger error %s: %s", code, e)
            diag.append("龙虎榜❌异常")

        # 5) 北向个股流向
        try:
            nbi = self.client.get_northbound_individual()
            if not nbi.empty:
                code_col = next((c for c in nbi.columns if "代码" in c), None)
                if code_col is not None:
                    mask = nbi[code_col].astype(str).str.contains(raw)
                    if not nbi[mask].empty:
                        row = nbi[mask].iloc[0]
                        net_col = next((c for c in nbi.columns if "净买入" in c or "净流入" in c), None)
                        if net_col is not None:
                            data["北向持股净买入"] = str(row[net_col])
                            diag.append("北向个股✅")
        except Exception as e:
            logger.warning("northbound_ind error %s: %s", code, e)

        if not data:
            return FactorResult(
                factor_name=self.name, has_data=False,
                detail={"数据状态": "、".join(diag)},
            )

        data["数据状态"] = "、".join(diag)
        return FactorResult(
            factor_name=self.name, detail=data, events=events,
        )
