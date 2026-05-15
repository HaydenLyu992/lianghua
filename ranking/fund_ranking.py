import logging
from datetime import datetime, time
import pandas as pd
import numpy as np
from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)


class FundRanking:
    """资金流向排行榜：从 AkShare 拉全市场数据，排序取 Top N。"""

    def __init__(self, client: AkShareClient):
        self.client = client

    def refresh(self, top_n: int = 50) -> tuple[list[dict], list[dict]]:
        """返回 (inflow_top, outflow_top)"""
        try:
            df = self.client.get_all_fund_flow()
            if df.empty:
                return [], []
        except Exception as e:
            logger.warning("Failed to fetch fund flow data: %s", e)
            return [], []

        # 计算主力净流入
        main_col = self._find_main_col(df)
        if not main_col:
            return [], []

        df["main_net"] = pd.to_numeric(df[main_col], errors="coerce").fillna(0)
        sorted_df = df.sort_values("main_net", ascending=False)

        inflow = self._format_rows(sorted_df.head(top_n), "main_net")
        outflow = self._format_rows(sorted_df.tail(top_n).iloc[::-1], "main_net")
        return inflow, outflow

    def _find_main_col(self, df) -> str | None:
        cols = {c.lower(): c for c in df.columns}
        for candidate in ["主力净流入-净额", "主力净流入", "主力资金净流入", "净额", "main_net_inflow"]:
            if candidate.lower() in cols:
                return cols[candidate.lower()]
        return None

    def refresh_surge(self, top_n: int = 50) -> list[dict]:
        """飙升榜：全市场涨幅最大个股。"""
        try:
            df = self.client.get_surge_board(top_n)
            if df.empty:
                return []
        except Exception as e:
            logger.warning("Failed to fetch surge board: %s", e)
            return []

        pct_col = self.client._find_pct_col(df)
        if not pct_col:
            return []

        results = []
        for _, row in df.iterrows():
            name_col = self._find_name_col(row)
            code_col = self._find_code_col(row)
            price_col = next((c for c in row.index if "最新价" in c or "price" in c.lower()), None)
            pct_val = float(row.get(pct_col, 0))
            price_val = row.get(price_col, None)
            results.append({
                "code": str(row.get(code_col, "")),
                "name": str(row.get(name_col, "")),
                "change_pct": pct_val,
                "price": float(price_val) if price_val is not None else None,
                "display": f"{pct_val:+.2f}%",
            })
        return results

    def _find_name_col(self, row) -> str | None:
        return next((c for c in row.index if "名称" in c or "简称" in c or "name" in c.lower()), None)

    def _find_code_col(self, row) -> str | None:
        return next((c for c in row.index if "代码" in c or "code" in c.lower()), None)

    def _format_rows(self, df, main_col: str) -> list[dict]:
        results = []
        cols_lower = {c.lower(): c for c in df.columns}
        for _, row in df.iterrows():
            name_col = self._find_name_col(row)
            code_col = self._find_code_col(row)
            main_val = float(row.get(main_col, 0))
            results.append({
                "code": str(row.get(code_col, "")),
                "name": str(row.get(name_col, "")),
                "main_net": main_val,
                "display": f"{main_val:+,.0f}",
                "super_large_net": self._safe_float(row, cols_lower, "超大单"),
                "large_net": self._safe_float(row, cols_lower, "大单"),
                "medium_net": self._safe_float(row, cols_lower, "中单"),
                "small_net": self._safe_float(row, cols_lower, "小单"),
            })
        return results

    def _safe_float(self, row, cols_lower: dict, keyword: str) -> float | None:
        kw = keyword.lower()
        col = cols_lower.get(kw)
        if col is None:
            # 模糊匹配：列名包含关键词 + "净流入" 或 "净额"
            for ck, cv in cols_lower.items():
                if kw in ck and ("净流入" in ck or "净额" in ck):
                    col = cv
                    break
        if col:
            try:
                return float(row.get(col, 0))
            except (ValueError, TypeError):
                pass
        return None
