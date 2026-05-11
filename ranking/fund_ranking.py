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
        for candidate in ["主力净流入", "主力净流入-净额", "主力资金净流入", "main_net_inflow"]:
            if candidate.lower() in cols:
                return cols[candidate.lower()]
        return None

    def _format_rows(self, df, main_col: str) -> list[dict]:
        results = []
        for _, row in df.iterrows():
            name_col = next((c for c in row.index if "名称" in c or "name" in c.lower()), None)
            code_col = next((c for c in row.index if "代码" in c or "code" in c.lower()), None)
            results.append({
                "code": str(row.get(code_col, "")),
                "name": str(row.get(name_col, "")),
                "main_net": float(row.get(main_col, 0)),
            })
        return results
