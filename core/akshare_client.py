import logging
from typing import Optional
import pandas as pd
import akshare as ak

from config import AKSHARE_CACHE_TTL

logger = logging.getLogger(__name__)


class AkShareClient:
    """AkShare 统一封装。所有 ak.xxx() 调用归口于此，上层不直接调 AkShare。"""

    def __init__(self):
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}

    # ---- 行情 ----

    def get_daily_kline(self, code: str) -> pd.DataFrame:
        """获取日K线。自动推断 sz/sh 前缀。"""
        clean = code.replace("sz", "").replace("sh", "")
        # 深交所：000xxx, 001xxx, 002xxx, 003xxx, 300xxx, 301xxx
        if clean.startswith(("0", "3")):
            code_key = f"sz{clean}"
        else:
            code_key = f"sh{clean}"
        return ak.stock_zh_a_hist(symbol=code_key, period="daily", adjust="qfq")

    def get_realtime_quote(self, code: str) -> dict:
        """实时行情快照"""
        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == code]
            if row.empty:
                return {}
            return row.iloc[0].to_dict()
        except Exception:
            logger.warning("Failed to fetch realtime quote for %s", code)
            return {}

    # ---- 基本面 ----

    def get_financial_summary(self, code: str) -> pd.DataFrame:
        return ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")

    def get_notices(self, code: str) -> pd.DataFrame:
        return ak.stock_notice_report(symbol=code)

    def get_restricted_release(self) -> pd.DataFrame:
        return ak.stock_restricted_release_queue_em()

    def get_institutional_visits(self, code: str) -> pd.DataFrame:
        return ak.stock_jgdy_sina(symbol=code)

    # ---- 资金流向 ----

    def get_northbound_flow(self) -> pd.DataFrame:
        return ak.stock_hsgt_hist_em(symbol="沪深股通")

    def get_northbound_individual(self) -> pd.DataFrame:
        """北向资金个股流向"""
        try:
            return ak.stock_hsgt_individual_em()
        except Exception:
            logger.warning("Failed to fetch northbound individual data")
            return pd.DataFrame()

    def get_margin_detail(self) -> pd.DataFrame:
        return ak.stock_margin_detail_sse()

    def get_fund_flow_individual(self, code: str) -> pd.DataFrame:
        return ak.stock_fund_flow_individual(symbol=code)

    def get_dragon_tiger(self) -> pd.DataFrame:
        return ak.stock_lhb_detail_em()

    def get_all_fund_flow(self) -> pd.DataFrame:
        """全市场个股资金流向"""
        return ak.stock_fund_flow_individual(symbol="all")

    # ---- 情绪 ----

    def get_limit_up_pool(self) -> pd.DataFrame:
        """涨停池"""
        return ak.stock_zt_pool_em(date=None) if hasattr(ak, 'stock_zt_pool_em') else pd.DataFrame()

    def get_limit_down_pool(self) -> pd.DataFrame:
        """跌停池"""
        return ak.stock_zt_pool_dtgc_em(date=None) if hasattr(ak, 'stock_zt_pool_dtgc_em') else pd.DataFrame()

    def get_hot_rank(self) -> pd.DataFrame:
        """个股热度排名"""
        return ak.stock_hot_rank_em() if hasattr(ak, 'stock_hot_rank_em') else pd.DataFrame()

    # ---- 行业 ----

    def get_industry_index(self) -> pd.DataFrame:
        return ak.stock_board_industry_index_ths()

    # ---- 宏观 ----

    def get_pmi(self) -> pd.DataFrame:
        return ak.macro_china_pmi()

    def get_cpi(self) -> pd.DataFrame:
        return ak.macro_china_cpi_yearly()

    def get_money_supply(self) -> pd.DataFrame:
        return ak.macro_china_money_supply()

    def get_lpr(self) -> pd.DataFrame:
        return ak.macro_china_lpr()
