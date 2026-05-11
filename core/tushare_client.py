import logging
from typing import Optional
import pandas as pd
import tushare as ts

from config import TUSHARE_TOKEN

logger = logging.getLogger(__name__)


class TushareClient:
    """Tushare 封装 — 仅在配置了 token 时启用，用于拉取高质量历史数据。"""

    def __init__(self):
        self._pro: Optional[ts.pro_api] = None
        if TUSHARE_TOKEN:
            ts.set_token(TUSHARE_TOKEN)
            self._pro = ts.pro_api()

    @property
    def available(self) -> bool:
        return self._pro is not None

    def get_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        if not self._pro:
            return pd.DataFrame()
        return self._pro.daily(ts_code=code, start_date=start, end_date=end)

    def get_income(self, code: str) -> pd.DataFrame:
        if not self._pro:
            return pd.DataFrame()
        return self._pro.income(ts_code=code)

    def get_balance_sheet(self, code: str) -> pd.DataFrame:
        if not self._pro:
            return pd.DataFrame()
        return self._pro.balancesheet(ts_code=code)

    def get_margin(self, code: str) -> pd.DataFrame:
        if not self._pro:
            return pd.DataFrame()
        return self._pro.margin_detail(ts_code=code)
