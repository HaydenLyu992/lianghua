import functools
import logging
import time
from typing import Optional
import pandas as pd
import akshare as ak

from config import AKSHARE_CACHE_TTL, AKSHARE_RETRY_MAX, AKSHARE_RETRY_DELAY, AKSHARE_RATE_LIMIT

logger = logging.getLogger(__name__)


def _retry(max_attempts: int = 3, base_delay: float = 1.0):
    """指数退避重试装饰器。不引入 tenacity 等额外依赖。"""
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "%s attempt %d/%d failed: %s, retrying in %.1fs",
                            func.__name__, attempt + 1, max_attempts, e, delay,
                        )
                        time.sleep(delay)
            logger.error("%s failed after %d attempts: %s", func.__name__, max_attempts, last_err)
            raise last_err
        return wrapper
    return deco


class AkShareClient:
    """AkShare 统一封装。所有 ak.xxx() 调用归口于此，上层不直接调 AkShare。
    内建重试、缓存、限流机制。"""

    def __init__(self):
        self._cache: dict[str, tuple[float, any]] = {}
        self._last_call: float = 0

    def _rate_limit(self):
        """简单限流：确保连续调用间隔 >= AKSHARE_RATE_LIMIT 秒"""
        now = time.time()
        gap = AKSHARE_RATE_LIMIT - (now - self._last_call)
        if gap > 0:
            time.sleep(gap)
        self._last_call = time.time()

    def _cached(self, key: str, ttl: int = None):
        """检查缓存是否有效。返回缓存数据或 None。"""
        if ttl is None:
            ttl = AKSHARE_CACHE_TTL
        entry = self._cache.get(key)
        if entry and (time.time() - entry[0]) < ttl:
            return entry[1]
        return None

    def _set_cache(self, key: str, data):
        self._cache[key] = (time.time(), data)

    # ---- 行情 ----

    def get_daily_kline(self, code: str) -> pd.DataFrame:
        """获取日K线。自动推断 sz/sh 前缀。"""
        clean = code.replace("sz", "").replace("sh", "")
        if clean.startswith(("0", "3")):
            code_key = f"sz{clean}"
        else:
            code_key = f"sh{clean}"
        cache_key = f"kline_{code_key}"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self._fetch_kline(code_key)
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def _fetch_kline(self, code_key: str) -> pd.DataFrame:
        return ak.stock_zh_a_hist(symbol=code_key, period="daily", adjust="qfq")

    def get_spot_df(self) -> pd.DataFrame:
        """全市场实时行情快照 DataFrame（缓存 60s）"""
        cache_key = "_spot_df"
        cached = self._cached(cache_key, ttl=60)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            df = ak.stock_zh_a_spot_em()
            self._set_cache(cache_key, df)
            return df
        except Exception:
            if cached is not None:
                return cached
            logger.warning("Failed to fetch spot data")
            return pd.DataFrame()

    def get_realtime_quote(self, code: str) -> dict:
        """实时行情快照"""
        try:
            df = self.get_spot_df()
            row = df[df["代码"] == code]
            if row.empty:
                return {}
            return row.iloc[0].to_dict()
        except Exception:
            logger.warning("Failed to fetch realtime quote for %s", code)
            return {}

    # ---- 基本面 ----

    def get_financial_summary(self, code: str) -> pd.DataFrame:
        cache_key = f"fin_{code}"
        cached = self._cached(cache_key, ttl=600)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self._fetch_financial(code)
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def _fetch_financial(self, code: str) -> pd.DataFrame:
        return ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def get_notices(self, code: str) -> pd.DataFrame:
        self._rate_limit()
        return ak.stock_notice_report(symbol=code)

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def get_restricted_release(self) -> pd.DataFrame:
        cache_key = "restricted_release"
        cached = self._cached(cache_key, ttl=3600)
        if cached is not None:
            return cached
        self._rate_limit()
        df = ak.stock_restricted_release_queue_em()
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def get_institutional_visits(self, code: str) -> pd.DataFrame:
        self._rate_limit()
        return ak.stock_jgdy_sina(symbol=code)

    # ---- 资金流向 ----

    def get_northbound_flow(self) -> pd.DataFrame:
        cache_key = "northbound_flow"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self._fetch_northbound()
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def _fetch_northbound(self) -> pd.DataFrame:
        return ak.stock_hsgt_hist_em(symbol="沪深股通")

    def get_northbound_individual(self) -> pd.DataFrame:
        """北向资金个股流向"""
        cache_key = "northbound_ind"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            df = ak.stock_hsgt_individual_em()
            self._set_cache(cache_key, df)
            return df
        except Exception:
            logger.warning("Failed to fetch northbound individual data")
            if cached is not None:
                return cached
            return pd.DataFrame()

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def get_margin_detail(self) -> pd.DataFrame:
        cache_key = "margin_detail"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()
        df = ak.stock_margin_detail_sse()
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def get_fund_flow_individual(self, code: str) -> pd.DataFrame:
        self._rate_limit()
        return ak.stock_fund_flow_individual(symbol=code)

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def get_dragon_tiger(self) -> pd.DataFrame:
        cache_key = "dragon_tiger"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()
        df = ak.stock_lhb_detail_em()
        self._set_cache(cache_key, df)
        return df

    def get_all_fund_flow(self) -> pd.DataFrame:
        """全市场个股资金流向"""
        cache_key = "all_fund_flow"
        cached = self._cached(cache_key, ttl=120)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            df = ak.stock_fund_flow_individual(symbol="all")
            self._set_cache(cache_key, df)
            return df
        except Exception:
            logger.warning("Failed to fetch all fund flow")
            if cached is not None:
                return cached
            return pd.DataFrame()

    # ---- 情绪 ----

    def get_limit_up_pool(self) -> pd.DataFrame:
        """涨停池"""
        cache_key = "limit_up"
        cached = self._cached(cache_key, ttl=60)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            if hasattr(ak, 'stock_zt_pool_em'):
                df = ak.stock_zt_pool_em(date=None)
                self._set_cache(cache_key, df)
                return df
        except Exception:
            logger.warning("Failed to fetch limit up pool")
        if cached is not None:
            return cached
        return pd.DataFrame()

    def get_limit_down_pool(self) -> pd.DataFrame:
        """跌停池"""
        cache_key = "limit_down"
        cached = self._cached(cache_key, ttl=60)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            if hasattr(ak, 'stock_zt_pool_dtgc_em'):
                df = ak.stock_zt_pool_dtgc_em(date=None)
                self._set_cache(cache_key, df)
                return df
        except Exception:
            logger.warning("Failed to fetch limit down pool")
        if cached is not None:
            return cached
        return pd.DataFrame()

    def get_hot_rank(self) -> pd.DataFrame:
        """个股热度排名"""
        cache_key = "hot_rank"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            if hasattr(ak, 'stock_hot_rank_em'):
                df = ak.stock_hot_rank_em()
                self._set_cache(cache_key, df)
                return df
        except Exception:
            logger.warning("Failed to fetch hot rank")
        if cached is not None:
            return cached
        return pd.DataFrame()

    # ---- 行业 ----

    def get_industry_index(self) -> pd.DataFrame:
        cache_key = "industry_index"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self._fetch_industry()
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def _fetch_industry(self) -> pd.DataFrame:
        return ak.stock_board_industry_index_ths()

    # ---- 宏观 ----

    def get_pmi(self) -> pd.DataFrame:
        cache_key = "pmi"
        cached = self._cached(cache_key, ttl=3600)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self._fetch_pmi()
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def _fetch_pmi(self) -> pd.DataFrame:
        return ak.macro_china_pmi()

    def get_cpi(self) -> pd.DataFrame:
        cache_key = "cpi"
        cached = self._cached(cache_key, ttl=3600)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self._fetch_cpi()
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def _fetch_cpi(self) -> pd.DataFrame:
        return ak.macro_china_cpi_yearly()

    def get_money_supply(self) -> pd.DataFrame:
        cache_key = "money_supply"
        cached = self._cached(cache_key, ttl=3600)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self._fetch_m2()
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def _fetch_m2(self) -> pd.DataFrame:
        return ak.macro_china_money_supply()

    def get_lpr(self) -> pd.DataFrame:
        cache_key = "lpr"
        cached = self._cached(cache_key, ttl=3600)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self._fetch_lpr()
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def _fetch_lpr(self) -> pd.DataFrame:
        return ak.macro_china_lpr()
