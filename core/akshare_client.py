import functools
import logging
import time
from typing import Optional
import pandas as pd
import akshare as ak
import requests
from requests.adapters import HTTPAdapter

from config import AKSHARE_CACHE_TTL, AKSHARE_RETRY_MAX, AKSHARE_RETRY_DELAY, AKSHARE_RATE_LIMIT

logger = logging.getLogger(__name__)

# ── 东方财富反爬虫对抗 ──
# 东方财富检测以下特征：
# 1. User-Agent 缺失或为默认 Python 值 → 直接 RST
# 2. Accept / Accept-Language / Referer 缺失 → 返回空数据
# 3. Sec-Fetch 系列头缺失 → JS Challenge
# 4. TLS 指纹 (JA3) 不匹配 → 连接拒绝
# 5. 请求频率过高 → IP 封禁
#
# 绕过策略：
# - 注入 Chrome 浏览器特征头
# - 对 eastmoney.com 域名添加 Referer
# - 使用 Tencent/Sina 数据源作为优先/备用源
# - 缓存 + 限流降低请求频率

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_EM_DOMAINS = ("eastmoney.com", "push2.eastmoney", "push2his.eastmoney", "emappdata.eastmoney")

_original_requests_get = None
_original_requests_post = None
_original_request_with_retry = None


def _is_eastmoney_url(url: str) -> bool:
    return any(d in url for d in _EM_DOMAINS)


def _patched_get(url, params=None, **kwargs):
    """全局 requests.get 补丁：对东方财富域名注入浏览器头。"""
    if _is_eastmoney_url(url):
        headers = kwargs.pop("headers", {}) or {}
        h = dict(_BROWSER_HEADERS)
        h["Referer"] = "https://data.eastmoney.com/"
        h.update(headers)
        kwargs["headers"] = h
    return _original_requests_get(url, params=params, **kwargs)


def _patched_post(url, data=None, json=None, **kwargs):
    """全局 requests.post 补丁：对东方财富域名注入浏览器头。"""
    if _is_eastmoney_url(url):
        headers = kwargs.pop("headers", {}) or {}
        h = dict(_BROWSER_HEADERS)
        h["Referer"] = "https://data.eastmoney.com/"
        h.update(headers)
        kwargs["headers"] = h
    return _original_requests_post(url, data=data, json=json, **kwargs)


def _patch_all():
    """三层补丁：requests.get/post + request_with_retry，全覆盖 AkShare 的 HTTP 调用。"""
    global _original_requests_get, _original_requests_post, _original_request_with_retry

    if _original_requests_get is None:
        _original_requests_get = requests.get
        _original_requests_post = requests.post
        requests.get = _patched_get
        requests.post = _patched_post
        logger.info("requests.get/post patched with browser headers for East Money domains")

    try:
        from akshare.utils import request as ak_req
        if _original_request_with_retry is None:
            _original_request_with_retry = ak_req.request_with_retry

            @functools.wraps(_original_request_with_retry)
            def _patched_retry(url, params=None, timeout=15, max_retries=3,
                               base_delay=1.0, random_delay_range=(0.5, 1.5)):
                import random
                last_exception = None
                for attempt in range(max_retries):
                    try:
                        with requests.Session() as session:
                            session.headers.update(_BROWSER_HEADERS)
                            if _is_eastmoney_url(url):
                                session.headers["Referer"] = "https://data.eastmoney.com/"
                            adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
                            session.mount("http://", adapter)
                            session.mount("https://", adapter)
                            response = session.get(url, params=params, timeout=timeout)
                            response.raise_for_status()
                            return response
                    except (requests.RequestException, ValueError) as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(*random_delay_range)
                            time.sleep(delay)
                raise last_exception

            ak_req.request_with_retry = _patched_retry
            logger.info("akshare request_with_retry patched")
    except ImportError:
        logger.warning("Cannot patch akshare.utils.request")


_patch_all()


# ── 重试装饰器 ──

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

    反反爬策略：
    1. 东方财富(_em) 接口请求已通过 monkey-patch 注入浏览器特征头
    2. 关键接口(如实时行情)优先用非东方财富数据源(Sina)兜底
    3. 内建重试、缓存、限流机制
    """

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

    # ── 行情 ──

    def get_daily_kline(self, code: str) -> pd.DataFrame:
        """获取日K线。自动推断 sz/sh 前缀。"""
        clean = code.replace("sz", "").replace("sh", "")
        if clean.startswith(("0", "3", "8")):
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
        # 优先腾讯数据源（push2his.eastmoney.com 做 TLS 阻断，腾讯可用）
        # 腾讯接口需要 sz/sh 前缀
        try:
            return ak.stock_zh_a_hist_tx(symbol=code_key, adjust="qfq")
        except Exception:
            clean = code_key.replace("sz", "").replace("sh", "")
            try:
                return ak.stock_zh_a_hist(symbol=code_key, period="daily", adjust="qfq")
            except Exception:
                return ak.stock_zh_a_hist(symbol=clean, period="daily", adjust="qfq")

    def get_spot_df(self) -> pd.DataFrame:
        """全市场实时行情快照 DataFrame（缓存 60s）。
        优先使用新浪数据源，东方财富被反爬时自动降级。
        """
        cache_key = "_spot_df"
        cached = self._cached(cache_key, ttl=60)
        if cached is not None:
            return cached

        self._rate_limit()

        # 优先用新浪（非东方财富源）
        for source_name, fetch_fn in [
            ("sina", self._fetch_spot_sina),
            ("em", self._fetch_spot_em),
        ]:
            try:
                df = fetch_fn()
                if not df.empty:
                    self._set_cache(cache_key, df)
                    return df
                logger.warning("spot %s returned empty", source_name)
            except Exception as e:
                logger.warning("spot %s failed: %s", source_name, e)

        if cached is not None:
            return cached
        return pd.DataFrame()

    def _fetch_spot_sina(self) -> pd.DataFrame:
        return ak.stock_zh_a_spot()

    def _fetch_spot_em(self) -> pd.DataFrame:
        return ak.stock_zh_a_spot_em()

    def get_realtime_quote(self, code: str) -> dict:
        """实时行情快照。兼容纯数字代码和带前缀代码两种格式。"""
        try:
            df = self.get_spot_df()
            row = df[df["代码"] == code]
            if row.empty:
                # Sina 数据源代码带 sh/sz/bj 前缀，尝试前缀匹配
                clean = code.replace("sz", "").replace("sh", "").replace("bj", "")
                row = df[df["代码"].astype(str).str[-6:] == clean[-6:]]
            if row.empty:
                return {}
            return row.iloc[0].to_dict()
        except Exception:
            logger.warning("Failed to fetch realtime quote for %s", code)
            return {}

    # ── 基本面 ──

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

    def get_restricted_release(self) -> pd.DataFrame:
        cache_key = "restricted_release"
        cached = self._cached(cache_key, ttl=3600)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self._fetch_restricted()
        self._set_cache(cache_key, df)
        return df

    def _fetch_restricted(self) -> pd.DataFrame:
        # 东方财富源优先，新浪兜底
        try:
            df = ak.stock_restricted_release_queue_em()
            if not df.empty:
                return df
        except Exception as e:
            logger.warning("restricted_release_em failed: %s, trying sina", e)
        try:
            return ak.stock_restricted_release_queue_sina()
        except Exception:
            return pd.DataFrame()

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def get_institutional_visits(self, code: str) -> pd.DataFrame:
        self._rate_limit()
        return ak.stock_jgdy_sina(symbol=code)

    # ── 资金流向 ──

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
        # 无参调用即可获取全部北向资金历史
        return ak.stock_hsgt_hist_em()

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

    def get_fund_flow_individual(self, code: str) -> pd.DataFrame:
        """个股资金流向。东方财富接口（列结构偶有变更，做保护）。"""
        cache_key = f"ff_{code}"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()

        try:
            df = ak.stock_fund_flow_individual(symbol=code)
            if not df.empty:
                self._set_cache(cache_key, df)
                return df
        except ValueError as e:
            # AkShare 内部列名数量与数据不匹配（东方财富改版导致）
            logger.warning("fund_flow_individual(%s) schema mismatch: %s", code, e)
        except Exception as e:
            logger.warning("fund_flow_individual(%s) failed: %s", code, e)

        if cached is not None:
            return cached
        return pd.DataFrame()

    def get_dragon_tiger(self) -> pd.DataFrame:
        cache_key = "dragon_tiger"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()
        df = self._fetch_dragon_tiger()
        self._set_cache(cache_key, df)
        return df

    def _fetch_dragon_tiger(self) -> pd.DataFrame:
        try:
            df = ak.stock_lhb_detail_em()
            if not df.empty:
                return df
        except Exception as e:
            logger.warning("lhb_detail_em failed: %s", e)
        try:
            return ak.stock_lhb_detail_daily_sina()
        except Exception:
            return pd.DataFrame()

    def get_all_fund_flow(self) -> pd.DataFrame:
        """全市场个股资金流向排行榜。"""
        cache_key = "all_fund_flow"
        cached = self._cached(cache_key, ttl=120)
        if cached is not None:
            return cached
        self._rate_limit()

        # 优先用个股资金流排名（东方财富接口，现在有浏览器头）
        for fn_name in ["stock_individual_fund_flow_rank", "stock_fund_flow_individual"]:
            try:
                fn = getattr(ak, fn_name)
                if fn_name == "stock_fund_flow_individual":
                    df = fn(symbol="all")
                else:
                    df = fn()
                if not df.empty:
                    self._set_cache(cache_key, df)
                    return df
            except Exception as e:
                logger.warning("%s failed: %s", fn_name, e)

        if cached is not None:
            return cached
        return pd.DataFrame()

    # ── 情绪 ──

    def get_limit_up_pool(self) -> pd.DataFrame:
        """涨停池"""
        cache_key = "limit_up"
        cached = self._cached(cache_key, ttl=60)
        if cached is not None:
            return cached
        self._rate_limit()
        from datetime import date
        today = date.today().strftime("%Y%m%d")
        try:
            if hasattr(ak, 'stock_zt_pool_em'):
                df = ak.stock_zt_pool_em(date=today)
                if not df.empty:
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
        from datetime import date
        today = date.today().strftime("%Y%m%d")
        try:
            if hasattr(ak, 'stock_zt_pool_dtgc_em'):
                df = ak.stock_zt_pool_dtgc_em(date=today)
                if not df.empty:
                    self._set_cache(cache_key, df)
                    return df
        except Exception:
            logger.warning("Failed to fetch limit down pool")
        if cached is not None:
            return cached
        return pd.DataFrame()

    def get_hot_rank(self) -> pd.DataFrame:
        """个股热度排名。纯东方财富接口，被拒时优雅降级。"""
        cache_key = "hot_rank"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            if hasattr(ak, 'stock_hot_rank_em'):
                df = ak.stock_hot_rank_em()
                if not df.empty:
                    self._set_cache(cache_key, df)
                    return df
        except Exception:
            logger.warning("Failed to fetch hot rank (East Money may be blocking)")
        if cached is not None:
            return cached
        return pd.DataFrame()

    # ── 行业 ──

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

    # ── 宏观 ──

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
