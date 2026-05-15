import functools
import logging
import threading
import time
from typing import Optional
import pandas as pd
import akshare as ak

from config import AKSHARE_CACHE_TTL, AKSHARE_RETRY_MAX, AKSHARE_RETRY_DELAY, AKSHARE_RATE_LIMIT

logger = logging.getLogger(__name__)


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
    内建缓存、限流、重试机制，优先使用 Sina/Tencent 等稳定性更好的数据源。
    """

    def __init__(self):
        self._cache: dict[str, tuple[float, any]] = {}
        self._last_call: float = 0
        self._cache_lock = threading.Lock()
        self._rate_lock = threading.Lock()

    def _rate_limit(self):
        """简单限流：确保连续调用间隔 >= AKSHARE_RATE_LIMIT 秒（线程安全）。"""
        with self._rate_lock:
            now = time.time()
            gap = AKSHARE_RATE_LIMIT - (now - self._last_call)
            if gap > 0:
                time.sleep(gap)
            self._last_call = time.time()

    def _cached(self, key: str, ttl: int = None):
        """检查缓存是否有效。返回缓存数据或 None。"""
        if ttl is None:
            ttl = AKSHARE_CACHE_TTL
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and (time.time() - entry[0]) < ttl:
                return entry[1]
            return None

    def _set_cache(self, key: str, data):
        with self._cache_lock:
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

    def get_stock_list(self) -> pd.DataFrame:
        """轻量级股票列表（仅代码+名称），缓存 1 小时，用于搜索建议。"""
        cache_key = "_stock_list"
        cached = self._cached(cache_key, ttl=3600)
        if cached is not None:
            return cached

        self._rate_limit()

        for source_name, fetch_fn in [
            ("sina", self._fetch_spot_sina),
            ("em", self._fetch_spot_em),
        ]:
            try:
                df = fetch_fn()
                if not df.empty:
                    code_col = next((c for c in df.columns if "代码" in c), None)
                    name_col = next((c for c in df.columns if "名称" in c or "简称" in c), None)
                    if code_col and name_col:
                        result = df[[code_col, name_col]].copy()
                        result = result.rename(columns={code_col: "代码", name_col: "名称"})
                        self._set_cache(cache_key, result)
                        return result
            except Exception as e:
                logger.warning("stock_list %s failed: %s", source_name, e)

        if cached is not None:
            return cached
        return pd.DataFrame()

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

    def get_tencent_quote(self, code: str) -> dict:
        """腾讯财经实时快照——极快（<500ms），提供PE/PB/市值/涨跌停价。"""
        import urllib.request

        clean = code.replace("sz", "").replace("sh", "").replace("bj", "").strip()
        if not clean.isdigit():
            return {}
        cache_key = f"txq_{clean}"
        cached = self._cached(cache_key, ttl=60)
        if cached is not None:
            return cached
        try:
            prefix = "sh" if clean.startswith(("6", "9")) else "bj" if clean.startswith("8") else "sz"
            url = f"https://qt.gtimg.cn/q={prefix}{clean}"
            req = urllib.request.urlopen(url, timeout=5)
            raw = req.read().decode("gbk")
            if "=" not in raw or '"' not in raw:
                return {}
            vals = raw.split('"')[1].split("~")
            if len(vals) < 53:
                return {}
            result = {
                "name": vals[1],
                "price": float(vals[3]) if vals[3] else 0,
                "last_close": float(vals[4]) if vals[4] else 0,
                "open": float(vals[5]) if vals[5] else 0,
                "change_pct": float(vals[32]) if vals[32] else 0,
                "high": float(vals[33]) if vals[33] else 0,
                "low": float(vals[34]) if vals[34] else 0,
                "turnover_pct": float(vals[38]) if vals[38] else 0,
                "pe_ttm": float(vals[39]) if vals[39] else 0,
                "mcap_yi": float(vals[44]) if vals[44] else 0,
                "float_mcap_yi": float(vals[45]) if vals[45] else 0,
                "pb": float(vals[46]) if vals[46] else 0,
                "limit_up": float(vals[47]) if vals[47] else 0,
                "limit_down": float(vals[48]) if vals[48] else 0,
                "pe_static": float(vals[52]) if vals[52] else 0,
                "volume": float(vals[6]) if vals[6] else 0,
                "amount_yi": float(vals[37]) if vals[37] else 0,
            }
            self._set_cache(cache_key, result)
            return result
        except Exception:
            logger.warning("Tencent quote failed for %s", code)
            return {}

    # ── 基本面 ──

    def get_financial_summary(self, code: str) -> pd.DataFrame:
        cache_key = f"fin_{code}"
        cached = self._cached(cache_key, ttl=600)
        if cached is not None:
            return cached
        raw_code = code.replace("sz", "").replace("sh", "").replace("bj", "").strip()
        if not raw_code.isdigit():
            return pd.DataFrame()
        self._rate_limit()
        df = self._fetch_financial(code)
        self._set_cache(cache_key, df)
        return df

    @_retry(max_attempts=AKSHARE_RETRY_MAX, base_delay=AKSHARE_RETRY_DELAY)
    def _fetch_financial(self, code: str) -> pd.DataFrame:
        return ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")

    def get_notices(self, code: str) -> pd.DataFrame:
        """获取个股公告。每日全量公告共享缓存，按代码过滤。"""
        raw_code = code.replace("sz", "").replace("sh", "").replace("bj", "").strip()
        if not raw_code.isdigit():
            return pd.DataFrame()

        cache_key = f"notices_{code}"
        cached = self._cached(cache_key, ttl=1800)
        if cached is not None:
            return cached

        # 先尝试从当日全量公告缓存中过滤（避免重复拉取）
        today = pd.Timestamp.now().strftime("%Y%m%d")
        daily_key = f"notices_all_{today}"
        all_notices = self._cached(daily_key, ttl=3600)
        if all_notices is not None:
            if not all_notices.empty:
                mask = all_notices["代码"].astype(str).str.contains(raw_code)
                result = all_notices[mask]
                self._set_cache(cache_key, result)
                return result
        else:
            self._rate_limit()
            for attempt in range(AKSHARE_RETRY_MAX):
                try:
                    all_notices = ak.stock_notice_report(symbol="全部", date=today)
                    if not all_notices.empty:
                        self._set_cache(daily_key, all_notices)
                        mask = all_notices["代码"].astype(str).str.contains(raw_code)
                        result = all_notices[mask]
                        self._set_cache(cache_key, result)
                        return result
                except Exception as e:
                    delay = AKSHARE_RETRY_DELAY * (2 ** attempt)
                    logger.warning("get_notices attempt %d/%d failed: %s, retrying in %.1fs",
                                   attempt + 1, AKSHARE_RETRY_MAX, e, delay)
                    time.sleep(delay)

        logger.warning("get_notices failed after %d attempts: %s", AKSHARE_RETRY_MAX, code)
        if cached is not None:
            return cached
        return pd.DataFrame()

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
        # 新浪源优先（有"代码"列），东方财富源兜底
        try:
            df = ak.stock_restricted_release_queue_sina()
            if not df.empty:
                return df
        except Exception as e:
            logger.warning("restricted_release_sina failed: %s, trying em", e)
        try:
            return ak.stock_restricted_release_queue_em()
        except Exception:
            return pd.DataFrame()

    def get_institutional_visits(self, code: str) -> pd.DataFrame:
        """个股机构调研。stock_jgdy_detail_em 拉全量后按代码过滤。"""
        raw_code = code.replace("sz", "").replace("sh", "").replace("bj", "").strip()
        if not raw_code.isdigit():
            return pd.DataFrame()

        cache_key = f"visits_{code}"
        cached = self._cached(cache_key, ttl=3600)
        if cached is not None:
            return cached

        self._rate_limit()

        for attempt in range(AKSHARE_RETRY_MAX):
            try:
                date_str = (pd.Timestamp.now() - pd.Timedelta(days=180)).strftime("%Y%m%d")
                df = ak.stock_jgdy_detail_em(date=date_str)
                if not df.empty:
                    code_col = next((c for c in df.columns if "代码" in c), None)
                    if code_col:
                        mask = df[code_col].astype(str).str.contains(raw_code)
                        result = df[mask]
                        self._set_cache(cache_key, result)
                        return result
                self._set_cache(cache_key, pd.DataFrame())
                return pd.DataFrame()
            except TypeError:
                # API 返回空数据，重试无意义
                self._set_cache(cache_key, pd.DataFrame())
                return pd.DataFrame()
            except Exception as e:
                delay = AKSHARE_RETRY_DELAY * (2 ** attempt)
                logger.warning("get_institutional_visits attempt %d/%d failed: %s, retrying in %.1fs",
                               attempt + 1, AKSHARE_RETRY_MAX, e, delay)
                time.sleep(delay)

        logger.warning("get_institutional_visits failed after %d attempts: %s", AKSHARE_RETRY_MAX, code)
        if cached is not None:
            return cached
        return pd.DataFrame()

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
        """个股资金流向。symbol 参数是排行周期而非代码，需拉全量后过滤。"""
        cache_key = f"ff_{code}"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached

        raw_code = code.replace("sz", "").replace("sh", "").replace("bj", "").strip()

        try:
            df = self.get_all_fund_flow()
            if df.empty:
                return pd.DataFrame()

            code_col = next((c for c in df.columns if "代码" in c), None)
            if code_col:
                mask = df[code_col].astype(str).str.contains(raw_code)
                result = df[mask]
                self._set_cache(cache_key, result)
                return result
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
        """全市场个股资金流向排行榜（今日主力净流入排名，通过 AkShare 获取）。"""
        cache_key = "all_fund_flow"
        cached = self._cached(cache_key, ttl=300)
        if cached is not None:
            return cached
        self._rate_limit()

        try:
            df = ak.stock_fund_flow_individual(symbol="即时")
            if df is not None and not df.empty:
                self._set_cache(cache_key, df)
                return df
        except Exception as e:
            logger.warning("AkShare fund flow fetch failed: %s", e)

        if cached is not None:
            return cached
        return pd.DataFrame()

    def get_5day_change_batch(self, codes: list[str]) -> dict[str, float | None]:
        """批量获取多只股票的5日涨跌幅。基于K线缓存，首次较慢后续命中缓存即时返回。"""
        result: dict[str, float | None] = {}
        for code in codes:
            result[code] = self._calc_5day_change(code)
        return result

    def _calc_5day_change(self, code: str) -> float | None:
        """基于K线收盘价计算个股5日涨跌幅。"""
        try:
            df = self.get_daily_kline(code)
            if df.empty:
                return None
            close_col = None
            for c in df.columns:
                if c.lower() in ("close", "收盘"):
                    close_col = c
                    break
            if close_col is None:
                close_col = df.columns[min(4, len(df.columns) - 1)]
            closes = pd.to_numeric(df[close_col], errors="coerce").dropna()
            if len(closes) < 6:
                return None
            today_close = closes.iloc[-1]
            ref_close = closes.iloc[-6]
            if ref_close == 0:
                return None
            return round((today_close - ref_close) / ref_close * 100, 2)
        except Exception:
            return None

    def get_surge_board(self, top_n: int = 50) -> pd.DataFrame:
        """飙升榜：全市场涨幅最大个股。从实时行情快照中按涨跌幅排序。"""
        cache_key = "surge_board"
        cached = self._cached(cache_key, ttl=60)
        if cached is not None:
            return cached.head(top_n)

        try:
            df = self.get_spot_df()
            if df.empty:
                return pd.DataFrame()

            pct_col = self._find_pct_col(df)
            if not pct_col:
                return pd.DataFrame()

            df["_pct"] = pd.to_numeric(df[pct_col], errors="coerce")
            df = df[df["_pct"].notna()].sort_values("_pct", ascending=False)
            self._set_cache(cache_key, df)
            return df.head(top_n)
        except Exception:
            logger.warning("Failed to fetch surge board")
            return pd.DataFrame()

    def _find_pct_col(self, df) -> str | None:
        cols_lower = {c.lower(): c for c in df.columns}
        for candidate in ["涨跌幅", "涨跌幅(%)", "change_pct", "pct_change"]:
            if candidate.lower() in cols_lower:
                return cols_lower[candidate.lower()]
        # 模糊匹配：包含 "涨跌" 和 "幅" 的列
        for c in df.columns:
            if "涨跌" in c and "幅" in c:
                return c
        return None

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
        return ak.stock_board_industry_summary_ths()

    def get_board_industry_index(self) -> pd.DataFrame:
        """行业板块实时行情（涨跌幅/成交额等）"""
        cache_key = "board_industry"
        cached = self._cached(cache_key, ttl=120)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            if hasattr(ak, "stock_board_industry_index_ths"):
                df = ak.stock_board_industry_index_ths()
                if not df.empty:
                    self._set_cache(cache_key, df)
                    return df
        except Exception as e:
            logger.warning("board_industry_index failed: %s", e)
        if cached is not None:
            return cached
        return pd.DataFrame()

    def get_concept_index(self) -> pd.DataFrame:
        """概念板块实时行情"""
        cache_key = "concept_index"
        cached = self._cached(cache_key, ttl=120)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            if hasattr(ak, "stock_board_concept_hist_ths"):
                df = ak.stock_board_concept_hist_ths()
                if not df.empty:
                    self._set_cache(cache_key, df)
                    return df
        except Exception as e:
            logger.warning("concept_hist failed: %s", e)
        if cached is not None:
            return cached
        return pd.DataFrame()

    def get_sector_fund_flow(self) -> pd.DataFrame:
        """板块资金流向排名"""
        cache_key = "sector_fund_flow"
        cached = self._cached(cache_key, ttl=120)
        if cached is not None:
            return cached
        self._rate_limit()
        for fn_name in ["stock_sector_fund_flow_rank", "stock_fund_flow_sector"]:
            try:
                fn = getattr(ak, fn_name, None)
                if fn is None:
                    continue
                df = fn()
                if not df.empty:
                    self._set_cache(cache_key, df)
                    return df
            except Exception as e:
                logger.warning("sector_fund_flow via %s failed: %s", fn_name, e)
        if cached is not None:
            return cached
        return pd.DataFrame()

    def get_industry_fund_flow(self) -> pd.DataFrame:
        """行业资金流向"""
        cache_key = "ind_fund_flow"
        cached = self._cached(cache_key, ttl=120)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            if hasattr(ak, "stock_fund_flow_industry"):
                df = ak.stock_fund_flow_industry()
                if not df.empty:
                    self._set_cache(cache_key, df)
                    return df
        except Exception as e:
            logger.warning("fund_flow_industry failed: %s", e)
        if cached is not None:
            return cached
        return pd.DataFrame()

    def get_industry_stocks(self, industry_name: str) -> pd.DataFrame:
        """行业成分股列表"""
        cache_key = f"ind_stocks_{industry_name}"
        cached = self._cached(cache_key, ttl=600)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            if hasattr(ak, "stock_board_industry_cons_ths"):
                df = ak.stock_board_industry_cons_ths(symbol=industry_name)
                if not df.empty:
                    self._set_cache(cache_key, df)
                    return df
        except Exception as e:
            logger.warning("industry_cons(%s) failed: %s", industry_name, e)
        if cached is not None:
            return cached
        return pd.DataFrame()

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
