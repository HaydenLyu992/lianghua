import logging
from datetime import datetime
import pandas as pd
import numpy as np

from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)


class TrendRanking:
    """趋势排行：基于5日资金流 + 涨跌停 + 行情数据的多维度趋势识别。"""

    def __init__(self, client: AkShareClient):
        self.client = client

    def refresh(self, top_n: int = 50) -> tuple[list[dict], list[dict]]:
        """返回 (上升趋势, 下跌趋势) 排行列表。两轮计算：首轮日数据快速筛选，二轮加5日K线精排。"""
        try:
            df = self.client.get_all_fund_flow()
            if df.empty:
                return [], []
        except Exception as e:
            logger.warning("Failed to fetch fund flow data for trend ranking: %s", e)
            return [], []

        # 获取涨跌停池
        zt_codes = self._safe_limit_pool(is_up=True)
        dt_codes = self._safe_limit_pool(is_up=False)

        # —— 第一轮：仅用日数据快速评分 ——
        results = []
        cols = {c.lower(): c for c in df.columns}

        for _, row in df.iterrows():
            code = self._cell(row, cols, ["代码"])
            name = self._cell(row, cols, ["名称"])
            if not code:
                continue

            pct = self._float_val(row, cols, ["涨跌幅"])
            main_net = self._float_val(row, cols, ["主力净流入-净额"])
            is_zt = code in zt_codes
            is_dt = code in dt_codes

            # —— 趋势得分模型（第一轮，仅日数据） ——
            up_score = 0.0
            if pct:
                up_score += np.clip(pct, -10, 10) * 3
            if main_net:
                up_score += np.clip(main_net / 1e8, -5, 5) * 2
            if is_zt:
                up_score += 30
            if is_dt:
                up_score -= 30

            down_score = -up_score

            results.append({
                "code": str(code),
                "name": str(name),
                "change_pct": round(pct, 2) if pct is not None else None,
                "main_net": main_net,
                "is_zt": is_zt,
                "is_dt": is_dt,
                "up_score": round(up_score, 1),
                "down_score": round(down_score, 1),
            })

        if not results:
            return [], []

        # —— 第二轮：取候选股，加入5日涨跌幅精排 ——
        CANDIDATE_N = 150
        rising_candidates = sorted(results, key=lambda x: x["up_score"], reverse=True)[:CANDIDATE_N]
        falling_candidates = sorted(results, key=lambda x: x["down_score"], reverse=True)[:CANDIDATE_N]
        all_candidates = rising_candidates + falling_candidates
        candidate_codes = list({r["code"] for r in all_candidates})

        changes_5d = self.client.get_5day_change_batch(candidate_codes)

        # 更新候选股的分数（加入5日涨跌因子）
        for r in all_candidates:
            chg_5d = changes_5d.get(r["code"])
            r["change_5d"] = chg_5d
            if chg_5d is not None:
                extra = np.clip(chg_5d, -30, 30) * 1.5
                r["up_score"] = round(r["up_score"] + extra, 1)
                r["down_score"] = round(r["down_score"] - extra, 1)

        # 确保未纳入候选的股票也有 change_5d 字段
        for r in results:
            if "change_5d" not in r:
                r["change_5d"] = None

        # 上升趋势：在候选股中按 up_score 降序取 top_n
        rising_top = []
        seen = set()
        for r in sorted(rising_candidates, key=lambda x: x["up_score"], reverse=True):
            if r["code"] not in seen:
                seen.add(r["code"])
                r["display"] = self._trend_display(r, "up")
                r["trend_label"] = self._trend_label(r, "up")
                rising_top.append(r)
            if len(rising_top) >= top_n:
                break

        # 下跌趋势：在候选股中按 down_score 降序取 top_n
        falling_top = []
        seen = set()
        for r in sorted(falling_candidates, key=lambda x: x["down_score"], reverse=True):
            if r["code"] not in seen:
                seen.add(r["code"])
                r["display"] = self._trend_display(r, "down")
                r["trend_label"] = self._trend_label(r, "down")
                falling_top.append(r)
            if len(falling_top) >= top_n:
                break

        return rising_top, falling_top

    def _safe_limit_pool(self, is_up: bool) -> set:
        try:
            if is_up:
                df = self.client.get_limit_up_pool()
            else:
                df = self.client.get_limit_down_pool()
            if df.empty:
                return set()
            code_col = next((c for c in df.columns if "代码" in c), None)
            if not code_col:
                return set()
            return set(str(c).replace("sz","").replace("sh","").replace("bj","").strip()
                      for c in df[code_col])
        except Exception as e:
            logger.warning("Limit pool fetch failed: %s", e)
            return set()

    def _cell(self, row, cols: dict, keywords: list[str]) -> str | None:
        for kw in keywords:
            for ck, cv in cols.items():
                if kw.lower() in ck:
                    return str(row.get(cv, ""))
        return None

    def _float_val(self, row, cols: dict, keywords: list[str]) -> float | None:
        for kw in keywords:
            for ck, cv in cols.items():
                if kw.lower() in ck:
                    try:
                        v = row.get(cv)
                        if v is None or v == "-" or v == "":
                            return None
                        return float(v)
                    except (ValueError, TypeError):
                        pass
        return None

    def _trend_display(self, r: dict, direction: str) -> str:
        """格式化趋势指标的显示值"""
        parts = []
        if direction == "up":
            if r["is_zt"]:
                parts.append("🔥涨停")
            if r["change_pct"] is not None:
                parts.append(f"日涨跌{r['change_pct']:+.1f}%")
        else:
            if r["is_dt"]:
                parts.append("💀跌停")
            if r["change_pct"] is not None:
                parts.append(f"日涨跌{r['change_pct']:+.1f}%")
        if r.get("change_5d") is not None:
            parts.append(f"5日{r['change_5d']:+.1f}%")
        if r["main_net"]:
            parts.append(f"主力{r['main_net']/1e8:+.1f}亿")
        return " · ".join(parts) if parts else f"评分{r['up_score']:.0f}"

    def _trend_label(self, r: dict, direction: str) -> str:
        """趋势强度标签"""
        score = r["up_score"] if direction == "up" else r["down_score"]
        if score >= 60:
            return "强势"
        elif score >= 30:
            return "走强"
        elif score >= 10:
            return "偏强"
        elif score >= -10:
            return "震荡"
        elif score >= -30:
            return "偏弱"
        elif score >= -60:
            return "走弱"
        else:
            return "弱势"

    def build_llm_prompt(self, rising: list[dict], falling: list[dict]) -> str:
        """构建LLM趋势分析提示词"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        rising_block = "\n".join(
            f"  {i+1}. {r['code']} {r['name']} | 日涨跌:{r['change_pct']:+.2f}% | "
            f"5日涨跌:{r.get('change_5d') or 0:+.2f}% | "
            f"主力净流入:{r['main_net']/1e8:.2f}亿 | 趋势评分:{r['up_score']:.0f} | {r['trend_label']}"
            for i, r in enumerate(rising[:15])
        ) if rising else "  暂无数据"

        falling_block = "\n".join(
            f"  {i+1}. {r['code']} {r['name']} | 日涨跌:{r['change_pct']:+.2f}% | "
            f"5日涨跌:{r.get('change_5d') or 0:+.2f}% | "
            f"主力净流入:{r['main_net']/1e8:.2f}亿 | 趋势评分:{r['down_score']:.0f} | {r['trend_label']}"
            for i, r in enumerate(falling[:15])
        ) if falling else "  暂无数据"

        return f"""# 当前时间：{now}（A股交易日盘中数据）

## 上升趋势排行 TOP15
{rising_block}

## 下跌趋势排行 TOP15
{falling_block}

---

## 🎯 分析任务

基于以上趋势排行数据，请做一次简洁的趋势分析：

### ⚠️ 铁律
1. 只能基于提供的数据分析，不要编造
2. 用具体数字说话
3. 数据不足时诚实标注

### 📊 输出结构（总篇幅800-1500字）

#### 📈 上升趋势概况
- 当前上升趋势的整体特征（是普涨还是结构性行情？）
- 哪些板块/题材在领涨？有什么共性？
- 资金面是否支持趋势延续？
- TOP3 上升趋势个股点评（各50-80字）

#### 📉 下跌趋势概况
- 当前下跌趋势的整体特征
- 哪些板块/题材在领跌？
- 是轮动调整还是系统性风险信号？
- TOP3 下跌趋势个股点评（各50-80字）

#### ⚠️ 风险提示
- 追涨风险提示
- 抄底风险提示

使用规范Markdown输出。"""
