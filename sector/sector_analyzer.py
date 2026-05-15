import asyncio
import logging
from datetime import datetime

import pandas as pd
from openai import AsyncOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from core.akshare_client import AkShareClient
from core.news_fetcher import NewsFetcher

logger = logging.getLogger(__name__)

SECTOR_SYSTEM_PROMPT = """你是一位顶级的A股板块轮动策略分析师，拥有20年从业经验。你同时精通：
1. 宏观经济与政策分析（能预判政策对特定产业链的传导影响）
2. 资金流向分析（能从主力资金/北向资金的动向中嗅探板块轮动信号）
3. 技术形态分析（能识别板块指数的趋势阶段和关键突破点）
4. 产业链研究（能拆解上下游关系，判断景气度传导路径）

你的分析风格：每个结论必须有具体数据支撑，数据不足时诚实标注，绝不编造。"""


def _is_excluded_stock(code: str) -> bool:
    """排除创业板(300)、科创板(688)、北交所(8)、港股(0开头但非主板)"""
    code = str(code).zfill(6)
    return (
        code.startswith("300")
        or code.startswith("301")
        or code.startswith("688")
        or code.startswith("8")
        or code.startswith("4")  # 退市板
    )


class SectorAnalyzer:
    """热门板块分析引擎：聚合多源数据 + LLM驱动分析，不指定个股，全局扫描。"""

    def __init__(self):
        self.client = AkShareClient()
        self.news_fetcher = NewsFetcher()
        self.llm = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    async def collect_sector_data(self) -> dict:
        """并行收集所有板块相关数据"""
        tasks = [
            self._safe_call("industry_index", self._fetch_industry_index),
            self._safe_call("concept_index", self._fetch_concept_index),
            self._safe_call("sector_fund_flow", self._fetch_sector_fund_flow),
            self._safe_call("industry_fund_flow", self._fetch_industry_fund_flow),
            self._safe_call("limit_up_dist", self._fetch_limit_up_distribution),
            self._safe_call("northbound_top", self._fetch_northbound_top),
            self._safe_call("policy_news", self._fetch_policy_news),
            self._safe_call("hot_rank", self._fetch_hot_rank),
        ]
        results = dict(await asyncio.gather(*tasks))
        return results

    async def analyze_hot_sectors(self) -> dict:
        """综合分析热门板块，返回结构化结果"""
        data = await self.collect_sector_data()

        prompt = self._build_analysis_prompt(data)
        markdown_content = await self._call_llm(prompt)

        return {
            "analysis_html": self._md_to_html(markdown_content),
            "analysis_md": markdown_content,
            "raw_data": data,
            "generated_at": datetime.now().isoformat(),
        }

    # ── 数据采集 ──────────────────────────────────

    async def _safe_call(self, name: str, fn):
        try:
            result = await fn()
            return (name, result)
        except Exception as e:
            logger.warning("Sector data source [%s] failed: %s", name, e)
            return (name, None)

    async def _fetch_industry_index(self) -> dict:
        df = self.client.get_board_industry_index()
        if df.empty:
            return {}
        name_col = next((c for c in df.columns if c in ("板块", "板块名称", "行业")), df.columns[0])
        pct_col = next((c for c in df.columns if "涨跌幅" in c), None)
        if pct_col is None:
            return {}
        df = df.copy()
        df["_pct"] = pd.to_numeric(df[pct_col], errors="coerce")
        top = df.nlargest(10, "_pct")
        bottom = df.nsmallest(5, "_pct")
        return {
            "top_industries": [
                {"name": r[name_col], "change_pct": round(float(r["_pct"]), 2)}
                for _, r in top.iterrows()
            ],
            "bottom_industries": [
                {"name": r[name_col], "change_pct": round(float(r["_pct"]), 2)}
                for _, r in bottom.iterrows()
            ],
        }

    async def _fetch_concept_index(self) -> dict:
        df = self.client.get_concept_index()
        if df.empty:
            return {}
        name_col = next((c for c in df.columns if c in ("板块", "板块名称", "概念")), df.columns[0])
        pct_col = next((c for c in df.columns if "涨跌幅" in c), None)
        if pct_col is None:
            return {}
        df = df.copy()
        df["_pct"] = pd.to_numeric(df[pct_col], errors="coerce")
        top = df.nlargest(15, "_pct")
        return {
            "top_concepts": [
                {"name": r[name_col], "change_pct": round(float(r["_pct"]), 2)}
                for _, r in top.iterrows()
            ],
        }

    async def _fetch_sector_fund_flow(self) -> dict:
        df = self.client.get_sector_fund_flow()
        if df.empty:
            return {}
        name_col = next((c for c in df.columns if c in ("板块", "板块名称", "名称", "行业")), df.columns[0])
        inflow_col = next(
            (c for c in df.columns if "净流入" in c or "主力净流入" in c or "main_net" in c.lower()), None
        )
        if inflow_col is None:
            return {}
        df = df.copy()
        df["_flow"] = pd.to_numeric(df[inflow_col], errors="coerce")
        top_inflow = df.nlargest(10, "_flow")
        top_outflow = df.nsmallest(5, "_flow")
        return {
            "top_inflow": [
                {"name": r[name_col], "net_inflow": round(float(r["_flow"]) / 10000, 2)}
                for _, r in top_inflow.iterrows()
            ],
            "top_outflow": [
                {"name": r[name_col], "net_inflow": round(float(r["_flow"]) / 10000, 2)}
                for _, r in top_outflow.iterrows()
            ],
        }

    async def _fetch_industry_fund_flow(self) -> dict:
        df = self.client.get_industry_fund_flow()
        if df.empty:
            return {}
        name_col = next((c for c in df.columns if c in ("板块", "板块名称", "名称", "行业")), df.columns[0])
        inflow_col = next(
            (c for c in df.columns if "净流入" in c or "主力净流入" in c or "main_net" in c.lower()), None
        )
        if inflow_col is None:
            return {}
        df = df.copy()
        df["_flow"] = pd.to_numeric(df[inflow_col], errors="coerce")
        top = df.nlargest(10, "_flow")
        return {
            "top_inflow_industries": [
                {"name": r[name_col], "net_inflow": round(float(r["_flow"]) / 10000, 2)}
                for _, r in top.iterrows()
            ],
        }

    async def _fetch_limit_up_distribution(self) -> dict:
        pool = self.client.get_limit_up_pool()
        if pool.empty:
            return {}
        industry_col = next(
            (c for c in pool.columns if "行业" in c or "板块" in c or "industry" in c.lower()), None
        )
        if industry_col:
            counts = pool[industry_col].value_counts().head(15).to_dict()
            return {"by_industry": counts}
        return {"by_industry": {}}

    async def _fetch_northbound_top(self) -> dict:
        df = self.client.get_northbound_individual()
        if df.empty:
            return {}
        code_col = next((c for c in df.columns if "代码" in c or "code" in c.lower()), df.columns[0])
        name_col = next((c for c in df.columns if "名称" in c or "name" in c.lower()), None)
        net_col = next((c for c in df.columns if "净买入" in c or "net" in c.lower()), None)

        if net_col is None:
            return {}
        df = df.copy()
        df["_net"] = pd.to_numeric(df[net_col], errors="coerce")
        top_buy = df.nlargest(20, "_net")
        top_sell = df.nsmallest(10, "_net")

        def _to_item(row):
            code = str(row.get(code_col, ""))
            return {
                "code": code,
                "name": str(row.get(name_col, "")) if name_col else "",
                "net_buy": round(float(row["_net"]) / 10000, 2) if pd.notna(row["_net"]) else 0,
                "is_excluded": _is_excluded_stock(code),
            }

        return {
            "top_buy": [_to_item(r) for _, r in top_buy.iterrows()],
            "top_sell": [_to_item(r) for _, r in top_sell.iterrows()],
        }

    async def _fetch_policy_news(self) -> dict:
        items = self.news_fetcher.fetch_macro_news(limit=20)
        return {"recent_news": items}

    async def _fetch_hot_rank(self) -> dict:
        df = self.client.get_hot_rank()
        if df.empty:
            return {}
        code_col = next((c for c in df.columns if "代码" in c or "code" in c.lower()), df.columns[0])
        name_col = next((c for c in df.columns if "名称" in c or "name" in c.lower()), df.columns[1])
        hot_col = next((c for c in df.columns if "热度" in c or "rank" in c.lower()), None)

        items = []
        for _, row in df.head(30).iterrows():
            code = str(row.get(code_col, ""))
            if not _is_excluded_stock(code):
                items.append({
                    "code": code,
                    "name": str(row.get(name_col, "")),
                    "hot": str(row.get(hot_col, "")) if hot_col else "",
                })
        return {"top_hot_stocks": items[:20]}

    # ── LLM 分析 ──────────────────────────────────

    def _build_analysis_prompt(self, data: dict) -> str:
        """构建发给LLM的完整分析Prompt"""
        parts = []
        parts.append(f"# 当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}（A股交易日盘中数据）\n")

        # 行业指数
        ind = data.get("industry_index") or {}
        if ind.get("top_industries"):
            items_str = "\n".join(
                f"  - {it['name']}: {it['change_pct']:+.2f}%"
                for it in ind["top_industries"]
            )
            parts.append(f"## 行业板块涨幅TOP10\n{items_str}\n")

        # 概念指数
        concept = data.get("concept_index") or {}
        if concept.get("top_concepts"):
            items_str = "\n".join(
                f"  - {it['name']}: {it['change_pct']:+.2f}%"
                for it in concept["top_concepts"]
            )
            parts.append(f"## 概念板块涨幅TOP15\n{items_str}\n")

        # 板块资金流向
        sff = data.get("sector_fund_flow") or {}
        if sff.get("top_inflow"):
            items_str = "\n".join(
                f"  - {it['name']}: 净流入 {it['net_inflow']:.2f} 亿"
                for it in sff["top_inflow"]
            )
            parts.append(f"## 板块主力资金净流入TOP10\n{items_str}\n")

        if sff.get("top_outflow"):
            items_str = "\n".join(
                f"  - {it['name']}: 净流出 {abs(it['net_inflow']):.2f} 亿"
                for it in sff["top_outflow"]
            )
            parts.append(f"## 板块主力资金净流出TOP5\n{items_str}\n")

        # 行业资金流向
        iff = data.get("industry_fund_flow") or {}
        if iff.get("top_inflow_industries"):
            items_str = "\n".join(
                f"  - {it['name']}: 净流入 {it['net_inflow']:.2f} 亿"
                for it in iff["top_inflow_industries"]
            )
            parts.append(f"## 行业资金净流入TOP10\n{items_str}\n")

        # 涨停分布
        limit = data.get("limit_up_dist") or {}
        if limit.get("by_industry"):
            items_str = "\n".join(
                f"  - {k}: {v} 家涨停" for k, v in limit["by_industry"].items()
            )
            parts.append(f"## 涨停板行业分布\n{items_str}\n")

        # 北向资金
        nb = data.get("northbound_top") or {}
        if nb.get("top_buy"):
            items_str = "\n".join(
                f"  - {it['name']}({it['code']}): 净买入 {it['net_buy']:.2f} 亿"
                for it in nb["top_buy"][:15]
            )
            parts.append(f"## 北向资金净买入TOP15（个股）\n{items_str}\n")

        # 政策新闻
        pn = data.get("policy_news") or {}
        if pn.get("recent_news"):
            items_str = "\n".join(
                f"  - {n.get('title', '')}" for n in pn["recent_news"][:15]
            )
            parts.append(f"## 近期宏观/政策新闻\n{items_str}\n")

        # 热门个股
        hot = data.get("hot_rank") or {}
        if hot.get("top_hot_stocks"):
            items_str = "\n".join(
                f"  - {s['name']}({s['code']}) 热度:{s['hot']}"
                for s in hot["top_hot_stocks"]
            )
            parts.append(f"## 市场热门个股（已排除创业板/科创板）\n{items_str}\n")

        data_block = "\n".join(parts)

        return f"""{data_block}

---

## 🎯 分析任务

基于以上**真实最新数据**，请你做一次完整的A股板块轮动分析。必须分别分析**行业板块**和**概念板块**，两者不可混为一谈。

### ⚠️ 铁律
1. **只能基于上面提供的数据进行分析**，数据中没有的板块/个股绝不要编造
2. **每个结论必须有具体数字支撑**："该板块涨幅X%，资金净流入Y亿，Z只个股涨停"
3. **如果某类数据缺失（显示为None或空），直接说明"该项数据暂不可用"，不得脑补**
4. **推荐的个股必须是数据中出现的、真实存在的代码**，如果数据中未出现某只股票，不能推荐
5. **区分确定性和不确定性**：有数据支撑的判断 vs 基于经验的推测，后者要明确标注"推测"
6. **排除规则**：不要推荐创业板(300xxx)、科创板(688xxx)、北交所(8xxxxx)股票

### 📊 输出结构

#### 🔥 当前最热行业板块 TOP5

对每个行业板块，输出：
- **板块名称**
- **热力指数**（综合涨幅+资金+涨停数的主观评分 1-10）
- **核心驱动逻辑**（至少3条，每条注明数据依据）
  - 政策/事件驱动：关联哪些具体政策或事件？
  - 资金流向验证：主力/北向资金是否配合？
  - 情绪/技术面验证：涨停密度、板块涨幅是否持续？
- **持续性判断**：是短期炒作（1-3天）还是中期趋势（2-4周）？依据是什么？
- **推荐个股**（2-3只该板块内的非创业板/科创板/北交所股票，必须是在上述数据中出现过的）：
  - 代码 + 名称
  - 推荐理由（具体、有数据支撑）
  - 风险等级（低/中/高）

#### 🚀 当前最热概念板块 TOP5

对每个概念板块，输出：
- **板块名称**
- **热力指数**（综合涨幅+资金+涨停数的主观评分 1-10）
- **核心驱动逻辑**（至少3条，每条注明数据依据）
  - 概念催化剂：是什么事件/政策/技术突破引爆了该概念？
  - 资金流向验证：主力资金是否在涌入该概念板块？
  - 情绪/技术面验证：概念内涨停家数、板块涨幅持续性如何？
- **持续性判断**：是短期题材炒作（1-3天）还是中期趋势（2-4周）？依据是什么？
- **与行业板块的关联**：该概念板块与哪些行业板块形成共振？是上游驱动下游，还是独立行情？
- **推荐个股**（2-3只该概念内的非创业板/科创板/北交所股票）：
  - 代码 + 名称
  - 推荐理由（具体、有数据支撑）
  - 风险等级（低/中/高）

#### 🔮 未来1-4周潜力板块 TOP3

识别那些可能接力的板块（可以是行业板块或概念板块，需标注类型）：
- **板块名称**（标注"行业"或"概念"）
- **看好逻辑**（政策预期/景气度拐点/资金埋伏信号）
- **催化剂事件**（即将发生的政策/财报/行业事件）
- **当前信号强度**（强/中/弱）
- **潜在标的**（1-2只）

#### ⚠️ 风险警示

- 需回避的板块及理由（区分行业板块和概念板块）
- 整体市场风险判断（基于宏观政策面）
- 如果数据不足以支撑判断，说明需要补充哪些信息

### 📝 格式要求

使用规范Markdown输出：## / ### 层级标题、**加粗**重点数据、- 列表、> 引用关键证据。
总篇幅3000-4500字。每个板块分析必须有血有肉，不能只有空洞结论。行业板块和概念板块必须分开独立分析。"""

    async def _call_llm(self, prompt: str) -> str:
        try:
            response = await self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=6144,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error("Sector analysis LLM call failed: %s", e)
            return f"## LLM分析暂时不可用\n\n错误信息: {e}\n\n请稍后重试。"

    def _md_to_html(self, md_text: str) -> str:
        try:
            import markdown
            md = markdown.Markdown(extensions=["extra", "sane_lists"])
            return md.convert(md_text)
        except Exception:
            escaped = md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f"<pre>{escaped}</pre>"
