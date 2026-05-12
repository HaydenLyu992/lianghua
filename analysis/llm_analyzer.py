import json
import logging
import re
from typing import Optional
from openai import AsyncOpenAI
import markdown

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from core.news_fetcher import NewsFetcher

logger = logging.getLogger(__name__)

md = markdown.Markdown(extensions=["extra", "sane_lists"])

SYSTEM_PROMPT = """你是一个A股新闻分析助手。对给定的新闻列表，分析每条新闻对对应股票的影响：

1. sentiment: "positive"(利好) / "negative"(利空) / "neutral"(中性)
2. impact: 1-5 的影响级别（5=重大影响）
3. summary: 10字以内的简述

以JSON数组格式返回，每个元素包含 title, sentiment, impact, summary。
只返回JSON，不要其他内容。"""


class LLMAnalyzer:
    """LLM 新闻分析器 — 使用 DeepSeek (OpenAI 兼容协议)，强制启用。"""

    def __init__(self):
        self.news_fetcher = NewsFetcher()
        self.client = AsyncOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )

    async def analyze_news(self, code: str) -> list[dict]:
        """拉取个股新闻并用 LLM 做情感分析"""
        raw_news = self.news_fetcher.fetch_stock_news(code, limit=20)
        if not raw_news:
            return []

        titles = [item["title"] for item in raw_news if item.get("title")]
        if not titles:
            return []

        response = await self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(titles, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or "[]"
        content = content.strip()
        if content.startswith("```"):
            parts = content.split("\n", 1)
            content = parts[1] if len(parts) > 1 else content
            content = content.rsplit("\n```", 1)[0] if content.endswith("```") else content.rstrip("`")
        return self._safe_json_parse(content)

    async def analyze_factors(self, code: str, name: str,
                               factors: list[dict]) -> dict[str, str]:
        """用LLM为每个因子生成深度分析描述——不写死规则，基于全部收集到的原始数据"""
        # 构建每个因子的完整原始数据
        data_blocks = []
        for f in factors:
            detail = f.get("detail") or {}
            events = f.get("events") or []

            clean_detail = {k: v for k, v in detail.items()
                           if not k.startswith("_") and k != "数据状态"}

            detail_lines = "\n".join(
                f"    {k}: {v}" for k, v in clean_detail.items()
            ) if clean_detail else "    (暂无子指标数据)"

            event_lines = "\n".join(
                f"    - [{ev.get('sentiment', 'neutral')}] {ev.get('title', '')}"
                for ev in events[:8]
            ) if events else "    (无关键事件)"

            data_blocks.append(f"""### {f['name']}（权重{f['weight']}%）
- 最终得分: {f['score']}/100
- 综合信号: {f['signal']}
- 原始子指标数据:
{detail_lines}
- 关键事件:
{event_lines}""")

        all_raw_data = "\n\n".join(data_blocks)

        factor_analysis_prompt = f"""你是一位严谨的A股量化分析师。以下是 "{code} {name}" 的6个因子全部原始评分数据。

请你基于这些**真实数据**，为每个因子写一段150-200字的深度分析。

## ⚠️ 铁律
1. **只能基于提供的数据进行分析**，数据里没有的东西绝对不要编造
2. **剔除噪音**：如果某个因子只有一个子指标且值为50，说明数据源可能缺失，直接说明"该因子数据不足，评分缺乏参考意义"
3. **数据充足时深入分析**：解读各个子指标的含义、相互关系、对得分的贡献
4. **指出亮点和风险**：数据中的极端值（≥75或≤25）要重点标注
5. **用具体数值说话**：如"净利润增速得分82，反映近两期利润增长约16%"

## 📊 原始数据

{all_raw_data}

## 📝 输出格式

对每个因子，严格使用以下格式输出：

### 基本面
（分析内容）

### 行业
（分析内容）

### 宏观
（分析内容）

### 资金流向
（分析内容）

### 情绪技术
（分析内容）

### 地缘
（分析内容）

只输出以上内容，不要额外说明。"""

        try:
            response = await self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一位严谨的量化分析师。你只基于提供的数据进行分析，"
                            "绝不编造任何信息。数据不足时你会诚实说明，数据充足时"
                            "你会深入解读每个指标的含义和关联。"
                        ),
                    },
                    {"role": "user", "content": factor_analysis_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            raw = (response.choices[0].message.content or "").strip()
            return self._parse_factor_analyses(raw, factors)
        except Exception as e:
            logger.warning("Factor analysis LLM call failed: %s", e)
            return self._fallback_factor_desc(factors)

    def _parse_factor_analyses(self, raw: str, factors: list[dict]) -> dict[str, str]:
        """解析LLM返回的 ### 名称 ... 格式"""
        result = {}
        # 按 "### " 分割
        blocks = re.split(r'\n###\s+', raw)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            # 第一行是名称，后面是内容
            lines = block.split("\n", 1)
            title = lines[0].strip().lstrip("#").strip()
            content = lines[1].strip() if len(lines) > 1 else ""

            # 匹配因子key
            label_to_key = {
                "基本面": "fundamental", "行业": "industry", "宏观": "macro",
                "资金流向": "fund_flow", "情绪技术": "sentiment", "地缘": "geo_external",
            }
            for label, key in label_to_key.items():
                if label in title:
                    result[key] = content
                    break
            else:
                # fallback: try matching by key directly
                for f in factors:
                    if f.get("name", "") in title or f.get("key", "") in title:
                        result[f["key"]] = content
                        break

        return result

    def _fallback_factor_desc(self, factors: list[dict]) -> dict[str, str]:
        """LLM不可用时的简单规则兜底"""
        result = {}
        for f in factors:
            key = f["key"]
            score = f["score"]
            signal = f["signal"]
            signal_cn = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}
            detail = f.get("detail") or {}
            diag = detail.get("数据状态", "")

            level = (
                "优秀" if score >= 80 else "良好" if score >= 65
                else "一般" if score >= 45 else "偏弱" if score >= 30 else "较差"
            )

            parts = [
                f"得分{score}分（{level}），发出{signal_cn.get(signal, '中性')}信号。",
                f"该因子占综合评分的{f['weight']}%权重。",
            ]
            if diag:
                parts.append(f"数据获取状态：{diag}。")

            # 列出子指标
            for k, v in detail.items():
                if not k.startswith("_") and k != "数据状态":
                    parts.append(f"{k}: {v}。")

            result[key] = " ".join(parts)
        return result

    async def comprehensive_analysis(self, code: str, name: str, total_score: int,
                                      signal: str, factors: list[dict],
                                      limit_info: dict, news: list[dict]) -> str:
        """多角色辩论式深度分析：8人激烈辩论，基于真实数据，剔除噪音"""
        # 构建完整原始数据（不仅是摘要，而是所有细节）
        factor_data_blocks = []
        for f in factors:
            detail = f.get("detail") or {}
            events = f.get("events") or []
            clean_detail = {k: v for k, v in detail.items()
                           if not k.startswith("_") and k != "数据状态"}

            detail_items = []
            for k, v in clean_detail.items():
                detail_items.append(f"    {k}: {v}")
            detail_str = "\n".join(detail_items) if detail_items else "    暂无子指标"

            event_items = []
            for ev in events[:5]:
                event_items.append(f"    [{ev.get('sentiment','neutral')}] {ev.get('title','')}")
            event_str = "\n".join(event_items) if event_items else "    无"

            factor_data_blocks.append(
                f"#### {f['name']}（权重{f['weight']}%）\n"
                f"- 得分: **{f['score']}/100** — {f['signal']}\n"
                f"- 子指标原始数据:\n{detail_str}\n"
                f"- 关键事件:\n{event_str}"
            )
        factor_raw_text = "\n\n".join(factor_data_blocks)

        news_text = "\n".join(
            f"- [{n.get('sentiment','neutral')}] {n.get('title','')} — {n.get('summary','')}"
            for n in (news or [])[:15]
        ) or "无相关新闻"

        limit_text = ""
        if limit_info:
            zt_p = limit_info.get('zt_prob', 0)
            dt_p = limit_info.get('dt_prob', 0)
            if limit_info.get("is_zt"):
                limit_text = (
                    f"🔥 该股**今日涨停**，连板{limit_info.get('streak',0)}天，"
                    f"封单{limit_info.get('seal',0)}亿。涨停概率估值{zt_p}%，跌停概率估值{dt_p}%。"
                )
            elif limit_info.get("is_dt"):
                limit_text = (
                    f"💀 该股**今日跌停**。涨停概率估值{zt_p}%，跌停概率估值{dt_p}%。"
                )
            else:
                limit_text = (
                    f"今日未涨跌停。涨停概率估值{zt_p}%，跌停概率估值{dt_p}%。"
                )

        debate_prompt = f"""# 🏛️ A股投资决策辩论会

你正在主持一场高水平的A股投资决策辩论。以下8个角色都由你扮演。

---

## 🚨 证据铁律（违反则辩论无效）

1. **只能引用"可用数据"中明确列出的数值和事件**，不允许编造任何数据
2. **如果某项数据缺失（如基本面只有财务✅而无公告数据），相关论点必须标注"数据不足"**
3. **噪音识别**：50分上下的子指标通常是数据缺失或中性信号，不应作为核心论据
4. **极端值优先**：≥75或≤25的指标才是真正的信号，应重点辩论
5. **每个论点必须可追溯**：说"资金面好"必须具体到"主力资金得分X，近5日净流入Y亿"

---

## 📊 辩论议题

> **股票**: {code} {name}
> **综合得分**: {total_score}/100
> **综合信号**: {signal}
> **核心问题**: 该股当前是否值得买入？

---

## 📈 可用数据（辩论的唯一依据）

### 各因子完整原始数据
{factor_raw_text}

### 涨跌停分析
{limit_text or '无异常'}

### 近期新闻
{news_text}

---

## 👥 辩论角色

### 🔴 看多方

**张首席（机构首席分析师）** — 20年经验，基本面+宏观专家。擅长从财报细节发现利好。风格：每句话都有数字支撑。

**赵游资（游资操盘手）** — 15年短线，管理50亿+资金。从龙虎榜、北向资金、主力流中嗅机会。风格：看筹码结构，重资金博弈。

**孙趋势（趋势交易大师）** — 18年技术分析，均线/MACD/RSI专家。风格：K线反映一切，纯技术面解读。

### 🔵 看空方

**周对冲（对冲基金经理）** — 三次成功做空，专挑估值泡沫和逻辑漏洞。风格：尖锐犀利，用反面案例打击多方。

**吴风控（首席风控官）** — 15年风控，专注压力测试和尾部风险。风格：偏执严谨，永远做最坏打算。

**郑逆向（独立逆向研究员）** — 唱反调专家，提前揭露黑天鹅。风格：寻找"房间里的大象"，挑战共识。

### ⚖️ 中立裁判

**王首席（首席经济学家）** — 25年宏观研究，前央行顾问。不站队，评质量。

**钱量化（量化策略总监）** — 物理博士，管200亿量化基金。纯数据+概率，淘汰逻辑谬误。

---

## 🎯 辩论流程

### 第一轮：多方立论
张首席（基本面+宏观）、赵游资（资金面+情绪）、孙趋势（技术面）分别立论。每人必须引用具体数字。150-200字/人。

### 第二轮：空方反击
周对冲、吴风控、郑逆向逐一批驳多方论点，指出数据中的风险信号和被忽视的利空。150-200字/人。

### 第三轮：自由辩论
多方回击空方质疑，空方深化质疑。必须直接点名，如"赵游资说主力净流入就是看多，但数据明确显示仅流入0.3亿，这体量不足以构成看多信号！"100-150字/人。

### 第四轮：裁判点评
王首席宏观评判，钱量化纯数据+概率审视。200-250字/人。

### 第五轮：最终裁决
联合给出明确结论：是否值得买入？如果值得，什么价位？如果不值得，等待什么条件？

---

## ⚠️ 格式要求

- 必须使用Markdown：##标题、**加粗**、-列表、>引用、---分割线
- 每个角色发言前用 `> **角色名**：` 格式
- 总篇幅2000-3000字
- 结论必须明确，禁止"可能"、"或许"等模糊措辞

现在开始辩论！"""

        try:
            response = await self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一位顶级的A股多空辩论主持人。你严格遵循证据铁律："
                            "每个论点必须有提供的真实数据支撑，绝不编造。"
                            "你能识别噪音（50分上下的中性信号）并引导辩论聚焦于"
                            "真正的信号（极端值、明确的事件）。\n\n"
                            "核心原则：\n"
                            "- 数据缺失就说数据缺失，不要脑补\n"
                            "- 得分50分 = 中性/数据不足，不值得激烈辩论\n"
                            "- 极端值（≥75或≤25）才是真正值得辩论的信号\n"
                            "- 每个论点必须可追溯到具体数字\n"
                            "- 输出规范Markdown，角色标注清晰\n"
                            "- 最终结论必须明确，不模棱两可"
                        ),
                    },
                    {"role": "user", "content": debate_prompt},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            raw = (response.choices[0].message.content or "").strip()
            return md.convert(raw)
        except Exception as e:
            logger.warning("Comprehensive analysis failed: %s", e)
            return self._fallback_summary(total_score, signal, factors, limit_info)

    def _fallback_summary(self, total_score: int, signal: str, factors: list[dict],
                          limit_info: dict) -> str:
        """LLM不可用时的规则兜底——返回Markdown转HTML"""
        bullish = sum(1 for f in factors if f["signal"] == "bullish")
        bearish = sum(1 for f in factors if f["signal"] == "bearish")
        neutral = len(factors) - bullish - bearish

        if total_score >= 80:
            action = "强势看多，可考虑逢低买入。建议买入区间为近期均价的-3%~-5%，止损设在买入价下方5%。"
        elif total_score >= 60:
            action = "中性偏多，可轻仓试探。建议回调至均线附近买入，止损设为买入价下方3%。"
        elif total_score >= 40:
            action = "方向不明，建议观望等待更明确信号。若已持仓可继续持有，不建议新开仓。"
        elif total_score >= 20:
            action = "中性偏空，不宜买入。若已持仓建议逢高减仓，止损不宜过远。"
        else:
            action = "强烈看空，不建议买入。持仓者应考虑止损离场。"

        limit_note = ""
        if limit_info.get("is_zt"):
            limit_note = "> ⚠️ 该股今日涨停，追高风险较大。\n\n"
        elif limit_info.get("is_dt"):
            limit_note = "> ⚠️ 该股今日跌停，短期仍有下行风险。\n\n"

        md_text = f"""## 📊 规则引擎综合研判（LLM暂不可用）

> **综合得分**: {total_score}/100 — **{signal}**

### 多空力量统计

| 方向 | 因子数 |
|------|--------|
| 🔴 看多 | {bullish} |
| 🔵 看空 | {bearish} |
| ⚪ 中性 | {neutral} |

### 操作建议

**{action}**

{limit_note}---
*此为本地规则引擎生成的兜底分析，详细辩论分析将在LLM服务恢复后提供。*"""

        return md.convert(md_text)

    def _safe_json_parse(self, content: str) -> list[dict]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning("LLM returned unparseable content: %s", content[:200])
        return []
