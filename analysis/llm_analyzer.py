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

        factor_analysis_prompt = f"""你是一位严谨的A股数据分析师。以下是 "{code} {name}" 的6个因子全部原始评分数据。

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
                            "你是一位严谨的数据分析师。你只基于提供的数据进行分析，"
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

        debate_prompt = f"""# 🏛️ A股投资决策深度辩论会

你正在主持一场高水平的A股投资决策辩论。以下8个角色都由你扮演。

---

## 🚨 证据铁律（违反则辩论无效）

1. **只能引用"可用数据"中明确列出的数值和事件**，不允许编造任何数据
2. **如果某项数据缺失，相关论点必须标注"⚠️ 数据不足"**，并说明缺失的数据对判断的影响
3. **噪音识别**：50分上下的子指标通常是数据缺失或中性信号，不应作为核心论据
4. **极端值优先**：≥75或≤25的指标才是真正的信号，应重点辩论和深度挖掘
5. **每个论点必须可追溯到具体数字**：说"资金面好"必须具体到"主力资金得分X，近5日净流入Y亿"
6. **区分相关性与因果性**：不能因为两个指标同向就断言因果关系

---

## 📊 辩论议题

> **股票**: {code} {name}
> **综合得分**: {total_score}/100
> **综合信号**: {signal}
> **核心问题**: 该股当前是否值得买入？什么价位买入？止盈止损设在哪里？

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

**张首席（机构首席分析师）** — 20年从业经验，CPA+CFA持证，基本面与宏观分析专家。擅长从财报细节中发现被市场忽视的利好，精通财务模型和估值方法。风格：每句话都有数字支撑，引用具体财务比率。

**赵游资（资深游资操盘手）** — 15年短线交易，管理50亿+资金池。从龙虎榜席位、北向资金流、主力筹码结构中嗅探交易机会。风格：看筹码博弈，重资金流向，懂市场心理。

**孙趋势（技术分析大师）** — 18年技术分析经验，精通均线系统、MACD背离、RSI结构、布林带、成交量价分析等多维技术指标。风格：K线反映一切信息，多指标共振才是真信号。

### 🔵 看空方

**周对冲（对冲基金经理）** — 三次成功做空经历，管理20亿对冲基金。专挑估值泡沫、财务瑕疵和逻辑漏洞。风格：尖锐犀利，用历史反面案例和数据矛盾打击多方论点。

**吴风控（首席风控官）** — 15年金融风控经验，专注压力测试和尾部风险建模。曾在2008、2015、2024年极端行情中成功预警。风格：偏执严谨，永远做最坏打算，量化风险敞口。

**郑逆向（独立逆向研究员）** — 以唱反调著称的独立研究员，提前揭露过多起财务造假和黑天鹅事件。风格：寻找"房间里的大象"，挑战市场共识，曝光隐藏风险。

### ⚖️ 中立裁判

**王首席（首席经济学家）** — 25年宏观研究，前央行货币政策委员会顾问。不站队多空，评估论证质量和宏观背景。风格：高屋建瓴，关注政策传导机制。

**钱分析（量化分析总监）** — 物理学博士转量化，管理百亿量化基金。纯数据驱动+贝叶斯概率思维，严格淘汰逻辑谬误和统计偏差。风格：一切用概率说话，识别数据中的伪模式。

---

## 🎯 辩论流程（总计5000-8000字）

### 【前置轮A】财务深度分析 — 张首席主笔（400-500字）

张首席对基本面因子中的**所有财务数据**进行深度拆解分析：

1. **盈利能力分析**：逐项解读净利润增速、ROE、毛利率等指标，判断盈利质量和可持续性
2. **估值分析**：结合PE/PB与行业均值对比，判断当前估值水位（低估/合理/高估）
3. **成长性分析**：营收增速、利润增速的趋势判断，是否存在加速增长或增长放缓信号
4. **财务健康度**：负债率、现金流等指标反映的财务风险
5. **近期公告解读**：限售解禁、机构调研、增减持等事件的影响评估
6. **关键结论**：用一句话总结该股基本面的核心矛盾（最大的亮点 vs 最大的隐忧）

> 要求：每个结论都附带具体数值，数据缺失的维度明确标注。不堆砌术语，要给出投资含义。

### 【前置轮B】技术深度分析 — 孙趋势主笔（400-500字）

孙趋势对所有技术指标进行多维深度分析：

1. **均线系统分析**：MA5/MA10/MA20/MA60 的排列形态（多头/空头/粘合），均线斜率变化趋势
2. **MACD 深度解读**：DIF/DEA 的相对位置、金叉死叉的时间节点、柱状图的背离信号
3. **RSI 结构分析**：当前RSI数值的历史百分位、超买超卖区间的持续时间、RSI背离形态
4. **布林带分析**（如有）：价格在布林带中的位置、带宽收窄/扩张趋势、突破信号
5. **成交量分析**（如有）：量价配合关系、放量/缩量的含义、换手率异常
6. **支撑/阻力位**：基于近期高低点和均线位置，标出关键支撑位和阻力位
7. **ATR波动率分析**（如有）：当前波动率水平，对止损设置的技术建议
8. **综合技术结论**：多指标共振方向，技术面给出的交易信号强度

> 要求：明确指出各指标之间是相互验证还是相互矛盾，不只看单一指标。

---

### 第一轮：多方立论（每人350-450字）

**张首席**从基本面和宏观面立论：引用具体财务数字、宏观指标，论证为什么该股基本面支撑上涨。必须回应前置轮中发现的隐忧。

**赵游资**从资金面和情绪面立论：引用主力资金流向、北向资金、龙虎榜、融资融券等数据，论证筹码结构有利于上涨。说明资金面的持续性和体量。

**孙趋势**从技术面立论：在技术深度分析基础上，给出操作层面的技术依据——当前处于什么技术阶段（筑底/拉升/见顶/下跌中继），关键突破点位在哪里。

---

### 第二轮：空方深度反击（每人350-450字）

**周对冲**逐一批驳多方基本面论点：指出财务数据中的矛盾、估值陷阱、盈利质量疑点。用至少一个历史反面案例说明类似情况下股价的表现。

**吴风控**从风险角度反击：量化最大回撤风险、流动性风险、事件风险。计算如果各项指标恶化到25分位，股价可能的跌幅。指出多方避而不谈的尾部风险。

**郑逆向**从市场共识的反面出击：寻找被忽视的利空（如行业政策风险、竞争格局恶化、大股东行为异常等）。挑战多方的核心假设。

---

### 第三轮：自由辩论（每人200-300字）

多方回击空方质疑，空方深化追问。必须**直接点名**反驳对象，如：

> "赵游资声称主力净流入构成看多信号，但数据显示主力净流入仅0.3亿，占日成交额不足2%，这一体量远不足以构成实质性看多信号！"
> "周对冲质疑PE偏高，但忽略了该股所处行业平均PE为35倍，当前25倍实为行业偏低水平！"

每人必须至少点名反驳一个对方的具体论点。

---

### 第四轮：裁判综合点评（每人350-450字）

**王首席**从宏观和行业层面做综合评判：当前宏观环境（货币政策、经济周期、政策方向）对这类股票的利弊。不评判多空对错，评判**论证质量**——谁的数据引用更准确，谁的逻辑链条更完整，谁在选择性使用数据。

**钱分析**从纯数据概率层面评判：识别双方法论中的确认偏误、幸存者偏差、过度拟合。用概率思维重新表述核心问题——"根据现有数据，该股上涨超过X%的概率约为Y%，下跌超过Z%的概率约为W%。"

---

### 第五轮：最终裁决（500-700字）

8位角色联合给出如下**无歧义、可直接执行**的结论：

```
## 📋 最终裁决

### 🎯 操作建议
**明确结论**: [买入 / 卖出 / 观望 — 三选一，禁止模糊]

### 💰 交易参数（如建议买入）
- **建议买入价格区间**: ¥X.XX — ¥Y.YY（说明定价依据：如20日均线支撑、布林下轨、前低等）
- **止损价格**: ¥X.XX（止损幅度X%，依据：ATR指标/前低支撑/均线支撑）
- **第一止盈目标**: ¥X.XX（涨幅X%，依据：前高阻力/布林上轨/估值上限）
- **第二止盈目标**: ¥X.XX（涨幅X%，依据：突破前高后的技术目标位）
- **仓位建议**: X成仓位（依据：综合得分对应的凯利公式/风险预算）
- **建议持仓周期**: X天 — X周（依据：技术信号的时间框架）

### ⚠️ 关键风险清单
1. 风险一（具体描述 + 触发条件）
2. 风险二（具体描述 + 触发条件）
3. 风险三（具体描述 + 触发条件）

### 📊 信心水平
本次判断的信心水平: X/10
- 数据充分度: X/10
- 信号一致性: X/10
- 不确定性来源: [具体说明]
```

---

## ⚠️ 格式要求

- 使用规范Markdown：##/###标题层级、**加粗**、-列表、>引用块、---分割线
- 每个角色发言使用 `> **角色名**：` 起始，发言内容另起段落
- 辩论必须激烈、有交锋感，但所有论据必须来自提供的数据
- 总篇幅5000-8000字
- 最终裁决必须全用"必须"、"坚决"等确定性措辞，**严禁"可能"、"或许"、"建议关注"等模糊表达**
- **严禁在辩论中引用任何未在"可用数据"中出现的数据、事件或指标**

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
                            "- 财务分析要深入拆解盈利能力、估值、成长性、健康度\n"
                            "- 技术分析要多指标共振验证，不只看单一指标\n"
                            "- 辩论要激烈有交锋，点名反驳对方具体论点\n"
                            "- 输出规范Markdown，角色标注清晰\n"
                            "- 最终裁决必须给出具体的买入价、止损价、止盈价、仓位，严禁模糊"
                        ),
                    },
                    {"role": "user", "content": debate_prompt},
                ],
                temperature=0.7,
                max_tokens=8192,
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
