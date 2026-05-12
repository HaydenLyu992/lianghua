import asyncio
import json
import logging
from config import FACTOR_WEIGHTS
from core.akshare_client import AkShareClient
from core.news_fetcher import NewsFetcher
from core.database import AsyncSession, AnalysisHistory
from factors.fundamental import FundamentalFactor
from factors.industry import IndustryFactor
from factors.macro import MacroFactor
from factors.fund_flow import FundFlowFactor
from factors.sentiment import SentimentFactor
from factors.geo_external import GeoExternalFactor
from analysis.limit_analyzer import LimitAnalyzer
from analysis.llm_analyzer import LLMAnalyzer

logger = logging.getLogger(__name__)


class StockAnalyzer:
    """综合分析引擎：并行调用 6 因子 + 涨跌停 + LLM，加权汇总生成报告。"""

    def __init__(
        self,
        client: AkShareClient,
        news_fetcher: NewsFetcher,
        llm: LLMAnalyzer,
    ):
        self.client = client
        self.news_fetcher = news_fetcher
        self.llm = llm

        self.factors = {
            "fundamental": FundamentalFactor(client),
            "industry": IndustryFactor(client),
            "macro": MacroFactor(client),
            "fund_flow": FundFlowFactor(client),
            "sentiment": SentimentFactor(client),
            "geo_external": GeoExternalFactor(client, news_fetcher),
        }
        self.limit_analyzer = LimitAnalyzer(client)

    async def analyze(self, code: str, name: str = "") -> dict:
        # 1) 并行跑 6 个因子 + 涨跌停
        tasks = [
            self._safe_factor(k, v, code, name)
            for k, v in self.factors.items()
        ]
        tasks.append(self._safe_limit(code))
        results = await asyncio.gather(*tasks)

        factor_results = {}
        limit_result = None
        for r in results:
            if isinstance(r, tuple):
                key, fr = r
                factor_results[key] = fr
            elif isinstance(r, dict) and "zt_prob" in r:
                limit_result = r

        # 2) 加权汇总
        total_score = 0
        total_weight = 0
        for key, fr in factor_results.items():
            w = FACTOR_WEIGHTS.get(key, 10)
            total_score += fr.score * w
            total_weight += w
        total_score = int(total_score / total_weight)

        # 3) LLM 新闻分析
        llm_news = await self.llm.analyze_news(code)

        # 4) 组装报告
        signal = (
            "强烈看多" if total_score >= 80 else
            "中性偏多" if total_score >= 60 else
            "中性" if total_score >= 40 else
            "中性偏空" if total_score >= 20 else
            "强烈看空"
        )

        factors_with_desc = []
        for key, fr in factor_results.items():
            label = self._factor_label(key)
            weight = FACTOR_WEIGHTS.get(key, 10)
            factors_with_desc.append({
                "key": key,
                "name": label,
                "score": fr.score,
                "signal": fr.signal,
                "weight": weight,
                "detail": fr.detail,
                "events": fr.events,
                "description": self._factor_desc(key, fr.score, fr.signal, fr.detail, weight),
            })

        stock_name = name or self._get_stock_name(code)

        # 5) LLM 综合分析摘要
        summary = await self.llm.comprehensive_analysis(
            code, stock_name, total_score, signal, factors_with_desc,
            limit_result, llm_news,
        )

        report = {
            "code": code,
            "name": stock_name,
            "score": total_score,
            "signal": signal,
            "factors": factors_with_desc,
            "limit": limit_result,
            "news": llm_news,
            "summary": summary,
        }

        # 5) 持久化到 analysis_history
        await self._save_history(code, total_score, factor_results, signal, report)

        return report

    async def _save_history(self, code, total_score, factor_results, signal, report):
        try:
            scores = {k: fr.score for k, fr in factor_results.items()}
            async with AsyncSession() as session:
                record = AnalysisHistory(
                    stock_code=code,
                    score_total=total_score,
                    score_fund=scores.get("fundamental", 0),
                    score_ind=scores.get("industry", 0),
                    score_macro=scores.get("macro", 0),
                    score_flow=scores.get("fund_flow", 0),
                    score_sent=scores.get("sentiment", 0),
                    score_geo=scores.get("geo_external", 0),
                    signal=signal,
                    report_json=report,
                )
                session.add(record)
                await session.commit()
        except Exception as e:
            logger.warning("Failed to save analysis history: %s", e)

    async def _safe_factor(self, key: str, factor, code: str, name: str):
        try:
            fr = await factor.analyze(code, name)
            return (key, fr)
        except Exception as e:
            logger.error("Factor %s failed: %s", key, e)
            from factors.base import FactorResult
            return (key, FactorResult(factor_name=key, score=50, signal="neutral"))

    async def _safe_limit(self, code: str):
        try:
            return await self.limit_analyzer.analyze(code)
        except Exception as e:
            logger.error("Limit analyzer failed: %s", e)
            return {"zt_prob": 0, "dt_prob": 0, "seal": 0, "streak": 0}

    def _factor_label(self, key: str) -> str:
        return {
            "fundamental": "基本面",
            "industry": "行业",
            "macro": "宏观",
            "fund_flow": "资金流向",
            "sentiment": "情绪技术",
            "geo_external": "地缘",
        }.get(key, key)

    def _factor_desc(self, key: str, score: int, signal: str, detail: dict,
                     weight: int) -> str:
        """根据因子类型和得分生成深度中文分析"""
        level = (
            "优秀" if score >= 80 else "良好" if score >= 65
            else "一般" if score >= 45 else "偏弱" if score >= 30 else "较差"
        )
        signal_cn = {"bullish": "看多信号", "bearish": "看空信号", "neutral": "中性信号"}
        signal_str = signal_cn.get(signal, "中性")
        base = f"得分{score}分（{level}），发出{signal_str}。该因子占综合评分的{weight}%权重。"

        if key == "fundamental":
            return self._fundamental_desc(score, detail, base)

        elif key == "industry":
            return self._industry_desc(score, detail, base)

        elif key == "macro":
            return self._macro_desc(score, detail, base)

        elif key == "fund_flow":
            return self._fund_flow_desc(score, detail, base)

        elif key == "sentiment":
            return self._sentiment_desc(score, detail, base)

        elif key == "geo_external":
            return self._geo_desc(score, detail, base)

        return base

    # ── 各因子详细分析 ──

    def _fundamental_desc(self, score: int, detail: dict, base: str) -> str:
        lines = [base, ""]
        if not detail:
            lines.append("暂无基本面数据，无法进行深度分析。建议确认该股票代码是否正确，或稍后重试。")
            return "\n".join(lines)

        profit_growth = detail.get("净利润增速")
        revenue_growth = detail.get("营收增速")
        roe = detail.get("ROE")
        pe = detail.get("市盈率")

        if profit_growth is not None:
            if profit_growth >= 70:
                lines.append(f"• 净利润增速得分{profit_growth}，处于较高水平，表明公司盈利增长强劲，是推动股价上行的核心动力。")
            elif profit_growth >= 50:
                lines.append(f"• 净利润增速得分{profit_growth}，处于中等水平，利润增速尚可但缺乏爆发力，需关注后续季度能否加速。")
            else:
                lines.append(f"• 净利润增速得分{profit_growth}，偏低，反映公司盈利能力趋弱或出现下滑，需警惕基本面恶化风险。")

        if revenue_growth is not None:
            if revenue_growth >= 70:
                lines.append(f"• 营收增速得分{revenue_growth}，公司收入扩张迅速，市场份额可能在提升，与利润增长形成良性循环。")
            elif revenue_growth >= 50:
                lines.append(f"• 营收增速得分{revenue_growth}，收入增长平稳，属于稳健经营状态，但缺乏超预期因素。")
            else:
                lines.append(f"• 营收增速得分{revenue_growth}，收入端表现疲弱，若营收持续萎缩将拖累利润，需关注公司业务是否面临天花板。")

        if roe is not None:
            if roe >= 70:
                lines.append(f"• ROE得分{roe}，净资产收益率出色，说明公司运用股东资本的效率高，具备护城河特征。")
            elif roe >= 50:
                lines.append(f"• ROE得分{roe}，净资产收益率一般，资本回报率处于市场中游，不算突出但也不算差。")
            else:
                lines.append(f"• ROE得分{roe}，净资产收益率偏低，公司为股东创造回报的能力不足，长期持有需谨慎。")

        if pe is not None:
            if pe >= 70:
                lines.append(f"• 市盈率得分{pe}，估值处于合理区间，既未明显高估也未低估，安全边际适中。")
            elif pe >= 50:
                lines.append(f"• 市盈率得分{pe}，估值略偏高或偏低，需结合行业平均市盈率进一步判断。")
            else:
                lines.append(f"• 市盈率得分{pe}，估值偏离合理区间较远，过高则泡沫风险大，过低则可能存在基本面隐忧。")

        if not lines[1:]:
            lines.append("各项财务指标暂未获取到具体数值，基本面分析受限。")
        return "\n".join(lines)

    def _industry_desc(self, score: int, detail: dict, base: str) -> str:
        lines = [base, ""]
        if not detail:
            lines.append("暂无行业数据。")
            return "\n".join(lines)

        change = detail.get("行业涨跌幅")
        leading = detail.get("领涨行业") or detail.get("领涨板块")
        lagging = detail.get("领跌行业") or detail.get("领跌板块")

        if change is not None:
            if isinstance(change, (int, float)):
                chg_str = f"{change:+.2f}%"
                if change > 2:
                    lines.append(f"• 所属行业板块涨跌幅{chg_str}，板块处于强势上涨趋势中，个股容易获得板块效应加持，顺势做多胜率较高。")
                elif change > 0:
                    lines.append(f"• 所属行业板块涨跌幅{chg_str}，板块微涨，方向偏多但力度不足，板块轮动特征明显，需关注是否会进一步走强。")
                elif change > -2:
                    lines.append(f"• 所属行业板块涨跌幅{chg_str}，板块小幅回调，属于正常调整范围，不构成趋势逆转信号。")
                else:
                    lines.append(f"• 所属行业板块涨跌幅{chg_str}，板块明显走弱，可能遭遇行业利空或资金流出，个股即使基本面良好也难独善其身。")
            else:
                lines.append(f"• 行业涨跌幅：{change}")

        if leading:
            lines.append(f"• 当前领涨板块：{leading}，这些板块是市场资金的主攻方向。")
        if lagging:
            lines.append(f"• 当前领跌板块：{lagging}，这些板块短期承压，宜回避或减仓。")

        if score >= 70:
            lines.append("• 整体判断：行业因子偏多，板块效应有利于该股上涨，可积极参与。")
        elif score >= 45:
            lines.append("• 整体判断：行业因子中性，板块方向不明朗，个股走势更多依赖自身基本面。")
        else:
            lines.append("• 整体判断：行业因子偏空，板块下行压力较大，即使看好个股也应控制仓位。")
        return "\n".join(lines)

    def _macro_desc(self, score: int, detail: dict, base: str) -> str:
        lines = [base, ""]
        if not detail:
            lines.append("暂无宏观数据。")
            return "\n".join(lines)

        pmi = detail.get("PMI")
        cpi = detail.get("CPI")
        m2 = detail.get("货币供应") or detail.get("M2")
        lpr = detail.get("LPR")

        if pmi is not None:
            if isinstance(pmi, (int, float)):
                if pmi >= 70:
                    lines.append(f"• PMI得分{pmi}，制造业PMI处于扩张区间(>50)，经济景气度向好，为股市提供基本面支撑。")
                elif pmi >= 50:
                    lines.append(f"• PMI得分{pmi}，PMI在荣枯线附近徘徊，经济复苏力度偏弱，宏观面对股市的推动力有限。")
                else:
                    lines.append(f"• PMI得分{pmi}，PMI低于荣枯线，经济下行压力较大，宏观面偏空。")
            else:
                lines.append(f"• PMI：{pmi}")

        if cpi is not None:
            if isinstance(cpi, (int, float)):
                if cpi >= 65:
                    lines.append(f"• CPI得分{cpi}，通胀水平适中（约1-3%），温和通胀有利于企业盈利和股市估值。")
                elif cpi >= 45:
                    lines.append(f"• CPI得分{cpi}，通胀略偏离理想区间，通缩或高通胀均不利于股市，需持续观察。")
                else:
                    lines.append(f"• CPI得分{cpi}，通胀形势不佳，极端通缩或恶性通胀都将冲击市场信心。")
            else:
                lines.append(f"• CPI：{cpi}")

        if m2 is not None:
            if isinstance(m2, (int, float)):
                if m2 >= 65:
                    lines.append(f"• M2增速得分{m2}，货币供应充裕，流动性宽松有利于推升资产价格。")
                elif m2 >= 45:
                    lines.append(f"• M2增速得分{m2}，流动性中性，市场资金面不紧不松，股市缺乏额外的流动性红利。")
                else:
                    lines.append(f"• M2增速得分{m2}，货币供应偏紧，流动性收缩可能压制股市估值。")
            else:
                lines.append(f"• M2：{m2}")

        if lpr is not None:
            if isinstance(lpr, (int, float)):
                if lpr >= 65:
                    lines.append(f"• LPR利率得分{lpr}，利率处于低位或下行通道，低利率环境降低企业融资成本，利好股市。")
                elif lpr >= 45:
                    lines.append(f"• LPR利率得分{lpr}，利率水平中性，对股市的边际影响不大。")
                else:
                    lines.append(f"• LPR利率得分{lpr}，利率偏高或上行，将压制企业盈利和股市估值，偏利空。")
            else:
                lines.append(f"• LPR：{lpr}")

        if score >= 70:
            lines.append("• 综合判断：宏观环境整体偏暖，有利于A股上涨，可适当提升仓位。")
        elif score >= 45:
            lines.append("• 综合判断：宏观面喜忧参半，对股市没有明确的方向指引，维持中性判断。")
        else:
            lines.append("• 综合判断：宏观环境偏冷，经济下行风险较大，建议降低风险敞口。")
        return "\n".join(lines)

    def _fund_flow_desc(self, score: int, detail: dict, base: str) -> str:
        lines = [base, ""]
        if not detail:
            lines.append("暂无资金流向数据。")
            return "\n".join(lines)

        north = detail.get("北向资金")
        margin = detail.get("融资融券")
        main = detail.get("主力资金")
        dt_score = detail.get("龙虎榜")
        nb_ind = detail.get("北向个股")
        diag = detail.get("数据状态", "")

        if north is not None:
            if north >= 65:
                lines.append(f"• 北向资金得分{north}，外资（北向资金）近期净流入，表明海外资金看好A股市场，聪明钱的流入通常具有前瞻性。")
            elif north >= 45:
                lines.append(f"• 北向资金得分{north}，外资流向偏中性，没有明显的抄底或出逃信号。")
            else:
                lines.append(f"• 北向资金得分{north}，外资持续净流出，可能是避险信号，需关注是否有系统性和宏观风险。")

        if main is not None:
            if main >= 65:
                lines.append(f"• 主力资金得分{main}，主力资金（大单/超大单）近5日净流入，说明机构或游资在积极布局，短期上涨动能较强。")
            elif main >= 45:
                lines.append(f"• 主力资金得分{main}，主力资金流向偏中性，多空力量较为均衡，短期方向不明确。")
            else:
                lines.append(f"• 主力资金得分{main}，主力资金净流出，大资金在撤退，散户接盘风险增大，短期内股价承压。")

        if margin is not None:
            if margin >= 65:
                lines.append(f"• 融资融券得分{margin}，融资余额较高，杠杆资金做多意愿强烈，但也要注意过度融资带来的回调风险。")
            elif margin >= 45:
                lines.append(f"• 融资融券得分{margin}，融资余额处于正常水平，杠杆资金参与度一般。")
            else:
                lines.append(f"• 融资融券得分{margin}，融资余额偏低或融券压力较大，杠杆资金观望或偏空。")

        if nb_ind is not None:
            if nb_ind >= 65:
                lines.append(f"• 北向个股得分{nb_ind}，该股获得北向资金直接增持，外资对该股的认可度较高。")
            elif nb_ind < 45:
                lines.append(f"• 北向个股得分{nb_ind}，北向资金在减持该股，外资态度偏谨慎。")

        if dt_score is not None and dt_score != 50:
            if dt_score >= 70:
                lines.append(f"• 龙虎榜得分{dt_score}，该股登上龙虎榜且资金净买入，游资或机构在积极抢筹，短线爆发力强。")
            else:
                lines.append(f"• 龙虎榜得分{dt_score}，该股龙虎榜表现偏弱，上榜但资金净卖出，需警惕主力借高出货。")

        if diag:
            lines.append(f"• 数据获取状态：{diag}")

        if score >= 70:
            lines.append("• 资金面判断：多路资金共振流入，资金面偏多，上涨的资金驱动力充足。")
        elif score >= 45:
            lines.append("• 资金面判断：各路资金方向不一，资金面中性，缺乏明确的增量资金信号。")
        else:
            lines.append("• 资金面判断：资金呈流出态势，无论是外资、主力还是杠杆资金都在撤退，需高度警惕。")
        return "\n".join(lines)

    def _sentiment_desc(self, score: int, detail: dict, base: str) -> str:
        lines = [base, ""]
        if not detail:
            lines.append("暂无情绪与技术数据。")
            return "\n".join(lines)

        zt_emo = detail.get("涨停情绪")
        dt_risk = detail.get("跌停风险")
        hot = detail.get("市场热度")
        ma = detail.get("均线多头")
        rsi_key = next((k for k in detail if "RSI" in str(k)), None)
        macd = detail.get("MACD金叉")

        if zt_emo is not None:
            if zt_emo >= 70:
                lines.append(f"• 涨停情绪得分{zt_emo}，市场涨停氛围浓厚，赚钱效应显著，短线投机情绪高涨。")
            elif zt_emo >= 50:
                lines.append(f"• 涨停情绪得分{zt_emo}，涨停情绪中等，市场有一定的赚钱效应但不算火爆。")
            else:
                lines.append(f"• 涨停情绪得分{zt_emo}，涨停家数较少，市场缺乏赚钱效应，短线情绪低迷。")

        if dt_risk is not None:
            if dt_risk >= 70:
                lines.append(f"• 跌停风险得分{dt_risk}（高分=低风险），跌停家数少，市场恐慌情绪低，系统性杀跌风险较小。")
            elif dt_risk >= 45:
                lines.append(f"• 跌停风险得分{dt_risk}，部分个股出现跌停，但未形成大面积恐慌，属正常市场调整。")
            else:
                lines.append(f"• 跌停风险得分{dt_risk}，跌停家数较多，恐慌情绪在蔓延，需警惕踩踏式下跌。")

        if hot is not None:
            if hot >= 70:
                lines.append(f"• 市场热度得分{hot}，该股关注度排名靠前，市场热度高，交投活跃，流动性好。")
            elif hot >= 45:
                lines.append(f"• 市场热度得分{hot}，关注度处于中游，属于正常水平。")
            else:
                lines.append(f"• 市场热度得分{hot}，该股关注度偏低，可能是冷门股，流动性风险需关注。")

        if ma is not None:
            if ma >= 65:
                lines.append("• 均线系统呈多头排列（MA5>MA20），短期趋势向上，均线支撑有效，技术面偏多。")
            else:
                lines.append("• 均线系统呈空头排列（MA5<MA20），短期趋势向下，均线压制明显，技术面偏空。")

        if rsi_key:
            rsi_val = detail[rsi_key]
            if rsi_val == 30:
                lines.append("• RSI处于超卖区间（<30），短期超跌严重，技术性反弹需求较强，但抄底需等待企稳信号。")
            elif rsi_val == 70:
                lines.append("• RSI处于超买区间（>70），短期涨幅过大，获利盘回吐压力增加，追高风险较大。")
            else:
                lines.append("• RSI处于中性区间（30-70），多空力量均衡，没有明显的超买超卖信号。")

        if macd is not None:
            if macd >= 65:
                lines.append("• MACD金叉（DIF上穿DEA），为经典买入信号，表明多头力量正在增强，上涨趋势可能启动或加速。")
            else:
                lines.append("• MACD死叉（DIF下穿DEA），为经典卖出信号，空头力量占优，下跌趋势可能延续。")

        if score >= 70:
            lines.append("• 综合判断：市场情绪偏乐观，技术面多数指标看多，适合顺势做多。")
        elif score >= 45:
            lines.append("• 综合判断：情绪与技术面多空交织，方向不明确，建议等待更清晰的信号。")
        else:
            lines.append("• 综合判断：情绪悲观，技术面走弱，不宜盲目抄底，耐心等待趋势反转信号。")
        return "\n".join(lines)

    def _geo_desc(self, score: int, detail: dict, base: str) -> str:
        lines = [base, ""]
        if not detail:
            lines.append("暂无地缘风险评估数据。")
            return "\n".join(lines)

        hits = detail.get("地缘事件数", detail.get("风险关键词命中数", "未知"))

        if score >= 70:
            lines.append(f"• 地缘风险得分{score}，当前全球地缘政治形势相对平静，没有重大冲突或危机事件对A股造成明显冲击。风险关键词命中数：{hits}。")
            lines.append("• 该因子对综合评分的影响较小（权重仅5%），主要起预警作用。在没有黑天鹅事件的时期，该因子通常维持中性偏高得分。")
        elif score >= 45:
            lines.append(f"• 地缘风险得分{score}，监测到少量地缘政治相关新闻（命中数：{hits}），存在一定的外部不确定性，但尚未达到需要大幅调整仓位的程度。")
            lines.append("• 建议关注相关事件的后续发展，若事态升级则需及时调整策略。")
        else:
            lines.append(f"• 地缘风险得分{score}，当前地缘政治风险较高，监测到较多负面事件（命中数：{hits}），可能包括战争冲突、制裁升级、贸易摩擦等。")
            lines.append("• 高风险环境下市场避险情绪浓厚，资金倾向于流出风险资产，建议降低仓位或配置防御性板块。")

        return "\n".join(lines)

    def _get_stock_name(self, code: str) -> str:
        try:
            df = self.client.get_spot_df()
            # 先精确匹配代码
            row = df[df["代码"] == code]
            if row.empty:
                # 模糊匹配（中文名或部分代码）
                clean = code.replace("sz", "").replace("sh", "").replace("bj", "")
                row = df[
                    df["代码"].astype(str).str.contains(clean, na=False)
                    | df["名称"].astype(str).str.contains(code, na=False)
                ]
            if not row.empty:
                return str(row.iloc[0].get("名称", code))
            return code
        except Exception:
            return code
