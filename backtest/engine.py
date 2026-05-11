import logging
from datetime import datetime, timedelta
from typing import Type
import pandas as pd
import backtrader as bt
import backtrader.analyzers as btanalyzers

from core.akshare_client import AkShareClient

logger = logging.getLogger(__name__)


class BacktestEngine:
    """回测引擎：封装 Backtrader，输入股票代码+参数，输出绩效报告。"""

    def __init__(self, client: AkShareClient):
        self.client = client

    def run(
        self,
        code: str,
        strategy_cls: Type[bt.Strategy],
        start: str,
        end: str,
        cash: float = 100_000,
        **strategy_kwargs,
    ) -> dict:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=0.0003)  # 万三佣金
        cerebro.addstrategy(strategy_cls, **strategy_kwargs)
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
        cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(btanalyzers.Returns, _name="returns")

        data = self._load_data(code, start, end)
        if data is None:
            return {"error": f"无法获取 {code} 的历史数据"}

        cerebro.adddata(data)
        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        end_value = cerebro.broker.getvalue()

        if not results:
            return {"error": "回测运行失败"}

        strat = results[0]
        return self._build_report(strat, start_value, end_value, cash, start, end)

    def _load_data(self, code: str, start: str, end: str) -> bt.feeds.PandasData | None:
        try:
            df = self.client.get_daily_kline(code)
            if df.empty:
                return None

            df["日期"] = pd.to_datetime(df["日期"])
            df = df[(df["日期"] >= start) & (df["日期"] <= end)]
            df.set_index("日期", inplace=True)
            df.columns = [c.lower() for c in df.columns]

            col_map = {}
            for col in df.columns:
                if "开" in col: col_map[col] = "open"
                elif "高" in col: col_map[col] = "high"
                elif "低" in col: col_map[col] = "low"
                elif "收" in col: col_map[col] = "close"
                elif "量" in col or "volume" in col: col_map[col] = "volume"

            df.rename(columns=col_map, inplace=True)
            for c in ["open", "high", "low", "close", "volume"]:
                if c not in df.columns:
                    return None

            return bt.feeds.PandasData(dataname=df)
        except Exception as e:
            logger.error("Failed to load data for %s: %s", code, e)
            return None

    def _build_report(self, strat, start_val, end_val, cash, start, end) -> dict:
        total_return = (end_val - cash) / cash * 100
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()

        return {
            "start_value": cash,
            "end_value": round(end_val, 2),
            "total_return": round(total_return, 2),
            "total_return_amount": round(end_val - cash, 2),
            "sharpe_ratio": round(sharpe.get("sharperatio", 0) or 0, 2),
            "max_drawdown": round(drawdown.get("max", {}).get("drawdown", 0) or 0, 2),
            "win_rate": round(
                (trades.get("won", {}).get("total", 0) / max(trades.get("total", {}).get("total", 1), 1)) * 100, 1
            ),
            "total_trades": trades.get("total", {}).get("total", 0),
            "start": start,
            "end": end,
        }
