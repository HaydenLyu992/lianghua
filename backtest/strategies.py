import backtrader as bt


class MACrossover(bt.Strategy):
    """均线金叉策略"""
    params = (("fast", 5), ("slow", 20))

    def __init__(self):
        self.ma_fast = bt.ind.SMA(self.data.close, period=self.params.fast)
        self.ma_slow = bt.ind.SMA(self.data.close, period=self.params.slow)
        self.crossover = bt.ind.CrossOver(self.ma_fast, self.ma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.sell()


class MomentumBreakout(bt.Strategy):
    """动量突破策略"""
    params = (("lookback", 20), ("holding", 10))

    def __init__(self):
        self.highest = bt.ind.Highest(self.data.high, period=self.params.lookback)
        self.bar_count = 0

    def next(self):
        if not self.position:
            if self.data.close[0] > self.highest[-1]:
                self.buy()
                self.bar_count = 0
        else:
            self.bar_count += 1
            if self.bar_count >= self.params.holding:
                self.sell()


class MeanReversion(bt.Strategy):
    """均值回归策略"""
    params = (("period", 20), ("threshold", 0.05))

    def __init__(self):
        self.sma = bt.ind.SMA(self.data.close, period=self.params.period)
        self.std = bt.ind.StdDev(self.data.close, period=self.params.period)

    def next(self):
        if self.sma[0] == 0:
            return
        zscore = (self.data.close[0] - self.sma[0]) / max(self.std[0], 0.01)
        if not self.position:
            if zscore < -self.params.threshold * 20:
                self.buy()
        elif zscore > self.params.threshold * 20:
            self.sell()


class RSIStrategy(bt.Strategy):
    """RSI 超买超卖策略"""
    params = (("period", 14), ("oversold", 30), ("overbought", 70))

    def __init__(self):
        self.rsi = bt.ind.RSI(self.data.close, period=self.params.period)

    def next(self):
        if not self.position:
            if self.rsi[0] < self.params.oversold:
                self.buy()
        elif self.rsi[0] > self.params.overbought:
            self.sell()


class BollingerBreakout(bt.Strategy):
    """布林带突破策略"""
    params = (("period", 20), ("devfactor", 2.0))

    def __init__(self):
        self.bb_top = bt.ind.BollingerBands(self.data.close, period=self.params.period,
                                            devfactor=self.params.devfactor).top
        self.bb_bot = bt.ind.BollingerBands(self.data.close, period=self.params.period,
                                            devfactor=self.params.devfactor).bot

    def next(self):
        if not self.position:
            if self.data.close[0] > self.bb_top[-1]:
                self.buy()
        elif self.data.close[0] < self.bb_bot[-1]:
            self.sell()


STRATEGIES = {
    "ma_cross": {"cls": MACrossover, "label": "均线金叉", "params": ["fast", "slow"]},
    "momentum": {"cls": MomentumBreakout, "label": "动量突破", "params": ["lookback", "holding"]},
    "mean_rev": {"cls": MeanReversion, "label": "均值回归", "params": ["period", "threshold"]},
    "rsi": {"cls": RSIStrategy, "label": "RSI超买超卖", "params": ["period", "oversold", "overbought"]},
    "bollinger": {"cls": BollingerBreakout, "label": "布林带突破", "params": ["period", "devfactor"]},
}
