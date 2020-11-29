import math
import talib as ta
from backfolio.core.utils import df_apply
from backfolio.strategy import RebalanceOnScoreSplitOrdersAggressiveBuySell

gr = math.sqrt(5) / 2 + 0.5


class MR01(RebalanceOnScoreSplitOrdersAggressiveBuySell):
    """
    A mean reversion strategy that buys the top performing coins based on the
    distance of closing price from its SMA. This distance is calculated for
    several periods based on fibonacci levels, and more weight is given to
    recent periods.

    Levels: [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

    Further, optimal version of strategy places limit orders at 1% above and
    below the last closing price for selling or buying assets. Assets are held
    for 1 tick and then, a sell order is placed. This sell order is modified
    at each tick until sold.
    """
    def __init__(self,
                 *args,
                 power=gr**2,
                 levels=[2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = levels
        self.power = power

    def calculate_scores(self, panel):
        score = 0
        close = panel[:, :, 'close']
        for idx, level in enumerate(self.levels):
            sma = close.rolling(level).mean()
            score += (1 - close / sma) * (len(self.levels) - idx)**self.power
        return score


class MR02(RebalanceOnScoreSplitOrdersAggressiveBuySell):
    """
    A mean reversion strategy that buys the top performing coins based on the
    distance of closing price from its SMA. This distance is calculated for
    several periods based on certain levels, and more weight is given to recent
    periods. Multiple such levels are considered in this strategy as opposed to
    `MR01` strategy.

    Levels:
        - [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        - [2, 4, 6, 9, 15, 24, 39, 63, 102, 166]

    Further, optimal version of strategy places limit orders at 1% above the
    last closing price for selling and at the last closing price for buying
    assets. Assets are held for 1 tick and then, a sell order is placed. This
    sell order is modified at each tick until sold.
    """
    def __init__(self,
                 *args,
                 markdn_buy=0,
                 markup_sell=1,
                 source='close',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source
        self.markdn_buy = markdn_buy
        self.markup_sell = markup_sell

    def calculate_scores(self, panel):
        levels = [[round(gr**(i + gr)) for i in range(10)],
                  [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]]

        source = panel[:, :, self.source]
        superscore = 0

        for level in levels:
            score = 0
            multi = list(reversed(level))
            for idx, period in enumerate(level):
                ma1 = source.rolling(period).mean()
                score += (1 - source / ma1) * multi[idx]
            score = score / sum(level)
            superscore += score
        return superscore


class MR03(RebalanceOnScoreSplitOrdersAggressiveBuySell):
    """
    A mean reversion strategy that buys the top performing coins based on the
    distance of closing price from its SMA. The distance calculation is same
    as that of MR02, however more calculations are done for obtaining average
    RSI and average price channel, which are further incorporated into this
    strategy.

    Further, optimal version of strategy places limit orders at 1% above the
    last closing price for selling and at the last closing price for buying
    assets. Assets are held for 1 tick and then, a sell order is placed. This
    sell order is modified at each tick until sold.
    """
    def calculate_scores(self, panel):
        levels = [[round(gr**(i + gr)) for i in range(10)],
                  [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]]

        score = 0
        close = panel[:, :, 'close']

        for level in levels:
            ss1, ss2, ss3 = 0, 0, 0
            multi = list(reversed(level))
            for idx, period in enumerate(level):
                sma = close.rolling(period).mean()
                rsi = df_apply(close, ta.RSI, period)
                center = (close.rolling(period).max() +
                          close.rolling(period).min()) / 2

                ss1 += (center / close - 1) * multi[idx]
                ss2 += (sma / close - 1) * multi[idx]
                ss3 += (100 - rsi) * multi[idx]
            score += ss1.rank(axis=1, pct=True)
            score += ss2.rank(axis=1, pct=True)
            score += ss3.rank(axis=1, pct=True)

        score[close.pct_change().shift(0) >= 0] = -1
        return score


class MR04(RebalanceOnScoreSplitOrdersAggressiveBuySell):
    """"""
    def calculate_scores(self, panel):
        levels = [[2, 3, 5, 8, 13, 21, 34, 55, 89, 144]]

        score, S1, S2, S3, S4 = 0, 0, 0, 0, 0
        close = panel[:, :, 'close']

        for level in levels:
            ss0, ss1, ss2, ss3 = 0, 0, 0, 0
            multi = list(reversed(level))
            for idx, period in enumerate(level):
                sma = close.rolling(period).mean()
                rsi = df_apply(close, ta.RSI, period)
                center = (close.rolling(period).max() +
                          close.rolling(period).min()) / 2
                ret = (1 / close).rolling(period).sum().multiply(
                    close, axis=1).div(period, axis=1)

                ss0 += (1 - ret)
                ss1 += (center / close - 1) * multi[idx]
                ss2 += (sma / close - 1) * multi[idx]
                ss3 += (100 - rsi) * multi[idx]

            score += ss0.rank(axis=1, pct=True)
            score += ss2.rank(axis=1, pct=True)
            #score += ss1.rank(axis=1, pct=True)
            #score += ss3.rank(axis=1, pct=True)
        return score
