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
    def __init__(self, *args, source='close', **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source

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
                # rsi = df_apply(close.ffill(), ta.RSI, period)
                center = (close.rolling(period).max() +
                          close.rolling(period).min()) / 2

                ss1 += (center / close - 1) * multi[idx]
                ss2 += (sma / close - 1) * multi[idx]
                # ss3 += (100 - rsi) * multi[idx]
            score += ss1.rank(axis=1, pct=True)
            score += ss2.rank(axis=1, pct=True)
            # score += ss3.rank(axis=1, pct=True)

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
                # rsi = df_apply(close, ta.RSI, period)
                center = (close.rolling(period).max() +
                          close.rolling(period).min()) / 2
                ret = (1 / close).rolling(period).sum().multiply(
                    close, axis=1).div(period, axis=1)

                ss0 += (1 - ret)
                ss1 += (center / close - 1) * multi[idx]
                ss2 += (sma / close - 1) * multi[idx]
                # ss3 += (100 - rsi) * multi[idx]

            score += ss0.rank(axis=1, pct=True)
            score += ss2.rank(axis=1, pct=True)
            #score += ss1.rank(axis=1, pct=True)
            #score += ss3.rank(axis=1, pct=True)
        return score


class MR06(RebalanceOnScoreSplitOrdersAggressiveBuySell):
    def mr06_loss_alpha(self,
                        close,
                        periods=[2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
                        freq=1,
                        pr=1,
                        mode='recent'):
        A1, A2, A3 = 0, 0, 0

        for idx, pd in enumerate([freq * x for x in periods]):
            m = min(periods[idx], periods[len(periods) - 1 - idx])**pr
            m = pd**pr if mode == 'trend' else m
            m = 1 / pd**pr if mode == 'recent' else m

            A1 -= (close.pct_change(pd)) * m
            A2 += (close.rolling(pd).mean() / close - 1) * m
            A3 -= ((1 / close).rolling(pd).sum() * close - 1) * m
        return alpharank(A1, A2, A2, A3)

    def mr06_regime(self,
                    close,
                    periods=[2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
                    freq=1):
        PC = close.pct_change(freq)
        for p in [freq * x for x in periods]:
            PC += close.pct_change(p) * p
        TR = PC[PC > 0].sum(axis=1) / \
            (PC[PC > 0].sum(axis=1)-PC[PC < 0].sum(axis=1))

        XR = TR.copy()
        XR[:] = 1
        for i in range(48, 61):
            XR[TR.rolling(i).mean() < 0.45] = -1

        return XR

    def calculate_scores(self, panel):
        CL = panel.loc[:, :, 'close']

        # regime indicator
        XR = self.mr06_regime(CL, freq=1)

        # alpha for different regimes
        S1 = self.mr06_loss_alpha(CL, freq=2, pr=1, mode='recent')  # default
        S2 = self.mr06_loss_alpha(CL, freq=2, pr=1,
                                  mode='balanced')  # downtrend alt

        S1[XR == -1] = S2
        S1[CL < 100e-8] = -1
        return S1


class MRN01(MR06):
    def __init__(self,
                 *args,
                 min_flow_rank=0.7,
                 exclude_symbols=['BNB/BTC'],
                 exclude_low_price_assets=100e-8,
                 flow_check_periods=[12, 48, 48 * 14, 48 * 30],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_check_periods = flow_check_periods
        self.min_flow_rank = min_flow_rank
        self.exclude_symbols = exclude_symbols
        self.exclude_low_price_assets = exclude_low_price_assets

    def calculate_alpha(self, panel):
        CL = panel.loc[:, :, 'close']
        XR = self.mr06_loss_alpha(CL, mode='recent')
        XT = self.mr06_loss_alpha(CL, mode='trend')
        return alpharank(XR, XT)

    def calculate_scores(self, panel):
        CL = panel.loc[:, :, 'close']
        VO = panel.loc[:, :, 'volume']
        FL = VO * CL

        S1 = self.calculate_alpha(panel)
        S1[VO == 0] = -1
        S1[S1.index.isin(self.exclude_symbols)] = -1

        if self.exclude_low_price_assets is not None:
            S1[CL < self.exclude_low_price_assets] = -1

        for prd in self.flow_check_periods:
            ma = FL.rolling(prd, min_periods=2).mean()
            S1[ma.rank(axis=1, pct=True) < self.min_flow_rank] = -1

        return S1
