from .rebalance_on_score_split_orders_aggressive_buy_sell import RebalanceOnScoreSplitOrdersAggressiveBuySell
from ..indicator.compute import *


class AlphaStrategy(RebalanceOnScoreSplitOrdersAggressiveBuySell):
    """
    A strategy that can be used with Alpha101 signals.
    Remember that, by default, itt rebalances every tick.
    """
    def alpha(self, panel):
        """
        Return the panel as is. Child strategies should override this method,
        if required.
        """
        return panel

    def calculate_scores(self, panel):
        self.open = panel[:, :, 'open'].copy()
        self.high = panel[:, :, 'high'].copy()
        self.low = panel[:, :, 'low'].copy()
        self.close = panel[:, :, 'close'].copy()
        self.volume = panel[:, :, 'volume'].copy()
        self.returns = self.close.pct_change()

        return self.alpha(panel)


class Alpha002Strategy(AlphaStrategy):
    def alpha(self, panel):
        return correlation(rank(delta(log(self.volume), 2)),
                           rank((self.close - self.open) / self.open), 6)
