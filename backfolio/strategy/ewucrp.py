import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from ..core.object import Advice
from ..core.event import StrategyAdviceEvent
from ..core.utils import fast_xs
from .base import BaseStrategy


# TODO: buy commission asset if we get low on it
class EwUCRPStrategy(BaseStrategy):
    """
    A simple strategy which invests our capital equally amongst all assets
    available on an exchange, and rebalances it on each tick.

    In principle, a EwUCRP (Equally-weighted Uniformly Constant Rebalanced
    Portfolio) is hard to beat, which makes this strategy an excellent
    benchmark for our use cases.

    Assets that have been delisted are removed from our holdings.

    This portfolio strategy is different than the Buy and Hold strategy above,
    in that rebalance is done daily here as opposted to when a new asset is
    introduced on the exchange.
    """
    def __init__(self, rebalance=1, max_order_size=None,
                 flow_period=12, flow_multiplier=0.025):
        super().__init__()
        self.rebalance = rebalance
        self.flow_period = flow_period
        self.flow_multiplier = flow_multiplier
        self.max_order_size = max_order_size
        self._last_rebalance = None

    def before_trading_start(self):
        self.broker.min_order_size = 0.001

    def transform_history(self, panel):
        panel = super().transform_history(panel)
        close = panel[:, :, 'close']
        volume = panel[:, :, 'volume']
        panel.loc[:, :, 'flow'] = (close*volume).rolling(
            self.flow_period).mean()
        return panel

    def rebalance_required(self):
        time = self.tick.time
        timediff = pd.to_timedelta(self.datacenter.timeframe*self.rebalance)
        return (not self._last_rebalance or
                time >= self._last_rebalance + timediff)

    def order_percent(self, symbol, amount, price=None, max_cost=None):
        max_flow = fast_xs(self.data, symbol)['flow'] * self.flow_multiplier
        if self.max_order_size:
            max_flow = max(max_flow, self.max_order_size)
        max_cost = min(max_cost, max_flow) if max_cost else max_flow
        args = ('MARKET', None, max_cost)
        if price:
            args = ('LIMIT', price, max_cost)
        self.order_target_percent(symbol, amount, *args)

    def advice_investments_at_tick(self, tick_event):
        n = 100./(1+len(self.data))

        rebalanced = self.account.equity*n/100
        equity = self.portfolio.equity_per_asset

        if self.rebalance_required():
            self.broker.cancel_pending_orders()
        else:
            return

        # sell assets that have higher equity first
        # once we have cash available, buy assets that have lower equity now
        rebalance = [k for k, v in equity.items() if v > rebalanced]
        rebalance += [k for k, v in equity.items() if v < rebalanced]
        for asset in rebalance:
            symbol = self.datacenter.assets_to_symbol(asset)
            if symbol not in self.data.index:
                continue
            price = fast_xs(self.data, symbol)['close']
            self.order_percent(symbol, n, price)

        self._last_rebalance = self.tick.time

