import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from ..core.object import Advice
from ..core.event import StrategyAdviceEvent
from ..core.utils import fast_xs
from .base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """
    A simple buy and hold strategy that buys a new asset whenever a new
    asset is encountered in the tick data. We rebalance our portfolio on this
    tick to include this new asset, such that all assets always have equal
    weights in our portfolio.

    Assets that have been delisted have no impact on portfolio (other than
    zero returns from them) as per the strategy logic.

    We ensure that we, also, keep an equal weight of cash/primary currency.

    This portfolio strategy is different than a UCRP strategy in that rebalance
    is only done when a new asset is introduced on the exchange, which may take
    weeks/months, as opposed to UCRP where rebalance is done daily, regardless.
    """

    def __init__(self, lag=0):
        super().__init__()
        self.lag = lag
        self.session_fields += ['added', 'pending', 'data_store']

    def reset(self, context):
        super().reset(context)
        self.added = []
        self.pending = []
        self.data_store = []

    # TODO: buy commission asset if we get low on it
    def advice_investments_at_tick(self, tick_event):
        """
        We sell our assets first so that we do have sufficient liquidity.
        Afterwards, we will issue a buy order.
        """
        new_symbols = [s for s in self.data.index if s not in self.added]
        self.pending.append(new_symbols)
        self.data_store.append(self.data)
        for sym in new_symbols:
            self.added.append(sym)

        if len(self.pending) < self.lag+1:
            return
        new_symbols = self.pending[-1-self.lag]
        data = self.data_store[-1-self.lag]

        if not new_symbols:
            return

        n = 100./(1+len(data))
        rebalanced = self.account.equity*n/100
        equity = self.portfolio.equity_per_asset

        # sell assets that have higher equity first
        # once we have cash available, buy assets that have lower equity
        rebalance = [k for k, v in equity.items() if v > rebalanced]
        rebalance += [k for k, v in equity.items() if v < rebalanced]
        for asset in rebalance:
            symbol = self.datacenter.assets_to_symbol(asset)
            if symbol not in data.index:
                continue
            self.order_target_percent(symbol, n, 'MARKET')
