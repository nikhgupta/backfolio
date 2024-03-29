import time
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from ..core.object import Advice
from ..core.event import StrategyAdviceEvent
from ..core.utils import fast_xs
from .rebalance_on_score_split_orders import RebalanceOnScoreSplitOrders


class RebalanceOnScoreSplitOrdersAggressiveBuySell(RebalanceOnScoreSplitOrders
                                                   ):
    def __init__(self,
                 *args,
                 markdn_buy_func=None,
                 markup_sell_func=None,
                 max_buy_loop=3,
                 max_sell_loop=3,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.max_buy_loop = max_buy_loop
        self.max_sell_loop = max_sell_loop

        if markdn_buy_func is not None:
            self.markdn_buy_func = markdn_buy_func
        else:
            self.markdn_buy_func = lambda curr, i: i * curr  # 2 => 2,4,6,8...

        if markup_sell_func is not None:
            self.markup_sell_func = markup_sell_func
        else:
            self.markup_sell_func = lambda curr, i: (
                i - 1) * 0.5 + curr  # 2 => 2,2.5,3...

    def at_each_tick_end(self):
        if not hasattr(self, "data"):
            return True

        if not hasattr(self, 'already_buy'):
            self.already_buy = 2
        if not hasattr(self, 'already_sell'):
            self.already_sell = 2

        if self.markup_sell is None or self.markdn_buy is None:
            return True

        if self.context.live_trading():
            time.sleep(
                5)  # have a break of 5 seconds when live trading to not DDOS

        reprocess = False
        data = self.data
        self.account._update_balance()
        equity = self.portfolio.equity_per_asset
        selected = self.selected_assets(data)
        rejected = self.rejected_assets(data)
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            rem = self.account.free[asset] / self.account.total[
                asset] * asset_equity if self.account.total[
                    asset] >= 1e-8 else 0
            if (symbol in rejected.index
                    and asset != self.context.commission_asset
                    and asset != self.context.base_currency
                    and self.already_sell < self.max_sell_loop
                    and symbol in data.index and symbol not in selected.index
                    and rem >= 1e-3):
                orig = self.markup_sell
                self.markup_sell = [
                    self.markup_sell_func(curr, self.already_sell)
                    for curr in orig
                ]
                reprocess = True
                # print("Selling asset: %s at %s markups" % (asset, self.markup_sell))
                asset_data = fast_xs(data, symbol)
                n, prices = 0, self.selling_prices(symbol, asset_data)
                N = asset_equity / self.account.equity * 100
                if rem >= 1e-2:
                    for price in prices:
                        x = (n - N) / len(prices)
                        self.order_percent(symbol, x, price, relative=True)
                else:
                    self.order_percent(symbol, 0, prices[-1], relative=True)
                self.markup_sell = orig
        if reprocess:
            self.already_sell += 1

        act_eq = self.account.equity * (1 -
                                        self.min_commission_asset_equity / 100)
        rem = self.account.free[self.context.base_currency] / act_eq * 100
        comm_eq = self.portfolio.equity_per_asset[
            self.context.commission_asset]
        min_comm = self.account.equity / 100 * self.min_commission_asset_equity
        if rem > 1 and self.already_buy < self.max_buy_loop:
            if comm_eq > min_comm / 2:
                if len(selected) == 0:
                    if not reprocess:
                        reprocess = False
                else:
                    orig = self.markdn_buy
                    self.markdn_buy = [
                        self.markdn_buy_func(curr, self.already_buy)
                        for curr in orig
                    ]
                    for symbol in selected.index:
                        asset_data = fast_xs(self.data, symbol)
                        prices = self.buying_prices(symbol, asset_data)
                        # print("BUYing %s at %s markdn" % (symbol, self.markdn_buy))
                        for price in prices:
                            self.order_percent(symbol,
                                               rem /
                                               (len(selected) * len(prices)),
                                               price,
                                               side='BUY',
                                               relative=True)
                    self.already_buy += 1
                    self.markdn_buy = orig
                    reprocess = True

        if reprocess:
            return False

        self.already_buy = 2
        self.already_sell = 2
