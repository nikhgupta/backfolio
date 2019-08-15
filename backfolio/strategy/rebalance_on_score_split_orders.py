import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from ..core.object import Advice
from ..core.event import StrategyAdviceEvent
from ..core.utils import fast_xs
from .rebalance_on_score import RebalanceOnScoreStrategy


class RebalanceOnScoreSplitOrders(RebalanceOnScoreStrategy):
    def __init__(self, *args, **kwargs):
        defaults = {"markdn_buy": [0.9,1.1,1], "markup_sell": [1.1,0.9,1]}
        kwargs = {**defaults, **kwargs}
        super().__init__(*args, **kwargs)

    def selling_prices(self, _symbol, data):
        if self.markup_sell is not None:
            price = data['price'] if 'price' in data else data['close']
            return [price*(1+x/100) for x in self.markup_sell]

    def buying_prices(self, _symbol, data):
        if self.markdn_buy is not None:
            price = data['price'] if 'price' in data else data['close']
            return [price*(1-x/100) for x in self.markdn_buy]

    def advice_investments_at_tick(self, _tick_event):
        # if we have no data for this tick, do nothing.
        if self.data.empty:
            return

        if hasattr(self, "init_strategy_advice_at_tick"):
            self.init_strategy_advice_at_tick()

        # increase number of assets if cash allows so.
        self.increase_assets_if_required()

        # transform tick data as per the child strategy,
        # and sort assets based on score assigned to them
        data = self.transform_tick_data(self.data)
        data = self.sorted_data(data)
        self.data = data
        if data.empty or (
                'score' not in data.columns and 'weight' not in data.columns):
            return

        # select the top performing assets based on their scores,
        # or weights and also, select coins to sell off.
        selected = self.selected_assets(data)
        rejected = self.rejected_assets(data)

        # assign weights and required equity for each asset
        data = self.assign_weights_and_required_equity(data, selected)
        equity = self.portfolio.equity_per_asset
        if not self._symbols:
            self._symbols = {asset: self.datacenter.assets_to_symbol(asset)
                             for asset, _ in equity.items()}

        if hasattr(self, "before_strategy_advice_at_tick"):
            self.before_strategy_advice_at_tick()

        # if we need to wait for rebalancing, do nothing.
        if self.rebalance_required(data, selected, rejected):
            self.broker.cancel_pending_orders()
        elif self.rebalance:
            return

        min_comm = self.min_commission_asset_equity
        comm_sym = self._symbols[self.context.commission_asset]
        if equity[self.context.commission_asset] < min_comm/100*self.account.equity:
            self.order_percent(comm_sym, min_comm, side='BUY')

        # first sell everything that is not in selected coins,
        # provided they are worth atleast 1% above the threshold.
        # If the asset is the one in which commission is being deducted,
        # ensure that we have it at a fixed percent of equity all the time.
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]

            if (symbol in rejected.index and
                    symbol in data.index and symbol not in selected.index):
                asset_data = fast_xs(data, symbol)
                if ((np.isnan(asset_data['required_equity']) or
                        asset_equity > asset_data['required_equity']/100) and
                        asset_equity > 1e-2):
                    n, prices = 0, self.selling_prices(symbol, asset_data)
                    N = asset_equity/self.account.equity*100
                    if asset == self.context.commission_asset:
                        n = self.min_commission_asset_equity
                    if prices:
                        for price in prices:
                            x = (n-N)/len(prices)
                            self.order_percent(symbol, x, price, relative=True, side='SELL')
                    else:
                        self.order_percent(symbol, (n-N), side='SELL', relative=True)
                elif asset_equity > asset_data['required_equity']/100 and asset_equity > 1e-3:
                    n, prices = 0, self.selling_prices(symbol, asset_data)
                    if asset != self.context.commission_asset:
                        if prices:
                            self.order_percent(symbol, 0, prices[-1], side='SELL')
                        else:
                            self.order_percent(symbol, 0, side='SELL')

        # next, sell assets that have higher equity first
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if symbol not in selected.index:
                continue
            asset_data = fast_xs(data, symbol)
            if (asset_equity > asset_data['required_equity'] and asset_equity > 1e-3):
                prices = self.selling_prices(symbol, asset_data)
                n = asset_data['weight'] * 100
                N = asset_equity/self.account.equity*100
                if asset == self.context.commission_asset:
                    n = max(self.min_commission_asset_equity, n)
                if prices:
                    for price in prices:
                        x = (n-N)/len(prices)
                        self.order_percent(symbol, x, price, side='SELL', relative=True)
                else:
                    self.order_percent(symbol, n, side='SELL')

        # finally, buy assets that have lower equity now
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if symbol not in selected.index:
                continue
            asset_data = fast_xs(data, symbol)
            if asset_equity < asset_data['required_equity']:
                prices = self.buying_prices(symbol, asset_data)
                n = asset_data['weight'] * 100
                N = asset_equity/self.account.equity*100
                diff = n*self.account.equity/100 - asset_equity
                if (asset == self.context.commission_asset and
                        asset_equity < min_comm):
                    self.order_percent(symbol, self.min_commission_asset_equity, side='BUY')
                    n -= self.min_commission_asset_equity
                if diff > 0.01:
                    if prices:
                        for price in prices:
                            x = (n-N)/len(prices)
                            #print("BUY: ", asset, x, n, N, price)
                            self.order_percent(symbol, x, price, side='BUY', relative=True)
                    else:
                        self.order_percent(symbol, n, side='BUY')
                elif prices:
                    self.order_percent(symbol, n, prices[-1], side='BUY')
                else:
                    self.order_percent(symbol, n, side='BUY')

        if hasattr(self, "after_strategy_advice_at_tick"):
            self.after_strategy_advice_at_tick()

        self._last_rebalance = self.tick.time
        self._save_state()

    def transform_order_calculation(self, advice, cost, quantity, price):
        """
        Hook into broker, before it creates an order for execution,
        to override the final order cost, quantity or price.
        Supplied `price` is limit price for LIMIT orders, else None.

        You should return a tuple consisting of new order cost,
        quantity and LIMIT price. If cost, quantity and price are all changed,
        it is the quantity that will be recalculated to fit cost and price.

        Do NOT set price for MARKET orders.
        You can return `(0, 0, 0)` to not place this order.
        """
        n = (1-self.min_commission_asset_equity/100)
        if cost > 0 and cost > self.account.free[advice.base] * n:
            cost = self.account.free[advice.base] * n

        # if the equity vs quantity calculation, messes up our
        # ordering side, ensure that we still rebalance, but
        # we use the correct LIMIT price this time, but
        # we do this only when the cost of order is less than
        # 3% of our account equity.
        if advice.is_limit and advice.asset != self.context.commission_asset:
            th = self.account.equity*self.min_commission_asset_equity/100
            if advice.is_buy and cost < 0 and abs(cost) < th:
                advice.side = "SELL"
                if price:
                    price = self.selling_prices(
                        advice.symbol(self), {"price": advice.last_price})
                if price:
                    price = price[0]

            elif advice.is_sell and cost > 0 and abs(cost) < th:
                advice.side = "BUY"
                if price:
                    price = self.buying_prices(
                        advice.symbol(self), {"price": advice.last_price})
                if price:
                    price = price[0]
        return (cost, quantity, price)
