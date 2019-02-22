import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from ..core.object import Advice
from ..core.event import StrategyAdviceEvent
from ..core.utils import fast_xs
from .base import BaseStrategy
from .mixins import *


class RebalanceOnScoreStrategy(
        StateMixin, MoneyManagementMixin, ScoringMixin, BlacklistedMixin,
        IndicatorsMixin, OrderMarkupMixin, OrderSizeMixin, RebalancingScheduleMixin,
        DataCleanerMixin, SelectedSymbolsMixin, NotifierMixin, BaseStrategy):
    """
    Strategy that buy top scoring assets, and sells other assets
    every given time interval.
    """
    def __init__(self, *args, aggressive=True, debug=True, **kwargs):
        """
        :param aggressive: whether to use aggressive mode for the strategy
        """
        super().__init__(*args, **kwargs)
        self._symbols = {}

        self.debug = debug
        self.aggressive = aggressive

    def selected_assets(self, data=None):
        """
        Allow child strategy to modify asset selection.
        """
        data = data if data is not None else self.data
        data = self.sorted_data(data)
        self.selected = super().selected_assets(data)
        return self.selected

    def rejected_assets(self, data=None, selected=None):
        """
        Allow child strategy to modify asset selection.
        """
        data = data if data is not None else self.data
        selected = self.selected if selected is None else selected
        if selected is not None:
            return data[~data.index.isin(selected.index)]
        return super().rejected_assets(data)

    def advice_investments_at_tick(self, _tick_event):
        # if we have no data for this tick, do nothing.
        if self.data.empty:
            return

        # transform tick data as per the child strategy,
        # and sort assets based on score assigned to them
        data = self.transform_tick_data(self.data)
        if hasattr(self, 'sorted_data'):
            data = self.sorted_data(data)

        self.data = data
        if data.empty or self.before_strategy_advice_at_tick():
            return

        # select the top performing assets based on their scores,
        # or weights and also, select coins to sell off.
        self.selected = selected = self.selected_assets(data)
        self.rejected = rejected = self.rejected_assets(data)

        # assign weights and required equity for each asset
        data['required_equity'] = self.set_required_equity_for_each_asset()
        current_equity = self.account.equity
        equity = self.portfolio.equity_per_asset
        if not self._symbols:
            self._symbols = {asset: self.datacenter.assets_to_symbol(asset)
                             for asset, _ in equity.items()}

        self.replenish_commission_asset_equity(equity)
        if hasattr(self, 'sell_blacklisted_assets'):
            self.sell_blacklisted_assets(equity)

        # first sell everything that is not in selected coins,
        # provided they are worth atleast 1% above the threshold.
        # If the asset is the one in which commission is being deducted,
        # ensure that we have it at a fixed percent of equity all the time.
        self.replenish_commission_asset_equity(equity, at=1/2)
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if (symbol in rejected.index and
                    symbol in data.index and symbol not in selected.index):
                asset_data = fast_xs(data, symbol)
                amount, percent = self.modify_order(asset, asset_data)
                if asset_equity > 1e-2:
                    self.sell_asset(asset, asset_equity, asset_data, symbol,
                                    amount=amount, percent=percent)
                elif asset_equity > 1e-3:
                    self.sell_asset(asset, asset_equity, asset_data, symbol,
                                    amount=amount, percent=percent,
                                    orders=1, when=(
                                        asset != self.context.commission_asset))

        # next, sell assets that have higher equity first
        self.replenish_commission_asset_equity(equity, at=1/2)
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if symbol not in selected.index:
                continue
            asset_data = fast_xs(data, symbol)
            if asset_equity > asset_data['required_equity'] and asset_equity > 1e-3:
                amount, percent = self.modify_order(asset, asset_data,
                    'rebalance_sell', asset_data[self.weight_col] * 100)
                self.sell_asset(asset, asset_equity, asset_data, symbol,
                                amount=amount, percent=percent)

        # finally, buy assets that have lower equity now
        self.replenish_commission_asset_equity(equity, at=1/2)
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if symbol not in selected.index:
                continue
            asset_data = fast_xs(data, symbol)
            if asset_equity < asset_data['required_equity']:
                amount, percent = self.modify_order(asset, asset_data,
                    'buy', asset_data[self.weight_col] * 100)
                self.buy_asset(asset, asset_equity, asset_data, symbol,
                               amount=amount, percent=percent)

        self.after_strategy_advice_at_tick()

    def at_each_tick_end(self):
        """
        This hook is called after the strategy has done sending advices and
        all orders have been queued up with the broker.

        In the aggressive mode of a strategy (controlled by self.aggressive),
        further buy/sell orders are queued up using this hook.
        """
        if not self.aggressive:
            return True

        if not hasattr(self, "data") or not hasattr(self, 'selected'):
            return True

        if not hasattr(self, 'already_buy'):
            self.already_buy = 2
        if not hasattr(self, 'already_sell'):
            self.already_sell = 2

        if self.markup_sell is None or self.markdn_buy is None:
            return True

        if self.context.live_trading():
            time.sleep(5) # have a break of 5 seconds when live trading to not DDOS

        # update account balance before adding new orders
        self.account._update_balance()

        data = self.data
        reprocess = False
        iterations = 5
        equity = self.portfolio.equity_per_asset
        selected = self.selected
        rejected = self.rejected
        comm_asset = self.context.commission_asset
        base_asset = self.context.base_currency
        current_equity = self.account.equity

        self.replenish_commission_asset_equity(equity, at=1/3)
        for asset, asset_equity in equity.items():
            free, total = self.account.free[asset], self.account.total[asset]
            if asset not in self._symbols or total < 1e-8:
                continue
            symbol = self._symbols[asset]
            remaining = free/total*asset_equity

            if (symbol in rejected.index and asset != comm_asset and
                    asset != base_asset and self.already_sell < iterations and
                    symbol in data.index and symbol not in selected.index and
                    remaining >= 1e-3):
                asset_data = fast_xs(data, symbol)
                amount, percent = self.modify_order(asset, asset_data, iteration=self.already_sell)
                self.sell_asset(asset, asset_equity, asset_data, symbol,
                                amount=amount, percent=percent,
                                markup_multiplier=self.already_sell)
                reprocess = True
        if reprocess:
            self.already_sell += 1

        remaining = self.account.free[base_asset]/current_equity*100
        comm_eq = self.portfolio.equity_per_asset[self.context.commission_asset]
        min_comm_eq = current_equity/100*self.min_commission_asset_equity
        available_eq = current_equity - min_comm_eq
        if remaining > 1 and self.already_buy < 5:
            self.replenish_commission_asset_equity(equity, at=1/3)

            if len(selected) == 0:
                if not reprocess:
                    reprocess = False
            else:
                for symbol in selected.index:
                    if (symbol not in self.data.index or
                            symbol in self.banned.index):
                        continue
                    asset_data = fast_xs(data, symbol)
                    amount, percent = self.modify_order(asset, asset_data,
                        'buy', remaining/len(selected),
                        iteration=self.already_buy, relative=True)
                    self.buy_asset(asset, asset_equity, asset_data, symbol,
                                   amount=amount, percent=percent,
                                   markdn_multiplier=self.already_buy,
                                   relative=True)
                self.already_buy += 1
                reprocess = True

        if reprocess:
            return False

        self.already_buy = 2
        self.already_sell = 2
