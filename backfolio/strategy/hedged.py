import numpy as np
from backfolio.strategy import BaseStrategy
from backfolio.strategy.mixins import *
from ..core.utils import fast_xs


class HedgedStrategy(
        StateMixin, MoneyManagementMixin, ScoringMixin, BlacklistedMixin,
        OrderMarkupMixin, OrderSizeMixin, RebalancingScheduleMixin,
        DataCleanerMixin, SelectedSymbolsMixin, NotifierMixin, BaseStrategy):
    """
    HedgedStrategy allows hedging multiple parametric variations of a single
    strategy. It is NOT meant to hedge different strategies together.

    All standard parameters are ignored from CHILD strategies and should be
    ascertained from the hedged strategy instead:
    - order size, flow and markup related parameters
    - assets, max_assets, selected_symbols, rebalance period
    - reserved_cash, min_commission_asset_equity
    - weighted parameter is always set to False. Strategies can not be weighted
      according to their scores individually. You can, however, set weights for
      a particular strategy when adding them to this strategy.

    You can, also, penalize an asset when it is rejected by a strategy. In effect,
    this will boost assets that are selected by more strategies.

    If you reselect_assets, assets recommended by child strategies will be
    scored based on their individual scores and top self.assets assets will
    be selected (weight based) only. Without this all assets recommended by
    child strategies are selected (weight based).
    """
    def __init__(self, *args, penalize=0, reselect_assets=False,
            aggressive=True, debug=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected    = {}
        self.rejected    = {}
        self.score_cols  = {}
        self.equity_cols = {}
        self.strategies  = {}

        self._symbols = {}
        self.debug = debug
        self.aggressive = aggressive
        self.weighted = False
        self.penalize = penalize
        self.reselect_assets = reselect_assets

    def with_child_strategies(self, fn=None):
        collection = {}
        for name, item in self.strategies.items():
            if fn is not None:
                collection[name] = fn(name, *item)
            else:
                collection[name] = yield(name, *item)
        return collection

    def reset(self, context):
        super().reset(context)
        for _, strat, _ in self.with_child_strategies():
            strat.reset(context)

    def add_strategy(self, name, strat, weight=1):
        self.strategies[name] = [strat, weight]
        return self.strategies

    def transform_history(self, panel):
        panel = super().transform_history(panel)
        for name, strat, _ in self.with_child_strategies():
            self.score_cols[name] = 'score_%s' % name
            panel.loc[:, :, self.score_cols[name]] = strat.calculate_scores(panel)
        panel.loc[:, :, 'score'] = panel[:, :, list(self.score_cols.values())].mean(axis=2)
        return panel

    def pass_variables_to_child_strategy(self, strat):
        for var in ["assets", "data", "tick", "blacklisted", "banned"]:
            setattr(strat, var, getattr(self, var))

    def advice_investments_at_tick(self, _tick_event):
        if self.data.empty:
            return

        data = self.data = self.transform_tick_data(self.data)
        if data.empty or self.before_strategy_advice_at_tick():
            return

        current_equity = self.account.equity
        equity = self.portfolio.equity_per_asset
        min_comm = self.min_commission_asset_equity
        comm_asset = self.context.commission_asset
        if not self._symbols:
            self._symbols = {asset: self.datacenter.assets_to_symbol(asset)
                             for asset, _ in equity.items()}

        available = current_equity * (1-min_comm/100)
        if self.reserved_cash:
            available *= (1-self.reserved_cash/100)

        self.before_strategy_advice_at_tick()
        for name, strat, weight in self.with_child_strategies():
            strat.score_col = "score_%s" % name
            self.equity_cols[name] = "equity_%s" % name
            self.pass_variables_to_child_strategy(strat)
            sel = self.selected[name] = strat.selected_assets()
            rej = self.rejected[name] = strat.rejected_assets()

            self.data[self.equity_cols[name]] = np.nan
            self.data.loc[sel.index, self.equity_cols[name]] = weight
            self.data.loc[rej.index, self.equity_cols[name]] = -weight*self.penalize

        data = self.data
        equity_cols = list(self.equity_cols.values())
        totals = data[equity_cols].sum(axis=1)
        totals = data[equity_cols][totals>totals.min()].mean(axis=1)
        totals = totals[totals>0]/totals[totals>0].sum()
        data['score'] = data[equity_cols].sum(axis=1)

        if self.reselect_assets:
            sel = self.all_selected = self.selected_assets(data)
            rej = self.all_rejected = self.rejected_assets(data, self.all_selected)
            data.loc[sel.index, 'required_equity'] = sel['score']/sel['score'].sum()*available
            data.loc[rej.index, 'required_equity'] = 0
        else:
            data['required_equity'] = data['score']*available
            self.all_selected = data[data['required_equity'] >  0]
            self.all_rejected = data[data['required_equity'] <= 0]

        self.replenish_commission_asset_equity(equity)
        if hasattr(self, 'sell_blacklisted_assets'):
            self.sell_blacklisted_assets(equity)

        # sell assets that were not selected by any strategies
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if (symbol in self.all_rejected.index and
                    symbol in data.index and symbol not in self.all_selected.index):
                asset_data = fast_xs(data, symbol)
                if asset_equity > asset_data['required_equity']/100 and asset_equity > 1e-2:
                    prices = self.selling_prices(symbol, asset_data)
                    N = asset_equity/current_equity*100
                    n = min_comm if asset == comm_asset else 0
                    if prices:
                        for price in prices:
                            x = (n-N)/len(prices)
                            self.order_percent(symbol, x, price, relative=True, side='SELL')
                    else:
                        self.order_percent(symbol, n, side='SELL')
                elif asset_equity > asset_data['required_equity']/100 and asset_equity > 1e-3:
                    n, prices = 0, self.selling_prices(symbol, asset_data)
                    if asset != comm_asset:
                        if prices:
                            self.order_percent(symbol, 0, prices[-1], side='SELL')
                        else:
                            self.order_percent(symbol, 0, side='SELL')

        self.replenish_commission_asset_equity(equity, at=1/2)
        # next, sell assets that have higher equity first
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if symbol not in self.all_selected.index:
                continue
            asset_data = fast_xs(data, symbol)
            if asset_equity > asset_data['required_equity'] and asset_equity > 1e-3:
                prices = self.selling_prices(symbol, asset_data)
                N = asset_equity/current_equity*100
                n = asset_data['required_equity']/current_equity*100
                n = max(min_comm, n) if asset == comm_asset else n
                if prices:
                    for price in prices:
                        x = (n-N)/len(prices)
                        self.order_percent(symbol, x, price, side='SELL', relative=True)
                else:
                    self.order_percent(symbol, n, side='SELL')

        self.replenish_commission_asset_equity(equity, at=1/2)
        # finally, buy assets that have lower equity now
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if symbol not in self.all_selected.index:
                continue
            asset_data = fast_xs(data, symbol)
            if asset_equity < asset_data['required_equity']:
                prices = self.buying_prices(symbol, asset_data)
                N = asset_equity/current_equity*100
                n = asset_data['required_equity']/current_equity*100
                n = n + min_comm if asset == comm_asset else n
                diff = asset_data['required_equity'] - asset_equity

                if diff > 0.01:
                    if prices:
                        for price in prices:
                            x = (n-N)/len(prices)
                            self.order_percent(symbol, x, price, side='BUY', relative=True)
                    else:
                        self.order_percent(symbol, n, side='BUY')
                elif prices:
                    self.order_percent(symbol, n, prices[-1], side='BUY')
                else:
                    self.order_percent(symbol, n, side='BUY')

        self.after_strategy_advice_at_tick()

    def at_each_tick_end(self):
        if not self.aggressive:
            return True

        if not hasattr(self, "data"):
            return True

        if not hasattr(self, 'already_buy'):
            self.already_buy = 2
        if not hasattr(self, 'already_sell'):
            self.already_sell = 2

        if self.markup_sell is None or self.markdn_buy is None:
            return True

        if self.context.live_trading():
            time.sleep(5) # have a break of 5 seconds when live trading to not DDOS

        reprocess = False
        data = self.data
        self.account._update_balance()
        equity = self.portfolio.equity_per_asset
        current_equity = self.account.equity
        comm_asset = self.context.commission_asset
        base_asset = self.context.base_currency

        self.replenish_commission_asset_equity(equity, at=1/3)
        for asset, asset_equity in equity.items():
            if asset not in self._symbols:
                continue
            free, total = self.account.free[asset], self.account.total[asset]
            if asset not in self._symbols or total < 1e-8:
                continue
            symbol = self._symbols[asset]
            remaining = free/total*asset_equity

            if (symbol in self.all_rejected.index and asset != comm_asset and
                    asset != base_asset and self.already_sell < 5 and
                    symbol in data.index and remaining >= 1e-3 and
                    symbol not in self.all_selected.index):
                orig = self.markup_sell
                self.markup_sell = [
                    self.markup_sell_func(curr, self.already_sell)
                    for curr in orig]
                reprocess = True
                # print("Selling asset: %s at %s markups" % (asset, self.markup_sell))
                asset_data = fast_xs(data, symbol)
                n, prices = 0, self.selling_prices(symbol, asset_data)
                N = asset_equity/current_equity*100
                if remaining >= 1e-2:
                    for price in prices:
                        x = (n-N)/len(prices)
                        self.order_percent(symbol, x, price, relative=True)
                else:
                    self.order_percent(symbol, 0, prices[-1], relative=True)
                self.markup_sell = orig
        if reprocess:
            self.already_sell += 1

        act_eq = current_equity*(1-self.min_commission_asset_equity/100)
        if self.reserved_cash:
            act_eq *= (1-self.reserved_cash/100)

        remaining = self.account.free[base_asset]/current_equity*100
        comm_eq = self.portfolio.equity_per_asset[self.context.commission_asset]
        min_comm_eq = current_equity/100*self.min_commission_asset_equity
        available_eq = current_equity - min_comm_eq
        if remaining > 1 and self.already_buy < 5:
            self.replenish_commission_asset_equity(equity, at=1/3)
            if len(self.all_selected) == 0:
                if not reprocess:
                    reprocess = False
            else:
                orig = self.markdn_buy
                self.markdn_buy = [
                    self.markdn_buy_func(curr, self.already_buy)
                    for curr in orig]
                for symbol in self.all_selected.index:
                    asset_data = fast_xs(data, symbol)
                    prices = self.buying_prices(symbol, asset_data)
                    weight = asset_data['required_equity']/data['required_equity'].sum()
                    weight /= len(self.all_selected)*len(prices)
                    for price in prices:
                        self.order_percent(symbol, remaining*weight, price,
                                           side='BUY', relative=True)
                self.already_buy += 1
                self.markdn_buy = orig
                reprocess = True

        if reprocess:
            return False

        self.already_buy = 2
        self.already_sell = 2
