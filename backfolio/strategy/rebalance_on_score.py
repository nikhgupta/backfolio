import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from ..core.object import Advice
from ..core.event import StrategyAdviceEvent
from ..core.utils import fast_xs
from .base import BaseStrategy


class RebalanceOnScoreStrategy(BaseStrategy):
    """
    Strategy that buy top scoring assets, and sells other assets
    every given time interval.
    """
    def __init__(self, rebalance=1, assets=3, max_assets=13,
                 max_order_size=0, min_order_size=0.001,
                 flow_period=1, flow_multiplier=2.5,
                 markup_sell=1, markdn_buy=1, reserved_cash=0,
                 min_commission_asset_equity=3, weighted=False,
                 min_ticks=0, debug=True):
        """
        :param rebalance: number of ticks in a rebalancing period
        :param assets: Number of assets strategy should buy.
        :param max_assets: If cash allows, try increasing assets
            to buy gradually uptil this value. This will only be
            done, when order size is limited using `max_order_size`
            or flow control.
        :param min_order_size: Minimum allowed order size in base asset.
        :param max_order_size: Maximum allowed order size in base asset
            (surpassed if flow is more than this amount). Set to `None`
            to not limit order size using this. To completely, disable
            order size limiting, also set `flow_multiplier` to `0`.
        :param flow_period: How many ticks to consider for flow based
            max order sizing?
        :param flow_multiplier: (in %) %age of flow used. Flow is the
            average BTC flow for that asset in given period. Set to `0`
            to disable flow based order size limiting.
        :param markdn_buy: (in %) %age of price to decrease buy orders by.
        :param markup_sell: (in %) %age of price to increase sell orders by.
        :param min_commission_asset_equity: (in %) capital to keep in
            commission asset
        :param reserved_cash: (in %) amount of cash wrt total account equity
            to keep in reserve at all times.
        """
        super().__init__(min_ticks=min_ticks)
        self.rebalance = rebalance
        self.assets = assets
        self.max_assets = max_assets
        self.min_order_size = min_order_size
        self.max_order_size = max_order_size
        self.flow_period = flow_period
        self.flow_multiplier = flow_multiplier
        self.markdn_buy = markdn_buy
        self.markup_sell = markup_sell
        self.min_commission_asset_equity = min_commission_asset_equity
        self.reserved_cash = reserved_cash
        self.weighted = weighted
        self.debug = debug
        self.min_ticks = min_ticks

        self.session_fields = ['state']
        self._last_rebalance = None
        self._state = []
        self._symbols = {}

    def reset(self, context):
        """
        Resets the component for reuse.
        Load current state from session at each reset.
        """
        super().reset(context)
        self._load_state()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state = val

    @property
    def current_state(self):
        return self.state[-1] if len(self.state) > 0 else {}

    def _load_state(self):
        """
        Load current state of session.
        Specifically, set `_last_rebalance` time.
        """
        if 'last_rebalance' in self.current_state:
            self._last_rebalance = self.current_state["last_rebalance"]
        return self.current_state

    def _save_state(self, **data):
        """
        Save current state of session and any given data at this time.
        Specifically, add `_last_rebalance` time to session state.
        """
        self._state.append({
            **data, **{
                "last_rebalance": self._last_rebalance,
                "time": self.tick.time}})

    def before_trading_start(self):
        """
        Ensure that broker rejects any orders with cost
        less than the `min_order_size` specified for this
        strategy.

        You can, similarily, set `max_order_size` for the
        broker, which sets a hard limit for the max order
        size. However, we will be using a soft limit on that
        and use flow based max order sizing. The reason for
        using a max order size is that for shorter TFs such
        as `1h`, an abruptly large order will remain unfilled
        in live mode, while does get filled when backtesting,
        resulting in abnormally high returns of strategy.

        You can, also, limit the max_position_held for an asset
        in a similar way. This ensures that buy orders are
        limited such that the new order will not increase the
        equity of the asset beyond max_position_held, at the
        current prices.

        Finally, NOTE that order sizing is applicable on both BUY
        and SELL orders.

        For example, binance rejects any order worth less
        than 0.001BTC. This will be useful to reject such
        orders beforehand in backtest as well as live mode.
        """
        # # set individually for each order based on flow
        # self.broker.max_order_size = self.max_order_size
        # self.broker.max_position_held = self.max_order_size
        self.broker.min_order_size = self.min_order_size

    def transform_history(self, panel):
        """
        Calculate Flow for each point in time for each asset.
        Also, ask the child strategy for an assigned score, and/or
        weight for each asset in new portfolio.

        `score` and `weight` are mutually exclusive, and `weight`
        takes preference.

        Therefore, if `weight`s are provided, `score` will be ignored,
        and the strategy will try to buy any asset with a positive weight,
        if the current equity allocation is less than the specified
        weight in the portfolio. Weights can be non-normalized.

        If `score` is provided (and not weights), assets are weighted
        equally. If `assets` is None, strategy will try to buy all assets
        with positive score, otherwise top N assets will be bought. In
        this scenario, weights are equally distributed.

        This is useful for any vectorized operation on the data as a whole.
        Look into `transform_tick_data` for an additional approach to
        specify `score` and `weight` at each tick.
        """
        panel = super().transform_history(panel)
        panel.loc[:, :, 'flow'] = (
            panel[:, :, 'close'] * panel[:, :, 'volume']).rolling(
            self.flow_period).mean()
        if hasattr(self, 'calculate_scores'):
            panel.loc[:, :, 'score'] = self.calculate_scores(panel)
        if hasattr(self, 'calculate_weights'):
            panel.loc[:, :, 'weight'] = self.calculate_weights(panel)
        return panel

    def order_percent(self, symbol, amount, price=None,
                      max_cost=None, side=None, relative=None):
        """
        Place a MARKET/LIMIT order for a symbol for a given percent of
        available account capital.

        We calculate the flow of BTC in last few ticks for that asset.
        This combined with overall max_order_size places an upper bound
        on the order cost.

        If a price is specified, a LIMIT order is issued, otherwise MARKET.
        If `max_cost` is specified, order cost is, further, limited by that
        amount.
        """
        if symbol in self.data.index:
            max_flow = fast_xs(self.data, symbol)['flow']*self.flow_multiplier/100
        else:
            max_flow = 1
        if self.max_order_size:
            max_flow = min(max_flow, self.max_order_size)
        max_cost = min(max_cost, max_flow) if max_cost else max_flow
        args = ('MARKET', None, max_cost, side)
        if price:
            args = ('LIMIT', price, max_cost, side)
        if relative:
            self.order_relative_target_percent(symbol, amount, *args)
        else:
            self.order_target_percent(symbol, amount, *args)

    def before_summary_report(self):
        """ Syntactic sugar to add a newline before printing summary. """
        if self.context.debug:
            print()

    def increase_assets_if_required(self):
        """
        Increase number of assets being traded if we have surplus cash,
        gradually. This function is called at each tick before any
        advices are given out to ensure we are using max available
        assets on each tick.

        OPTIMIZE: (old legacy code) This can be optimized further.
        """
        if self.assets:
            limited = self.max_order_size or self.flow_multiplier
            if limited and self.assets < self.max_assets:
                equity = self.account.equity
                if (equity > self.assets**(self.assets**0.2)):
                    self.assets = min(self.max_assets, self.assets+1)

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
                    price = self.selling_price(
                        advice.symbol(self), {"price": advice.last_price})

            elif advice.is_sell and cost > 0 and abs(cost) < th:
                advice.side = "BUY"
                if price:
                    price = self.buying_price(
                        advice.symbol(self), {"price": advice.last_price})
        return (cost, quantity, price)

    def selling_price(self, _symbol, data):
        """
        Price at which an asset should be sold.
        MARKET order is used if returned price is None.
        """
        if self.markup_sell is not None:
            price = data['price'] if 'price' in data else data['close']
            return price*(1+self.markup_sell/100)

    def buying_price(self, _symbol, data):
        """
        Price at which an asset should be bought.
        MARKET order is used if returned price is None.
        """
        if self.markdn_buy is not None:
            price = data['price'] if 'price' in data else data['close']
            return price*(1-self.markdn_buy/100)

    def transform_tick_data(self, data):
        """
        Allow strategy to transform tick data at each tick,
        as well as provide score and/or weight for assets
        at each tick.

        By default, drop any asset without any data, and
        ignore tick data for assets without any volume.
        """
        if hasattr(self, 'calculate_scores_at_each_tick'):
            scores = self.calculate_scores_at_each_tick(data)
            if scores is not None:
                data.loc[:, 'score'] = scores

        if hasattr(self, 'calculate_weights_at_each_tick'):
            weights = self.calculate_weights_at_each_tick(data)
            if weights is not None:
                data.loc[:, 'weight'] = weights

        data = data.dropna(how='all')
        data = data[data['volume'] > 0]
        return data

    def sorted_data(self, data):
        """
        Sort tick data based on score, volume and closing price of assets.

        Child strategy can use this method to implement their own sorting.
        The top N assets for used for placing buy orders if `weight`
        column is missing from the assigned data.
        """
        if 'score' in data.columns:
            return data.sort_values(
                by=['score', 'volume', 'close'], ascending=[False, False, True])
        else:
            return data

    def rebalance_required(self, data, selected, rejected):
        """
        Check whether rebalancing is required at this tick or not.
        If `rebalance` is set as None, we will rebalance (trade) on
        each tick.

        Specifying `rebalance` as None means that we are not doing
        a time-based rebalancing. Assets are, instead, rebalanced
        based on Signals.

        Pending orders are cancelled at the start of each rebalancing
        tick.

        As a side effect, at the moment, you SHOULD cancel your open
        orders for an asset yourself if you are not doing time-based
        rebalancing.

        # FIXME: ensure Strategy is able to work with Signals as well,
        instead of just time-based rebalancing.
        """
        if not self.rebalance:
            return False
        time = self.tick.time
        timediff = pd.to_timedelta(self.datacenter.timeframe*self.rebalance)
        return (not self._last_rebalance or
                time >= self._last_rebalance + timediff)

    def selected_assets(self, data):
        """
        Allow child strategy to modify asset selection.
        By default:
            - Select all assets with positive weights,
              if asset weights are specified
            - Select all assets with positive score,
              if `assets` is not None
            - Select top N assets, if `assets` is specified.
        """
        # remove any assets with nan weight or score
        if "weight" in data.columns:
            data = data[np.isfinite(data['weight'])]
            return data[data['weight'] > 0]
        elif self.assets:
            data = data[np.isfinite(data['score'])]
            return data[data['score'] > 0].head(self.assets)
        else:
            data = data[np.isfinite(data['score'])]
            return data[data['score'] > 0]

    def rejected_assets(self, data):
        """
        Allow child strategy to modify asset selection.
        By default:
            - Reject all assets with zero,
              if asset weights are specified
            - Reject all assets with negative score,
              if `assets` is not None
            - Reject all but top N assets, if `assets` is specified.

        Be careful not to specify negative weights.
        """
        if "weight" in data.columns:
            return data[data['weight'] == 0]
        elif self.assets:
            return data.tail(len(data) - self.assets)
        else:
            return data[data['score'] < 0]

    def assign_weights_and_required_equity(self, data, selected):
        if "weight" not in data.columns:
            data['weight'] = 0
            data.loc[selected.index, 'weight'] = 1
        else:
            data.loc[~data.index.isin(selected.index), 'weight'] = 0
        data['weight'] /= data['weight'].sum()

        # if commission_asset falls to 50% of its required minimum,
        # we need to buy commission asset, therefore, adjust for it
        asset_equity = self.portfolio.equity_per_asset
        comm_percent = self.min_commission_asset_equity/100
        required_comm_equity = self.account.equity * comm_percent
        current_comm_equity = asset_equity[self.context.commission_asset]
        if current_comm_equity < required_comm_equity:
            data['weight'] *= 1 - comm_percent
        if self.reserved_cash:
            data["weight"] *= (1-self.reserved_cash/100)
        data['required_equity'] = data['weight'] * self.account.equity
        return data

    def calculate_weights_at_each_tick(self, data):
        """
        Default implementation for assigning weights
        to positive/top scoring assets.

        This allows to easily switch a strategy from
        equal weights to weights based on score
        by adding `weighted=True` strategy parameter.
        """
        if not self.weighted or 'weight' in data.columns:
            return
        weights = data['score'].copy()
        weights[weights < 0] = 0
        if self.assets:
            weights = weights.sort_values(ascending=False)
            weights.iloc[self.assets:] = 0
        return weights

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
        # ensure tyhat we have it at a fixed percent of equity all the time.
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if (symbol in rejected.index and
                    symbol in data.index and symbol not in selected.index):
                asset_data = fast_xs(data, symbol)
                if asset_equity <= asset_data['required_equity']/100:
                    continue
                n, price = 0, self.selling_price(symbol, asset_data)
                if asset == self.context.commission_asset:
                    n = self.min_commission_asset_equity
                    self.order_percent(symbol, n, price, side='SELL')
                else:
                    self.order_percent(symbol, n, price)

        # next, sell assets that have higher equity first
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if symbol not in selected.index:
                continue
            asset_data = fast_xs(data, symbol)
            if asset_equity > asset_data['required_equity']:
                price = self.selling_price(symbol, asset_data)
                n = asset_data['weight'] * 100
                if asset == self.context.commission_asset:
                    n = max(self.min_commission_asset_equity, n)
                self.order_percent(symbol, n, price, side='SELL')

        # finally, buy assets that have lower equity now
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if symbol not in selected.index:
                continue
            asset_data = fast_xs(data, symbol)
            if asset_equity < asset_data['required_equity']:
                price = self.buying_price(symbol, asset_data)
                n = asset_data['weight'] * 100
                if (asset == self.context.commission_asset and
                        asset_equity < min_comm):
                    self.order_percent(symbol,
                                       self.min_commission_asset_equity,
                                       side='BUY')
                    n -= self.min_commission_asset_equity
                self.order_percent(symbol, n, price, side='BUY')

        if hasattr(self, "after_strategy_advice_at_tick"):
            self.after_strategy_advice_at_tick()

        self._last_rebalance = self.tick.time
        self._save_state()

    def after_order_created_done(self, event):
        """
        Once an order has been created, if the order has an ID (usually,
        in live mode), notify all connected services about the order.
        """
        order = event.item
        if order.id and self.debug:
            self.notify(
                "  Created %4s %s order with ID %s for %0.8f %s at %.8f %s" % (
                   order.side, order.order_type, order.id, abs(order.quantity),
                   order.asset, order.fill_price, order.base),
                formatted=True, now=event.item.time, publish=False)

    def after_order_rejected(self, event):
        """
        Once an order has been rejected,
        notify all connected services about the order.
        This notifies in backtest mode too.
        """
        order = event.item
        if self.debug:
            self.notify(
                " Rejected %4s %s order for %0.8f %s at %.8f %s (Reason: %s)"
                % (order.side, order.order_type, abs(order.quantity),
                   order.asset, order.fill_price, order.base, event.reason),
                formatted=True, now=event.item.time, publish=False)

