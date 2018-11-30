"""
Core class for implementing a strategy used by CryptoFolio
"""

import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from .core.object import Advice
from .core.event import StrategyAdviceEvent
from .core.utils import fast_xs


class BaseStrategy(object):
    """
    BaseStrategy is an abstract base class providing an interface for all
    subsequent (inherited) strategy handling objects.

    The goal of a (derived) Strategy object is to generate StrategyAdviceEvent
    objects for each asset based on the inputs of Bars (OHLCV) generated
    in a TickUpdateEvent.

    This is designed to work both with historic and live data as the Strategy
    object is agnostic to where the data came from, since it obtains the bar
    tuples from a queue object.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.session_data = []
        self._selected_symbols = []
        self.session_fields = ['session_data']

    @property
    def symbols(self):
        return self._selected_symbols

    @symbols.setter
    def symbols(self, arr=[]):
        self._selected_symbols = arr

    def transform_history(self, panel):
        if self.symbols:
            panel = panel[self.symbols]
        return panel

    def reset(self, context):
        """ Routine to run when trading session is resetted. """
        self.context = context
        return self

    @property
    def account(self):
        return self.context.account

    @property
    def broker(self):
        return self.context.broker

    @property
    def portfolio(self):
        return self.context.portfolio

    @property
    def reporters(self):
        return self.context.reporters

    @property
    def benchmarks(self):
        return self.context.benchmarks

    @property
    def datacenter(self):
        return self.context.datacenter

    @property
    def events(self):
        return self.context.events

    @property
    def current_time(self):
        return self.context.current_time

    def log(self, *args, **kwargs):
        self.context.log(*args, **kwargs)

    def notify(self, *args, **kwargs):
        return self.context.notify(*args, **kwargs)

    @abstractmethod
    def advice_investments_at_tick(self, _tick_event):
        """ Advice investment for assets """
        raise NotImplementedError("Strategy must implement \
                                  `advice_investments_at_tick()`")

    def _order_target(self, symbol, quantity, quantity_type,
                      order_type='MARKET', limit_price=None, max_cost=0,
                      side=None, exchange=None):
        name = self.__class__.__name__
        if symbol not in self.tick.history.index:
            # self.context.log("Ignoring order for: %s as it was not found in \
            #                   current tick" % symbol)
            return

        position = 0
        asset, base = self.datacenter.symbol_to_assets(symbol)
        if asset in self.account.total:
            position = self.account.total[asset]
        if quantity == 0 and position == 0:
            return  # empty order advice

        last = fast_xs(self.tick.history, symbol)['close']
        advice = Advice(
            name, asset, base, exchange, last, quantity, quantity_type,
            order_type, limit_price, max_cost, side, position, self.tick.time)
        self.events.put(StrategyAdviceEvent(advice))
        return advice

    def order_relative_target_percent(self, symbol, quantity, order_type='MARKET',
                             limit_price=None, max_cost=0, side=None):
        return self._order_target(symbol, quantity, 'REL_PERCENT',
                                  order_type, limit_price, max_cost, side)


    def order_target_percent(self, symbol, quantity, order_type='MARKET',
                             limit_price=None, max_cost=0, side=None):
        return self._order_target(symbol, quantity, 'PERCENT',
                                  order_type, limit_price, max_cost, side)

    def order_target_amount(self, symbol, quantity, order_type='MARKET',
                            limit_price=None, max_cost=0, side=None):
        return self._order_target(symbol, quantity, 'SHARE',
                                  order_type, limit_price, max_cost, side)


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


class RebalanceOnScoreStrategy(BaseStrategy):
    """
    Strategy that buy top scoring assets, and sells other assets
    every given time interval.
    """
    def __init__(self, rebalance=1, assets=3, max_assets=13,
                 max_order_size=0, min_order_size=0.001,
                 flow_period=1, flow_multiplier=2.5,
                 markup_sell=1, markdn_buy=1, reserved_cash=0,
                 min_commission_asset_equity=3, weighted=False, debug=True):
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
        super().__init__()
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
        max_flow = fast_xs(self.data, symbol)['flow']*self.flow_multiplier/100
        if self.max_order_size:
            max_flow = max(max_flow, self.max_order_size)
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
        if cost > 0 and cost >= self.account.cash * 0.95:
            cost = self.account.cash * 0.95

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
                by=['score', 'volume', 'close'], ascending=[0, 0, 1])
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
        data['weight'] /= data['weight'].sum()

        # if commission_asset falls to 50% of its required minimum,
        # we need to buy commission asset, therefore, adjust for it
        asset_equity = self.portfolio.equity_per_asset
        comm_percent = self.min_commission_asset_equity/100
        required_comm_equity = self.account.equity * comm_percent
        current_comm_equity = asset_equity[self.context.commission_asset]
        if current_comm_equity < required_comm_equity:
            data['weight'] *= 1 - comm_percent

        data['required_equity'] = data['weight'] * self.account.equity
        if self.reserved_cash:
            data["required_equity"] *= (1-self.reserved_cash/100)
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
            weights = weights.sort_values(ascending=0)
            weights.iloc[self.assets:] = 0
        return weights

    def advice_investments_at_tick(self, _tick_event):
        # if we have no data for this tick, do nothing.
        if self.data.empty:
            return

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

        min_comm = self.min_commission_asset_equity/100*self.account.equity

        # if we need to wait for rebalancing, do nothing.
        if self.rebalance_required(data, selected, rejected):
            self.broker.cancel_pending_orders()
        elif self.rebalance:
            return

        # first sell everything that is not in selected coins,
        # provided they are worth atleast 1% above the threshold.
        # If the asset is the one in which commission is being deducted,
        # ensure tyhat we have it at a fixed percent of equity all the time.
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            base = self.context.base_currency
            if (asset in self.account.total and
                self.account.total[asset] > 1e-8 and
                symbol not in data.index and asset != base):
                self.account.total[base] += asset_equity
                asset_equity = self.account.total[asset] = 0
                self.account.locked[asset] = self.account.free[asset] = 0
                continue
                # from IPython import embed; embed()

            if (symbol in rejected.index and
                    symbol in data.index and symbol not in selected.index):
                asset_data = fast_xs(data, symbol)
                if asset_equity <= asset_data['required_equity']/100:
                    continue
                n, price = 0, self.selling_price(symbol, asset_data)
                if asset == self.context.commission_asset:
                    n = self.min_commission_asset_equity
                    price = asset_data['close']
                # order can be placed on any side - we dont care due to amount
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
                formatted=True, now=event.item.time)

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
                formatted=True, now=event.item.time)


class RebalanceOnScoreSplitOrders(RebalanceOnScoreStrategy):
    def selling_prices(self, _symbol, data):
        if self.markup_sell is not None:
            price = data['price'] if 'price' in data else data['close']
            return [price*(1+x/100) for x in self.markup_sell]
        else:
            return [price]

    def buying_prices(self, _symbol, data):
        if self.markdn_buy is not None:
            price = data['price'] if 'price' in data else data['close']
            return [price*(1-x/100) for x in self.markdn_buy]
        else:
            return [price]

    def advice_investments_at_tick(self, _tick_event):
        # if we have no data for this tick, do nothing.
        if self.data.empty:
            return

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

        min_comm = self.min_commission_asset_equity/100*self.account.equity

        # if we need to wait for rebalancing, do nothing.
        if self.rebalance_required(data, selected, rejected):
            self.broker.cancel_pending_orders()
        elif self.rebalance:
            return

        # first sell everything that is not in selected coins,
        # provided they are worth atleast 1% above the threshold.
        # If the asset is the one in which commission is being deducted,
        # ensure that we have it at a fixed percent of equity all the time.
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            base = self.context.base_currency
            if (asset in self.account.total and
                self.account.total[asset] > 1e-8 and
                symbol not in data.index and asset != base and
                self.context.backtesting()):
                self.account.total[base] += asset_equity
                asset_equity = self.account.total[asset] = 0
                self.account.locked[asset] = self.account.free[asset] = 0
                if self.context.debug:
                    print("========= IGNORING COIN: %s ============" % asset)
                continue

            if (symbol in rejected.index and
                    symbol in data.index and symbol not in selected.index):
                asset_data = fast_xs(data, symbol)
                #if symbol == "SKY/BTC": embed()
                if asset_equity > asset_data['required_equity']/100:
                    n, prices = 0, self.selling_prices(symbol, asset_data)
                    N = asset_equity/self.account.equity*100
                    if asset == self.context.commission_asset:
                        n = self.min_commission_asset_equity
                        prices = [asset_data['close']]

                    for price in prices:
                        x = (n-N)/len(prices)
                        #print("MSELL: ", asset, x, n, N, price)
                        self.order_percent(symbol, x, price, relative=True)

        # next, sell assets that have higher equity first
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if symbol not in selected.index:
                continue
            asset_data = fast_xs(data, symbol)
            if asset_equity > asset_data['required_equity']:
                prices = self.selling_prices(symbol, asset_data)
                n = asset_data['weight'] * 100
                N = asset_equity/self.account.equity*100
                if asset == self.context.commission_asset:
                    n = max(self.min_commission_asset_equity, n)
                for price in prices:
                    x = (n-N)/len(prices)
                    #print("SELL: ", asset, x, n, N, price)
                    self.order_percent(symbol, x, price, side='SELL', relative=True)

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
                if (asset == self.context.commission_asset and
                        asset_equity < min_comm):
                    self.order_percent(symbol, self.min_commission_asset_equity, side='BUY')
                    n -= self.min_commission_asset_equity
                for price in prices:
                    x = (n-N)/len(prices)
                    #print("BUY: ", asset, x, n, N, price)
                    self.order_percent(symbol, x, price, side='BUY', relative=True)

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
        if cost > 0 and cost >= self.account.cash * 0.95:
            cost = self.account.cash * 0.95

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
                        advice.symbol(self), {"price": advice.last_price})[0]

            elif advice.is_sell and cost > 0 and abs(cost) < th:
                advice.side = "BUY"
                if price:
                    price = self.buying_prices(
                        advice.symbol(self), {"price": advice.last_price})[0]
        return (cost, quantity, price)
