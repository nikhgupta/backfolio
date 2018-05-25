"""
Core class for implementing a strategy used by CryptoFolio
"""

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
        self.session_fields = ['session_data']

    def transform_history(self, panel):
        return panel

    def reset(self, context):
        """ Routine to run when trading session is resetted. """
        self.context = context

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
                      exchange=None):
        name = self.__class__.__name__
        if symbol not in self.tick.history.index:
            # self.context.log("Ignoring order for: %s as it was not found in \
            #                   current tick" % symbol)
            return
        last = fast_xs(self.tick.history, symbol)['close']
        advice = Advice(name, symbol, exchange, last, quantity, quantity_type,
                        order_type, limit_price, max_cost, self.tick.time)
        self.events.put(StrategyAdviceEvent(advice))
        return advice

    def order_target_percent(self, symbol, quantity, order_type='MARKET',
                             limit_price=None, max_cost=0):
        return self._order_target(symbol, quantity, 'PERCENT',
                                  order_type, limit_price, max_cost)

    def order_target_amount(self, symbol, quantity, order_type='MARKET',
                            limit_price=None, max_cost=0):
        return self._order_target(symbol, quantity, 'SHARE',
                                  order_type, limit_price, max_cost)


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
    def __init__(self, assets=3, max_assets=7, max_order_size=0.12,
                 markup_sell=0.01, markdn_buy=0.01,
                 rebalance=1, capital_used=98,
                 flow_period=12, flow_multiplier=0.025):
        super().__init__()
        self.assets = assets
        self.rebalance = rebalance
        self.max_assets = max_assets
        self.max_order_size = max_order_size
        self.markdn_buy = markdn_buy
        self.markup_sell = markup_sell
        self.flow_period = flow_period
        self.flow_multiplier = flow_multiplier
        self.capital_used = capital_used
        self.session_fields = ['state']
        self._last_rebalance = None
        self._state = []

    def reset(self, context):
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
        if 'last_rebalance' in self.current_state:
            self._last_rebalance = self.current_state["last_rebalance"]
        return self.current_state

    def _save_state(self, **data):
        self._state.append({
            **data, **{
                "last_rebalance": self._last_rebalance,
                "time": self.tick.time}})

    def before_trading_start(self):
        self.broker.min_order_size = 0.001

    def transform_history(self, panel):
        close = panel[:, :, 'close']
        volume = panel[:, :, 'volume']
        panel.loc[:, :, 'flow'] = (close*volume).rolling(
            self.flow_period).mean()
        panel.loc[:, :, 'score'] = self.calculate_score(panel)
        return panel

    def order_percent(self, symbol, amount, price=None, max_cost=None):
        max_flow = fast_xs(self.data, symbol)['flow']*self.flow_multiplier
        if self.max_order_size:
            max_flow = max(max_flow, self.max_order_size)
        max_cost = min(max_cost, max_flow) if max_cost else max_flow
        args = ('MARKET', None, max_cost)
        if price:
            args = ('LIMIT', price, max_cost)
        self.order_target_percent(symbol, amount, *args)

    def selling_price(self, symbol, data):
        price = fast_xs(data, symbol)['close']
        markup = price * (1+self.markup_sell)
        return markup

    def buying_price(self, symbol, data):
        price = fast_xs(data, symbol)['close']
        markdn = price * (1-self.markdn_buy)
        return markdn

    def before_summary_report(self):
        print()

    def percentage_share(self, selected=[]):
        # decide the percentage distribution for each asset
        # if we are limiting the max order size, increase the
        # number of assets held gradually and assign based on
        # new number of assets. Also, ensure that we keep same
        # percentage of cash/base-asset as reserve.
        n = 0.01
        if self.assets:
            n = self.capital_used/(1+self.assets)
            if self.max_order_size and self.assets < self.max_assets:
                cash = self.account.cash
                if (cash > self.max_order_size*self.assets*1.25):
                    self.assets = min(self.max_assets, self.assets+1)
                    n = self.capital_used/(1+self.assets)
        else:
            n = self.capital_used/(1+len(selected))
        return n

    def transform_order_calculation(self, _request, cost, quantity, price):
        if cost > 0 and cost >= self.account.cash * 0.95:
            cost = self.account.cash * 0.95
            quantity = cost/price
        return (cost, quantity, price)

    def rebalance_required(self):
        time = self.tick.time
        timediff = pd.to_timedelta(self.datacenter.timeframe*self.rebalance)
        return (not self._last_rebalance or
                time >= self._last_rebalance + timediff)

    def sorted_data(self, data):
        data = data.sort_values(
            by=['score', 'volume', 'close'], ascending=[0, 0, 1])
        return data[data['volume'] > 0]

    def selected_assets(self, data):
        if self.assets:
            return data.head(self.assets)
        else:
            return data[data['score'] > 0]

    def rejected_assets(self, data):
        if self.assets:
            return data.tail(len(data) - self.assets)
        else:
            return data[data['score'] < 0]

    def after_order_created_done(self, event):
        order = event.item
        side = 'buy' if order.quantity > 0 else 'sell'
        if order.id:
            self.notify(
                "  Created %4s %s order with ID %s for %0.8f %s at %.8f %s" % (
                   side, order.order_type, order.id, abs(order.quantity),
                   order.asset, order.fill_price, order.base),
                formatted=True, now=event.item.time)

    def after_order_rejected(self, event):
        order = event.item
        side = 'buy' if order.quantity > 0 else 'sell'
        self.notify(
            " Rejected %4s %s order for %0.8f %s at %.8f %s (Reason: %s)" % (
              side, order.order_type, abs(order.quantity), order.asset,
              order.fill_price, order.base, event.reason),
            formatted=True, now=event.item.time)

    def in_rejected(self, symbol, selected, rejected):
        return symbol in rejected.index and symbol not in selected.index

    def in_selected(self, symbol, selected, rejected):
        return symbol in selected.index and symbol not in rejected.index

    def advice_investments_at_tick(self, _tick_event):
        # if we have no data for this tick, do nothing.
        if self.data.empty:
            return

        # if we need to wait for rebalancing, do nothing.
        if self.rebalance_required():
            self.broker.cancel_pending_orders()
        else:
            return

        # sort assets based on score assigned to them, and
        # ignore assets without any volume in last tick (padded data)
        data = self.sorted_data(self.data)

        # select the top performing assets based on their scores
        # if nothing has been selected, stop processing further.
        selected = self.selected_assets(data)
        rejected = self.rejected_assets(data)

        # get the equity for each asset and target equity amount
        n = self.percentage_share(selected)
        rebalanced = self.account.equity*n/100
        equity = self.portfolio.equity_per_asset

        # first sell everything that is not in selected coins,
        # provided they are worth atleast 1% above the threshold.
        # If the asset is the one in which commission is being deducted,
        # ensure that we have it at a fixed percent of equity all the time.
        for asset, asset_equity in equity.items():
            symbol = self.datacenter.assets_to_symbol(asset)
            if (self.in_rejected(symbol, selected, rejected) and
                    asset_equity > rebalanced/100):
                if asset == self.context.commission_asset:
                    price = fast_xs(data, symbol)['close']
                    self.order_percent(symbol, 3)
                else:
                    price = self.selling_price(symbol, data)
                    self.order_percent(symbol, 0, price)

        # next, sell assets that have higher equity first
        for asset, asset_equity in equity.items():
            symbol = self.datacenter.assets_to_symbol(asset)
            if (asset_equity > rebalanced and
                    self.in_selected(symbol, selected, rejected)):
                price = self.selling_price(symbol, data)
                self.order_percent(symbol, n, price)

        # finally, buy assets that have lower equity now
        for asset, asset_equity in equity.items():
            symbol = self.datacenter.assets_to_symbol(asset)
            if (asset_equity < rebalanced and
                    self.in_selected(symbol, selected, rejected)):
                price = self.buying_price(symbol, data)
                self.order_percent(symbol, n, price)

        self._last_rebalance = self.tick.time
        self._save_state()
