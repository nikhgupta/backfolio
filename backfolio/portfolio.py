import numpy as np
import pandas as pd
from os.path import join

from .core.event import OrderRequestedEvent
from .core.utils import as_df, items_as_df, make_path, fast_xs


class BasePortfolio(object):
    """
    BasePortfolio is a base class providing an interface for all subsequent
    (inherited) portfolios.

    The goal of a (derived) BasePortfolio object is to record any advices given
    out by the strategy, place new orders to the broker, and keep a track of
    various statistics of our portfolio/equity curve.

    Whether the order that was placed is accepted by the broker is not the
    concern of BasePortfolio object.
    """

    def __init__(self):
        self.name = 'portfolio'

    def reset(self, context):
        self.context = context
        self.positions = []

        self.daily = None
        self.timeline = []
        self.asset_equity = []
        self.advice_history = []
        self.orders = []
        self.open_orders = []
        self.filled_orders = []
        self.rejected_orders = []
        self.unfilled_orders = []
        self._converted_to_pandas = False

        self.session_fields = ['timeline', 'asset_equity', 'advice_history',
                               'orders', 'filled_orders', 'rejected_orders',
                               'unfilled_orders', 'open_orders']
        return self

    @property
    def account(self):
        return self.context.account

    @property
    def broker(self):
        return self.context.broker

    @property
    def strategy(self):
        return self.context.strategy

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

    @property
    def cash(self):
        """
        The amount of cash (base currency) we held at any moment.
        """
        return self.account.cash

    @property
    def equity(self):
        if self._converted_to_pandas:
            equity = self.timeline.equity.iloc[-1]
        elif len(self.timeline):
            equity = self.timeline[-1]['equity']
        else:
            equity = np.nan
        if hasattr(self.strategy, 'transform_account_equity'):
            equity = self.strategy.transform_account_equity(equity)
        return equity

    @property
    def equity_per_asset(self):
        equity = self.asset_equity[-1].copy()
        equity.pop("time")
        equity.pop("BTC")
        return equity

    # TODO: When an asset does not trade wrt base currency, but has tick data
    # for some other currency that trades in base currency, we should calculate
    # equity by first converting to other asset and then to base currency
    def update_portfolio_value_at_tick(self, tick_event):
        """
        For each asset held in our Account, calculate its worth with
        `self.currency` as base.

        This assumes that tick data provides exchange rates for the assets vs
        the base currency.
        """
        equity = {}
        time = tick_event.item.time
        data = tick_event.item.history

        for asset, quantity in self.account.total.items():
            symbol = self.datacenter.assets_to_symbol(asset)
            if asset == self.context.base_currency:
                equity[asset] = quantity
            elif symbol in data.index:
                price = fast_xs(data.fillna(0), symbol)['open']
                equity[asset] = quantity * price
            else:
                equity[asset] = 0

        total_equity = sum([v for k, v in equity.items()])
        self.asset_equity.append({**equity, **{"time": time}})
        self.positions.append({**self.account.total, **{"time": time}})
        self.timeline.append({"time": time, "equity": total_equity,
                              "cash": self.account.cash})

    def record_advice_from_strategy(self, advice_event):
        self.advice_history.append(advice_event.item)

    def place_order_after_advice(self, advice_event):
        advice = advice_event.item
        self.events.put(OrderRequestedEvent(advice))
        return advice

    def record_created_order(self, order_event, created_order=None):
        item = created_order if created_order else order_event.item
        self.orders.append(item)
        self.open_orders.append(item)

    def record_filled_order(self, order_filled_event):
        # ensure that the status of order is marked as closed in portfolio
        order = order_filled_event.item
        self.filled_orders.append(order)
        if order in self.open_orders:
            self.open_orders.remove(order)

    def record_rejected_order(self, rejected_order_event):
        order = rejected_order_event.item
        self.rejected_orders.append(order)
        if order in self.open_orders:
            self.open_orders.remove(order)

    def record_unfilled_order(self, unfilled_order_event):
        order = unfilled_order_event.item
        self.unfilled_orders.append(order)
        if order in self.open_orders:
            self.open_orders.remove(order)

    def update_commission_paid(self, event):
        if not len(self.timeline):
            raise ValueError("Commission paid before the first tick? WTF!")

        comm_paid = 0
        if 'commission_paid' in self.timeline[-1]:
            comm_paid = self.timeline[-1]['commission_paid']
        comm_paid += event.item.commission
        self.timeline[-1]['commission_paid'] = comm_paid
        self.timeline[-1]['commission_asset'] = event.item.commission_asset

    def trading_session_tick_complete(self):
        pass

    @property
    def closed_orders(self):
        if self._converted_to_pandas:
            return self.orders[self.orders.status == 'closed']
        else:
            return [o for o in self.orders if o.is_closed]

    def trading_session_complete(self):
        # record advices
        self.advice_history = items_as_df(self.advice_history, 'id')
        # record orders
        self.orders = items_as_df(self.orders, 'local_id')
        self.open_orders = items_as_df(self.open_orders, 'local_id')
        self.filled_orders = items_as_df(self.filled_orders, 'local_id')
        self.rejected_orders = items_as_df(self.rejected_orders, 'local_id')
        self.unfilled_orders = items_as_df(self.unfilled_orders, 'local_id')
        # record asset quantity/position and equity over time
        self.positions = as_df(self.positions, 'time', dupes='index')
        self.asset_equity = as_df(self.asset_equity, 'time', dupes='index')

        # returns, cumulative returns, commission paid, cash and total equity
        self.timeline = as_df(self.timeline, 'time', dupes='index')
        if 'commission_paid' in self.timeline.columns:
            self.timeline['commission_paid'].fillna(0, inplace=True)
            self.timeline['commission_paid'] = \
                self.timeline['commission_paid'].cumsum()
        self.timeline = self.timeline.fillna(method='pad')
        self.timeline['returns'] = (
            self.timeline.equity/self.timeline.equity.shift() - 1).fillna(0)
        self.timeline['cum_returns'] = (
            1+self.timeline['returns']).cumprod()

        # same as above for daily time period
        # this is the equity/commission paid till the start # of current day
        self.daily = self.timeline.groupby(pd.Grouper(freq='D')).last()
        self.daily['returns'] = (
            self.daily.equity / self.daily.equity.shift(1) - 1).fillna(0)
        self.daily['cum_returns'] = (1+self.daily['returns']).cumprod()

        self._converted_to_pandas = True

    def save_as_benchmark(self, cache_name):
        data_dir = join(self.context.root_dir, "benchmarks")
        cache = join(data_dir, "%s.csv" % cache_name)
        make_path(data_dir)
        self.timeline.to_csv(cache, index=True)
