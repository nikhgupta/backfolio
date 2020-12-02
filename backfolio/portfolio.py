import math
import numpy as np
import pandas as pd
from os.path import join, dirname

from .core.object import OrderGroup
from .core.event import OrderRequestedEvent
from .core.utils import as_df, make_path, fast_xs


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
        self.last_tick = None
        self.delisted_assets = []

        self.session_fields = [
            'timeline', 'asset_equity', 'advice_history', 'orders',
            'filled_orders', 'rejected_orders', 'unfilled_orders',
            'open_orders', 'positions'
        ]
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
            raise ValueError(
                "Not sure how to calculate equity here. Data missing?")
        if hasattr(self.strategy, 'transform_account_equity'):
            equity = self.strategy.transform_account_equity(equity)
        return equity

    @property
    def equity_per_asset(self, n=1):
        if self._converted_to_pandas:
            equity = self.asset_equity.iloc[-n].to_dict()
        elif len(self.asset_equity):
            equity = self.asset_equity[-n].copy()
        else:
            equity = {}
        if 'time' in equity:
            equity.pop("time")
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
        self.last_tick = tick_event.item
        res = self.last_positions_and_equities_at_tick_open()
        self.asset_equity.append({
            **res['asset_equity'],
            **{
                "time": res['time']
            }
        })
        self.positions.append({**res['positions'], **{"time": res['time']}})
        self.timeline.append({**res['summary'], **{"time": res['time']}})

    def last_positions_and_equities_at_tick_open(self, field='open'):
        base = self.context.base_currency
        resp = {
            "positions": self.account.total,
            "asset_equity": {},
            "time": None,
            "summary": {
                "equity": 0,
                "cash": self.account.cash,
                "total_cash": self.account.total[base]
            }
        }

        if self.last_tick is None:
            return resp

        data = self.last_tick.history
        for asset, quantity in self.account.total.items():
            symbol = self.datacenter.assets_to_symbol(asset)
            if asset == self.context.base_currency:
                resp['asset_equity'][asset] = quantity
            elif symbol in data.index:
                price = fast_xs(data.fillna(0), symbol)[field]
                resp['asset_equity'][asset] = quantity * price
            elif self.is_holding_delisted_asset(asset, symbol) and len(
                    self.asset_equity):
                if self.context.live_trading():
                    last_equity = 0
                else:
                    last_equity = self.equity_per_asset[asset]
                    self.context.delist_asset(asset, last_equity)
                resp['asset_equity'][asset] = last_equity
            elif len(self.asset_equity) and asset in self.delisted_assets:
                resp['asset_equity'][asset] = 0
            elif len(self.asset_equity) and asset in self.equity_per_asset:
                resp['asset_equity'][asset] = self.equity_per_asset[asset]
            elif len(self.asset_equity):
                self.context.notify(
                    "!!!! NOT SURE WHAT TO DO - ASSET's VALUE COULD NOT BE DETERMINED (in portfolio.py)"
                )
                if self.context.backtesting():
                    from IPython import embed
                    embed()
                    raise "asset's value could not be determined."
                else:
                    resp['asset_equity'][asset] = 0
            else:
                resp['asset_equity'][asset] = 0

        resp['time'] = self.last_tick.time
        resp['summary']['equity'] = math.fsum(
            [v for k, v in resp['asset_equity'].items()])
        return resp

    def last_positions_and_equities_at_tick_close(self):
        return self.last_positions_and_equities_at_tick_open(field='close')

    def is_holding_delisted_asset(self, asset, symbol=None):
        if (asset not in self.account.total or self.account.total[asset] < 1e-8
                or asset == self.context.base_currency):
            return False
        if not symbol:
            symbol = self.datacenter.assets_to_symbol(asset)
        if symbol not in self.datacenter.history.axes[0]:
            return True
        history = self.datacenter.history[symbol]
        history = history[self.context.current_time.strftime("%Y-%m-%d %H:%M"
                                                             ):]
        without_history = history.dropna(how='all').empty
        if without_history:
            self.delisted_assets.append(asset)
        return without_history

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
        OrderGroup.add_order_to_groups(order, self.order_groups)

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

    def _load_order_groups(self):
        self.order_groups = OrderGroup.load_all(self.closed_orders)

    @property
    def closed_orders(self):
        if self._converted_to_pandas:
            return self.orders[self.orders.status == 'closed']
        else:
            return [o for o in self.orders if o.is_closed]

    def trading_session_complete(self):
        props = [
            'advice_history', 'orders', 'open_orders', 'filled_orders',
            'rejected_orders', 'unfilled_orders', 'order_groups', 'positions',
            'asset_equity', 'timeline'
        ]
        for prop in props:
            setattr(self, prop, as_df(getattr(self, prop)))

        for prop in ['positions', 'asset_equity', 'timeline']:
            prop = getattr(self, prop)
            if not prop.empty:
                prop.set_index('time', inplace=True)

        if 'commission_paid' in self.timeline.columns:
            self.timeline['commission_paid'].fillna(0, inplace=True)
            self.timeline['commission_paid'] = \
                self.timeline['commission_paid'].cumsum()
        self.timeline = self.timeline.fillna(method='pad')
        self.timeline['returns'] = (
            self.timeline.equity / self.timeline.equity.shift() - 1).fillna(0)
        self.timeline['cum_returns'] = (1 + self.timeline['returns']).cumprod()

        # same as above for daily time period
        # this is the equity/commission paid till the start # of current day
        self.daily = self.timeline.groupby(pd.Grouper(freq='D')).last()
        self.daily['returns'] = (
            self.daily.equity / self.daily.equity.shift(1) - 1).fillna(0)
        self.daily['cum_returns'] = (1 + self.daily['returns']).cumprod()

        self._converted_to_pandas = True

    def save_as_benchmark(self, *args):
        data_dir = join(self.context.root_dir, "benchmarks")
        cache_name = "/".join(args)
        cache = join(data_dir, "%s.csv" % cache_name)
        make_path(dirname(cache))
        self.timeline.to_csv(cache, index=True)
