import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from ..core.object import Advice
from ..core.event import StrategyAdviceEvent
from ..core.utils import fast_xs


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

    def before_strategy_advice_at_tick(self):
        return False

    def after_strategy_advice_at_tick(self):
        return False

    def before_trading_start(self):
        pass

    def transform_history(self, panel):
        """
        This is useful for any vectorized operation on the data as a whole.
        """
        return panel

    def transform_tick_data(self, data):
        return data

    def selected_assets(self, data):
        return data

    def rejected_assets(self, data):
        return data

    def before_trading_start(self):
        pass

    def buying_prices(self, _symbol, data):
        pass

    def selling_prices(self, _symbol, data):
        pass

    def set_required_equity_for_each_asset(self):
        return 1/len(self.data)

    def before_summary_report(self):
        """ Syntactic sugar to add a newline before printing summary. """
        if self.context.debug:
            print()

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
