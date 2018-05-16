"""
Core class for implementing a strategy used by CryptoFolio
"""

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

    def log(self, *args, **kwargs):
        self.log(*args, **kwargs)

    def notify(self, *args, **kwargs):
        return self.context.notify(*args, **kwargs)

    @abstractmethod
    def advice_investments_at_tick(self, _tick_event):
        """ Advice investment for assets """
        raise NotImplementedError("Strategy must implement \
                                  `advice_investments()`")

    def _order_target(self, tick, symbol, quantity, quantity_type,
                      order_type='MARKET', limit_price=None, exchange=None):
        name = self.__class__.__name__
        if symbol not in tick.history.index:
            self.context.log("Ignoring order for: %s as it was not found in \
                             current tick" % symbol)
            return
        last = fast_xs(tick.history, symbol)['close']
        advice = Advice(tick, name, symbol, exchange, last,
                        quantity, quantity_type, order_type, limit_price)
        self.events.put(StrategyAdviceEvent(advice))
        return advice

    def order_target_percent(self, tick, symbol, quantity,
                             order_type='MARKET', limit_price=None):
        return self._order_target(tick, symbol, quantity, 'PERCENT',
                                  order_type, limit_price)

    def order_target_amount(self, tick, symbol, quantity,
                            order_type='MARKET', limit_price=None):
        return self._order_target(tick, symbol, quantity, 'SHARE',
                                  order_type, limit_price)


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
        tick = tick_event.item
        data = tick.history.dropna()
        new_symbols = [s for s in data.index if s not in self.added]
        self.pending.append(new_symbols)
        self.data_store.append(data)
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
            self.order_target_percent(tick, symbol, n, 'MARKET')


# TODO: buy commission asset if we get low on it
class EwUCRPStrategy(BaseStrategy):
    """
    A simple strategy which invests our capital equally amongst all assets
    available on an exchange, and rebalances it daily.

    In principle, a EwUCRP (Equally-weighted Uniformly Constant Rebalanced
    Portfolio) is hard to beat, which makes this strategy an excellent
    benchmark for our use cases.

    Assets that have been delisted are removed from our holdings.

    This portfolio strategy is different than the Buy and Hold strategy above,
    in that rebalance is done daily here as opposted to when a new asset is
    introduced on the exchange.
    """
    def __init__(self):
        super().__init__()
        self.bought = []
        self.session_fields += ['bought']

    def advice_investments_at_tick(self, tick_event):
        tick = tick_event.item
        data = tick.history.dropna()
        n = 100./(1+len(data))

        rebalanced = self.account.equity*n/100
        equity = self.portfolio.equity_per_asset

        # sell assets that have higher equity first
        # once we have cash available, buy assets that have lower equity now
        rebalance = [k for k, v in equity.items() if v > rebalanced]
        rebalance += [k for k, v in equity.items() if v < rebalanced]
        for asset in rebalance:
            symbol = self.datacenter.assets_to_symbol(asset)
            self.order_target_percent(tick, symbol, n, 'MARKET')
            if symbol not in self.bought:
                self.bought.append(symbol)
