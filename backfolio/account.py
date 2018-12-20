import ccxt
import datetime
from abc import ABCMeta, abstractmethod


class AbstractAccount(object):
    """
    AbstractAccount is an abstract class providing an interface for all
    subsequent (inherited) accounts.

    The goal of a (derived) Account object is to provide information about
    the current balance for all assets that are being traded.

    In a live trading environment, this interface will connect with the online
    exchange and get balances for various assets.
    """

    __metaclass__ = ABCMeta

    def __init__(self, initial_capital=None):
        self.free = {}
        self.locked = {}
        self.total = {}
        self.last_update_at = None
        self.initial_capital = initial_capital
        self._extra_capital = None
        self.session_fields = []
        self.lender = {}

    def __repr__(self):
        bal = dict([key, val] for key, val in self.total.items() if val > 0)
        return "%s(%s)" % (self.__class__.__name__, bal)

    def reset(self, context):
        """ Routine to run when trading session is resetted. """
        self.context = context
        self.free = {}
        self.locked = {}
        self.total = {}
        self.last_update_at = None
        self.get_balance(refresh=True)
        self.lender = {k: 0 for k, v in self.total.items()}
        return self

    def _adjust_for_extra_capital(self):
        if self._extra_capital is not None:
            return

        if not self.initial_capital:
            self._extra_capital = 0
            return

        timeline = self.portfolio.timeline
        if len(timeline) > 0:
            self._extra_capital = timeline[0]['equity'] - self.initial_capital
            timeline[0]['equity'] -= self._extra_capital
            timeline[0]['cash'] -= self._extra_capital

    @property
    def cash(self):
        if (not self.free or not self.context.backtesting()):
            self._update_balance()
        cash = self.free[self.context.base_currency]
        # if self.equity and self.equity > 0:
        #     cash = min(self.equity, cash)
        self._adjust_for_extra_capital()
        return cash - self._extra_capital if self._extra_capital else cash

    @property
    def equity(self):
        return self.portfolio.equity

    @property
    def portfolio(self):
        return self.context.portfolio

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

    def get_balance(self, symbol=None, refresh=False):
        """
        Get account balance for a single asset or for all.
        If refresh parameter is not provided, cached balance can be returned.
        """
        expiry = datetime.datetime.utcnow() - self.datacenter.timeframe_delta
        if refresh or (self.last_update_at and self.last_update_at < expiry):
            self.free, self.locked, self.total = self._update_balance()
            self.last_update_at = datetime.datetime.utcnow()

    def lock_cash_for_order_if_required(self, event, order):
        pass

    def update_after_order_rejected(self, event):
        pass

    def update_after_order_filled(self, event):
        pass

    def update_after_order_unfilled(self, event):
        pass

    @abstractmethod
    def _update_balance(self):
        """ Refresh/update account balance """
        raise NotImplementedError("Account must implement `_update_balance()`")

    @abstractmethod
    def update_after_order(self, _order_filled_event):
        """ Routine to run after a successfully accepted/filled order """
        raise NotImplementedError("Account must implement \
                                  `update_after_order()`")


import math
class SimulatedAccount(AbstractAccount):
    def __init__(self, *args, initial_balance={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_balance = initial_balance

    def assert_balance_matched(self, *assets):
        for asset in list(assets):
            actual = self.free[asset] + self.locked[asset]
            expected = self.total[asset]
            if abs(actual - expected) > 1e-8 and abs(actual/expected - 1) > 1e-6:
                symbol = self.datacenter.assets_to_symbol(asset)
                price = self.datacenter._data_seen[-1].data['history']
                if symbol in price.index:
                    price = price.loc[symbol, 'close']
                else:
                    price = None
                if not price or price*abs(actual - expected) > 1e-8:
                    print("Found balance mismatch for %s: %s (F+L) vs %s (T)" % (asset, actual, expected))
                    from IPython import embed; embed()
                    return False
            elif expected <= -1e-8 and not self.context.allow_shorting:
                print("Found negative balance for %s: %s" % (asset, expected))
                from IPython import embed; embed()
                return False


    def display_stats(self):
        print("Free:   %s" % {k: v for k,v in self.free.items()   if v > 0})
        print("Total:  %s" % {k: v for k,v in self.total.items()  if v > 0})
        print("Locked: %s" % {k: v for k,v in self.locked.items() if v > 0})

    def _update_balance(self):
        if self.free:
            return (self.free, self.locked, self.total)
        if not self.free:
            self.free = self.initial_balance.copy()
            self.total = self.free.copy()
        for sym in self.datacenter.all_symbols():
            asset, base = self.datacenter.symbol_to_assets(sym)
            self.locked[asset] = 0
            self.locked[base] = 0
            if asset not in self.free:
                self.free[asset] = 0
            if base not in self.free:
                self.free[base] = 0
            if asset not in self.total:
                self.total[asset] = 0
            if base not in self.total:
                self.total[base] = 0
        return (self.free, self.locked, self.total)

    def lock_cash_for_order_if_required(self, _event, order):
        if order.quantity > 0:
            self.free[order.base] -= order.order_cost
            self.locked[order.base] += order.order_cost
        elif order.quantity < 0:
            self.free[order.asset] -= abs(order.quantity)
            self.locked[order.asset] += abs(order.quantity)

    def update_after_order_filled(self, event):
        """ Update account balances after an order has been filled """
        order = event.item
        self.free[order.commission_asset] -= order.commission
        self.total[order.commission_asset] -= order.commission

        if order.quantity > 0:
            self.total[order.base] -= order.order_cost
            self.locked[order.base] -= order.order_cost
            self.free[order.asset] += order.quantity
            self.total[order.asset] += order.quantity
        elif order.quantity < 0:
            self.total[order.asset] -= abs(order.quantity)
            self.locked[order.asset] -= abs(order.quantity)
            self.free[order.base] += abs(order.order_cost)
            self.total[order.base] += abs(order.order_cost)

    def update_after_order_unfilled(self, event):
        order = event.item
        if order.quantity > 0:
            self.free[order.base] += order.order_cost
            self.locked[order.base] -= order.order_cost
        elif order.quantity < 0:
            self.free[order.asset] += abs(order.quantity)
            self.locked[order.asset] -= abs(order.quantity)

    def update_after_order_rejected(self, event):
        self.update_after_order_unfilled(event)


class CcxtExchangeAccount(AbstractAccount):
    def __init__(self, exchange, *args, params={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.exchange = getattr(ccxt, exchange)(params)

    def _update_balance(self):
        response = self.exchange.fetch_balance()
        for d in response['info']['balances']:
            self.free[d['asset']] = float(d['free'])
            self.locked[d['asset']] = float(d['locked'])
            self.total[d['asset']] = float(d['free']) + float(d['locked'])
        return (self.free, self.locked, self.total)
