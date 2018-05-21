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

    def __init__(self):
        self.free = {}
        self.locked = {}
        self.total = {}
        self.last_update_at = None
        self.session_fields = []

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

    @property
    def cash(self):
        self._update_balance()
        return self.free[self.context.base_currency]

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


class SimulatedAccount(AbstractAccount):
    def __init__(self, initial_balance={}):
        super().__init__()
        self.initial_balance = initial_balance

    def _update_balance(self):
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
        else:
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
        else:
            self.total[order.asset] -= abs(order.quantity)
            self.locked[order.asset] -= abs(order.quantity)
            self.free[order.base] += abs(order.order_cost)
            self.total[order.base] += abs(order.order_cost)

    def update_after_order_unfilled(self, event):
        order = event.item
        if order.quantity > 0:
            self.free[order.base] += order.order_cost
            self.locked[order.base] -= order.order_cost
        else:
            self.free[order.asset] += abs(order.quantity)
            self.locked[order.asset] -= abs(order.quantity)

    def update_after_order_rejected(self, event):
        self.update_after_order_unfilled(event)


class CcxtExchangeAccount(AbstractAccount):
    def __init__(self, exchange, params={}):
        super().__init__()
        self.exchange = getattr(ccxt, exchange)(params)

    def _update_balance(self):
        response = self.exchange.fetch_balance()
        for d in response['info']['balances']:
            self.free[d['asset']] = float(d['free'])
            self.locked[d['asset']] = float(d['locked'])
            self.total[d['asset']] = float(d['free']) + float(d['locked'])
        return (self.free, self.locked, self.total)
