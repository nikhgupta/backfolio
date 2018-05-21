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
        self.balance = {}
        self.last_update_at = None
        self.session_fields = []

    def __repr__(self):
        bal = dict([key, val] for key, val in self.balance.items() if val > 0)
        return "%s(%s)" % (self.__class__.__name__, bal)

    def reset(self, context):
        """ Routine to run when trading session is resetted. """
        self.context = context
        self.balance = {}
        self.last_update_at = None
        self.balance_at_ticks = []
        self.get_balance(refresh=True)

    @property
    def cash(self):
        return self.balance[self.context.base_currency]

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
            self.balance = self._update_balance()
            self.last_update_at = datetime.datetime.utcnow()
        return self.balance[symbol] if symbol else self.balance

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
        if not self.balance:
            self.balance = self.initial_balance.copy()
        for sym in self.datacenter.all_symbols():
            asset, base = self.datacenter.symbol_to_assets(sym)
            if asset not in self.balance:
                self.balance[asset] = 0
            if base not in self.balance:
                self.balance[base] = 0
        return self.balance

    def update_after_order(self, order):
        """ Update account balances after an order has been filled """
        self.balance[order.base] -= order.order_cost
        self.balance[order.commission_asset] -= order.commission
        if order.asset not in self.balance:
            self.balance[order.asset] = 0
        self.balance[order.asset] += order.quantity


class CcxtExchangeAccount(AbstractAccount):
    def __init__(self, exchange, params={}):
        super().__init__()
        self.exchange = getattr(ccxt, exchange)(params)

    @property
    def cash(self):
        response = self.exchange.fetch_balance()
        return response[self.context.base_currency]['free']

    def _update_balance(self):
        response = self.exchange.fetch_balance()
        for d in response['info']['balances']:
            self.balance[d['asset']] = float(d['free']) + float(d['locked'])
        return self.balance

    def update_after_order(self, order):
        pass
