import ccxt
from math import isnan
from datetime import datetime
from abc import ABCMeta, abstractmethod
from .core.utils import fast_xs, comp8
from .core.object import Order
from ccxt.base.errors import InsufficientFunds, OrderNotFound


class AbstractBroker(object):
    """
    AbstractBroker is an abstract base class providing an interface for
    all subsequent (inherited) brokers.

    The goal of a (derived) BaseBroker object is to place new orders (can be
    market, limit order, etc.) generated by the portfolio to the exchange or
    reject them if required.

    A broker lets us know whether the order that was placed was accepted by the
    exchange, or rejected by it. If accepted, it also makes us aware of the
    price at which the order was filled.

    Brokers can be used to subclass simulated brokerages or live brokerages,
    with identical interfaces. This allows strategies to be backtested in a
    very similar manner to the live trading engine.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        self.session_fields = []

    def reset(self, context):
        """ Routine to run when trading session is resetted. """
        self.context = context
        self.min_order_size = 1e-4
        self.max_order_size = 0
        self.max_position_held = 0
        return self

    @property
    def account(self):
        return self.context.account

    @property
    def portfolio(self):
        return self.context.portfolio

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
    def min_order_size(self):
        return self._min_order_size

    @min_order_size.setter
    def min_order_size(self, size=0.0001):
        self._min_order_size = size

    @property
    def max_order_size(self):
        return self._max_order_size

    @max_order_size.setter
    def max_order_size(self, size=0):
        self._max_order_size = size

    @property
    def max_position_held(self):
        return self._max_position_held

    @max_position_held.setter
    def max_position_held(self, equity=0):
        self._max_position_held = equity

    @abstractmethod
    def create_order_after_placement(self, _order_requested_event):
        """
        Execute an order generated by the portfolio.
        """
        raise NotImplementedError("Broker must implement \
                                  `create_order_after_placement()`")

    @abstractmethod
    def execute_order_after_creation(self, _order_requested_event):
        """
        Execute an order generated by the portfolio.
        """
        raise NotImplementedError("Broker must implement \
                                  `execute_order_after_creation()`")

    @abstractmethod
    def cancel_pending_orders(self):
        """
        Cancel all unfilled/pending orders.
        """
        raise NotImplementedError("Broker must implement \
                                  `cancel_pending_orders()`")

    def check_order_statuses(self):
        pass


class SimulatedBroker(AbstractBroker):
    """
    Broker used for backtesting.

    An order is accepted if we have the required cash, and rejected otherwise.

    A MARKET order is executed at the opening price of next tick.
    A LIMIT order is executed if the range of next tick covers LIMIT price.
    """

    def check_order_statuses(self):
        for order in self.portfolio.open_orders:
            order.mark_pending(self)

    def create_order_after_placement(self, order_requested_event,
                                     exchange=None):
        advice = order_requested_event.item
        symbol = advice.symbol(self)

        if symbol not in self.datacenter._current_real.index:
            return

        if not exchange:
            if hasattr(self.datacenter.exchange, "name"):
                exchange = self.datacenter.exchange.name
            else:
                exchange = self.datacenter.name

        quantity, cost, price = self.calculate_order_shares_and_cost(advice)
        if abs(quantity) < 1e-8 or price < 1e-8:  # empty order
            return
        comm, comm_asset, comm_rate, comm_cost = self.calculate_commission(
            advice, quantity, cost)

        order = Order(advice, exchange, quantity, cost,
                      price, comm, comm_asset, comm_cost)
        order.mark_created(self)
        return order

    def execute_order_after_creation(self, order_created_event, exchange=None):
        # FIXME: if order is pending recalculate order cost, price, etc.
        order = order_created_event.item
        pending = order_created_event.pending
        quantity, price = order.quantity, order.fill_price
        comm = order.commission
        comm_asset_balance = self.account.free[order.commission_asset]
        symbol = order.symbol(self)
        symbol_data = None
        pbase = self.context.base_currency
        base_rate = 1 if order.base == pbase else self.get_asset_rate(order.base)
        tx_asset = order.base if order.is_buy else order.asset
        cash, cost = self.account.free[order.base], order.order_cost
        asset_pos = self.account.free[order.asset]

        if symbol in self.datacenter._current_real.index:
            symbol_data = fast_xs(self.datacenter._current_real, symbol)

        if (not symbol_data and order.is_open and
                not self.datacenter._current_real.empty):
            order.mark_rejected(self, 'Symbol obsolete: %s' % symbol)
        # elif cash < 1e-8 and order.base == self.context.base_currency:
        #     order.mark_rejected(self, 'Cash depleted')
        #     # self.datacenter._continue_backtest = False
        elif order.is_buy and comp8(cost, cash) > 0 and not pending:
            order.mark_rejected(
                self,
                'Insufficient Base Asset: %0.8f (cost) vs %0.8f (cash)' % (
                    cost, cash))
        elif (order.is_sell and comp8(abs(order.quantity), asset_pos) > 0 and
                not pending):
            order.mark_rejected(
                self,
                'Insufficient Tx Asset: %0.8f (quantity) vs %0.8f (available)'
                % (abs(order.quantity), asset_pos))
        elif ((comp8(comm, comm_asset_balance) > 0 or
                comm_asset_balance < 0) and not pending):
            order.mark_rejected(
                self,
                'Insufficient Brokerage: %0.8f (comm) vs %0.8f (asset bal)' % (
                    comm, comm_asset_balance))
        elif (order.commission_asset == self.context.base_currency and
                comp8(comm, self.account.cash) > 0):
            order.mark_rejected(
                self,
                'Insufficient Cash: %0.8f (comm) vs %0.8f (asset bal)' % (
                    comm, self.account.cash))
        elif comp8(abs(cost)*base_rate, self.min_order_size) < 0:
            return
            # order.mark_rejected(self, track=False)
        elif order.is_open and order.is_buy and order.quantity < 0:
            order.mark_rejected(
                self, 'Cannot sell asset for BUY order: %s' % order)
        elif order.is_open and order.is_sell and order.quantity > 0:
            order.mark_rejected(
                self, 'Cannot buy asset for SELL order: %s' % order)
        elif order.is_open and order.order_type == 'LIMIT' and (
                quantity > 0 and price >= symbol_data['low'] and
                self.context.consider_limit_filled_on_touch):
            order.mark_closed(self, limit_price=limit_price)
        elif order.is_open and order.order_type == 'LIMIT' and (
                quantity < 0 and price <= symbol_data['high'] and
                self.context.consider_limit_filled_on_touch):
            order.mark_closed(self)
        elif order.is_open and order.order_type == 'LIMIT' and (
                quantity > 0 and price > symbol_data['low']):
            order.mark_closed(self)
        elif order.is_open and order.order_type == 'LIMIT' and (
                quantity < 0 and price < symbol_data['high']):
            order.mark_closed(self)
        elif order.is_open and order.order_type == 'MARKET':
            order.mark_closed(self)
        return order

    # OPTIMIZE: takes a really long time ~10%
    def cancel_pending_orders(self, symbol=None):
        for order in self.portfolio.open_orders:
            if not order.is_cancelled and (not symbol or order.symbol == symbol):
                order.mark_cancelled(self)

    # TODO: instead allow passing a % which will be added to opening price
    # for limit orders
    def get_order_price(self, advice):
        symbol = advice.symbol(self)
        price = advice.limit_price
        opened_at = fast_xs(self.datacenter._current_real, symbol)['open']
        if price is None or advice.order_type == 'MARKET':
            if symbol not in self.datacenter._current_real.index:
                return
            price = opened_at
        elif price is not None:
            if price >= opened_at and advice.quantity > 0:
                price = opened_at
            if price <= opened_at and advice.quantity < 0:
                price = opened_at
        return price

    def get_slippage(self, advice, direction):
        slippage = 0
        if advice.order_type == 'MARKET':
            slippage = self.context.slippage()/100
            slippage *= -1 if direction < 0 else 1
        return slippage

    def get_asset_rate(self, asset):
        symbol = self.datacenter.assets_to_symbol(asset, self.context.base_currency)
        return fast_xs(self.datacenter._current_real, symbol)['open']

    def calculate_order_shares_and_cost(self, advice):
        # Get the specified LIMIT price or open price of next tick for MARKET
        pbase = self.context.base_currency
        price = self.get_order_price(advice)
        base_rate = 1 if advice.base == pbase else self.get_asset_rate(advice.base)

        # If we can not get price, or if quantity was specified to be 0 (sell
        # everything) and we do not have any position, do nothin.
        if not price or (advice.quantity == 0 and not advice.position):
            return (0, 0, 0)

        quantity = 0
        if advice.quantity_type == 'SHARE':
            # Quantity remains fixed here. Price can be varied.
            # However, when order size is limited, we have no way to reduce
            # order size but to decrease quantity.
            quantity = advice.quantity
            price = price * (1 + self.get_slippage(advice, quantity))
        elif advice.quantity_type == 'PERCENT':
            # Cost remains fixed here. Price/Quantity can be varied.
            required = self.account.equity * advice.quantity/100. / base_rate
            required = required - advice.position * price
            price = price * (1 + self.get_slippage(advice, required))
            quantity = required/price
        elif advice.quantity_type == 'REL_PERCENT':
            # Cost remains fixed here. Price/Quantity can be varied.
            required = advice.quantity/100.0*self.account.equity / base_rate
            price = price * (1 + self.get_slippage(advice, required))
            quantity = required/price

        # lets, calculate the cost of this order
        cost = quantity * price

        # ensure that we do not hold more than the specified amount of equity
        # for this asset. We decrease the quantity such that the new asset
        # allocation is equal to max_position_held.
        if self.max_position_held >= 1e-8:
            asset_rate = 1 if advice.asset == pbase else self.get_asset_rate(advice.asset)
            equity = (advice.position + quantity) * asset_rate
            if abs(equity) > self.max_position_held:
                quantity = (1 if cost > 0 else -1)*self.max_position_held/asset_rate
                quantity -= advice.position
                cost = quantity * price

        # ensure that the cost does not exceed max permittable order size
        # globally for the broker.
        if self.max_order_size and abs(cost) > self.max_order_size / base_rate:
            cost = self.max_order_size * (-1 if cost < 0 else 1) / base_rate
            quantity = cost/price

        # if the advice has a max order size defined with it, limit to that
        if advice.max_order_size and abs(cost) > advice.max_order_size / base_rate:
            cost = advice.max_order_size * (-1 if cost < 0 else 1) / base_rate
            quantity = cost/price

        # finally, allow the strategy to modify order cost, quantity or price
        # cost should be returned in same currency as advice.base
        if hasattr(self.strategy, 'transform_order_calculation'):
            tp = price if advice.order_type == 'LIMIT' else None
            cost, quantity, tp = self.strategy.transform_order_calculation(
                advice, cost, quantity, tp)
            if advice.order_type == 'LIMIT' and tp is None:
                advice.order_type = 'MARKET'
                advice.limit_price = None
                price = self.get_order_price(advice)
                price = price * (1 + self.get_slippage(advice, quantity))
            elif advice.order_type == 'LIMIT' and price != tp:
                price = tp
            elif advice.order_type == 'MARKET' and tp:
                advice.order_type = 'LIMIT'
                price = tp
            quantity = cost/price

        # ensure that when selling, we cannot sell more than free quantity
        if quantity < 0 and abs(quantity) > self.account.free[advice.asset]:
            quantity = -self.account.free[advice.asset]
            cost = quantity * price

        return (round(quantity, 8), round(price*quantity, 8), round(price, 8))

    def calculate_commission(self, advice, asset_quantity, order_cost):
        comm_asset = self.context.commission_asset

        order_cost_in_base = order_cost
        if advice.base != self.context.base_currency:
            base_asset_rate = self.get_asset_rate(advice.base)
            order_cost_in_base = order_cost * base_asset_rate

        comm_asset_rate = 1
        if comm_asset and comm_asset != self.context.base_currency:
            comm_asset_rate = self.get_asset_rate(comm_asset)

        comm = self.context.commission
        if callable(self.context.commission):
            comm_cost, comm_asset = self.context.commission(
                    advice, asset_quantity, order_cost_in_base, comm_asset_rate)
        else:
            comm_cost = self.context.commission/100. * abs(order_cost_in_base)
        comm = comm_cost/comm_asset_rate
        return (round(comm, 8), comm_asset, comm_asset_rate, comm_cost)


class CcxtExchangePaperBroker(SimulatedBroker):
    def __init__(self, exchange, params={}):
        super().__init__()
        self.exchange = getattr(ccxt, exchange)(params)

    def get_slippage(self, advice, direction):
        """ 1/3rd slippage than backtests to account for the fact that we are
        placing an order in real time. """
        return super().get_slippage(advice, direction)/3

    def get_order_price(self, advice):
        symbol = advice.symbol(self)
        price = advice.limit_price
        if price is None:
            book = self.exchange.fetch_order_book(symbol)
            if advice.quantity > 0:
                price = book['bids'][0][0]
            else:
                price = book['asks'][0][0]
        return price

    def get_commission_asset_rate(self, comm_symbol):
        return self.exchange.fetch_ticker(comm_symbol)['last']

    def create_order_after_placement(self, order_requested_event,
                                     exchange=None):
        """
        Execute an order in paper trading mode.
        All orders are converted to MARKET orders in this mode with minimal
        slippage (price for order is obtained in real-time).
        """
        order_requested_event.order_type = 'MARKET'
        super().create_order_after_placement(
            order_requested_event, self.exchange.name)

    def execute_order_after_creation(self, order_created_event,
                                     exchange=None):
        """
        Execute an order in paper trading mode.
        All orders are converted to MARKET orders in this mode with minimal
        slippage (price for order is obtained in real-time).
        """
        order_created_event.order_type = 'MARKET'
        super().execute_order_after_creation(
            order_created_event, self.exchange.name)


class CcxtExchangeBroker(CcxtExchangePaperBroker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_slippage(self, _advice, _direction):
        return 0

    def cancel_pending_orders(self, symbol=None):
        """
        API calls made to obtain pending orders without a symbol is
        throttled in general.

        Polling frequency should be set to above 2-3 mins for avoiding bans.
        For example, on binance this call is rate-limited to 1 per 154sec.
        """
        orders = self.portfolio.orders
        for idx, order in enumerate(orders):
            if not order.id or isnan(order.id) or not order.is_open:
                continue
            try:
                symbol = order.symbol(self)
                self.exchange.cancel_order(order.id, symbol)
                order.mark_cancelled(self)
                self.context.notify(
                    "Cancelled open %4s order: %s for %s at %.8f" % (
                     order.side, order.id, order.asset,
                     order.fill_price), formatted=True)
            except OrderNotFound:
                pass
            except Exception as e:
                self.context.notify_error(e)

    def reject_order(self, order, reason, track=True):
        order.mark_rejected(self, '[BROKER]: %s' % reason, track)
        return order

    def __place_order_on_exchange(self, order, symbol, quantity, price):
        resp = None
        try:
            if order.order_type == 'LIMIT' and quantity > 0:
                resp = self.exchange.create_limit_buy_order(
                    symbol, quantity, price)
            elif order.order_type == 'LIMIT' and quantity < 0:
                resp = self.exchange.create_limit_sell_order(
                    symbol, abs(quantity), price)
            elif order.order_type == 'MARKET' and quantity > 0:
                resp = self.exchange.create_market_buy_order(
                    symbol, quantity)
            elif order.order_type == 'MARKET' and quantity < 0:
                resp = self.exchange.create_market_sell_order(
                    symbol, abs(quantity))
        except InsufficientFunds:
            message = "Insufficient funds for Order: %s, Q:%.8f, P:%.8f" % (
                order, quantity, price)
            order.mark_rejected(self, message)
            self.notify(message, formatted=True)
        except Exception as e:
            message = "Failed Order: %s, Q:%0.8f, P:%0.8f" % (
                order, quantity, price)
            order.mark_rejected(self, message)
            self.notify(message, formatted=True)
            self.notify_error(e)
        finally:
            return resp

    def execute_order_after_creation(self, order_created_event, exchange=None):
        order = order_created_event.item
        if order.id:
            return
        quantity, price = order.quantity, order.fill_price
        cash, cost = self.account.free[order.base], order.order_cost
        cost = order.order_cost
        symbol = order.symbol(self)

        market_data = self.datacenter.load_markets()
        if symbol not in market_data:
            order.mark_rejected(self, 'Symbol obsolete: %s' % symbol, track=False)
        limits = market_data[symbol]['limits']

        if not quantity:
            return self.reject_order(order, 'Quantity zero?', track=False)
        if abs(quantity) < limits['amount']['min']:
            return self.reject_order(order, 'Quantity < min. Exchange Value', track=False)
        if abs(cost) < limits['cost']['min']:
            return self.reject_order(order, 'Cost < min. Exchange Value', track=False)
        if price < limits['price']['min']:
            return self.reject_order(order, 'Price < min. Exchange Value', track=False)
        if abs(cost) < self.min_order_size:
            return self.reject_order(order, 'Cost < min. Order Size', track=False)

        # create order on exchange
        resp = self.__place_order_on_exchange(order, symbol, quantity, price)
        if not resp:
            return

        cost = resp['cost'] if resp['cost'] else resp['amount']*resp['price']
        comm = resp['fee']['cost'] if resp['fee'] else 0
        comm_asset = resp['fee']['currency'] if resp['fee'] else None
        order.id = resp['id']
        order.time = resp['datetime']
        order.cost = cost
        order.commission = comm
        order.commission_asset = comm_asset
        return order

    def check_order_statuses(self):
        orders = [o for o in self.portfolio.orders
                  if o.id and not isnan(o.id) and o.is_open]

        for idx, order in enumerate(orders):
            resp = {}
            symbol = order.symbol(self)

            try:
                resp = self.exchange.fetch_order(order.id, symbol)
            except Exception as e:
                resp = {'status': 'unknown', 'remaining': 0,
                        'datetime': datetime.utcnow()}

            order.status = resp['status']
            order.remaining = resp['remaining']
            order.updated_at = resp['datetime']
            self.portfolio.orders[idx] = order

            if order.is_closed:
                order.mark_closed(self)
            elif order.is_cancelled:
                data = order.data.copy()
                data['quantity'] = data['quantity'] - data['remaining']
                if data['quantity']:
                    Order(**data).mark_closed(self)

                data = order.data.copy()
                data['quantity'] = data['remaining']
                Order(**data).mark_cancelled(self)
            else:
                order.mark_pending(self)
