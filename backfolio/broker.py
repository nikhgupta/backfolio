import ccxt
from math import isnan
from datetime import datetime
from abc import ABCMeta, abstractmethod
from ccxt.base.errors import OrderNotFound
from .core.utils import fast_xs
from .core.object import Order
from .core.event import (
    OrderFilledEvent,
    OrderRejectedEvent,
    OrderUnfilledEvent,
    OrderCreatedEvent
)


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

    def create_order_after_placement(self, order_requested_event,
                                     exchange=None):
        request = order_requested_event.item
        self._future_tick_data = self.datacenter._current_real
        symbol = self.datacenter.assets_to_symbol(request.asset, request.base)

        if symbol not in self._future_tick_data.index:
            return

        if not exchange:
            exchange = self.datacenter.exchange.name

        quantity, cost, price = self.calculate_order_shares_and_cost(request)
        comm, comm_asset, comm_rate = self.calculate_commission(
            request, quantity, cost)

        order = Order(request, exchange, quantity, cost,
                      price, comm, comm_asset)
        self.events.put(OrderCreatedEvent(order))
        return order

    def execute_order_after_creation(self, order_created_event, exchange=None):
        order = order_created_event.item
        cash, cost = self.account.cash, order.order_cost
        quantity, price = order.quantity, order.fill_price
        comm = order.commission
        comm_asset_balance = self.account.balance[order.commission_asset]
        symbol = self.datacenter.assets_to_symbol(order.asset, order.base)
        symbol_data = fast_xs(self._future_tick_data, symbol)

        if cash < 1e-8:
            order.mark_rejected()
            event = OrderRejectedEvent(order, 'Cash depleted.')
            self.datacenter._continue_backtest = False
        elif abs(cost) < self.min_order_size:
            # NOTE: we are in a backtest session, and this may produce lot
            # of rejected orders, so we will just ignore this case.
            return
            # event = OrderRejectedEvent(order, 'Cost < Min Order Size')
        elif cost > cash:
            order.mark_rejected()
            event = OrderRejectedEvent(order, 'Insufficient Cash: %0.8f \
                                       (cost) vs %0.8f (cash)' % (cost, cash))
        elif order.order_type == 'LIMIT' and (
                quantity > 0 and price <= symbol_data['low']):
            order.mark_unfilled()
            event = OrderUnfilledEvent(order, 'Limit order unfilled: %0.8f \
                                       (price) vs %0.8f (low)' % (
                                           price, symbol_data['low']))
        elif order.order_type == 'LIMIT' and (
                quantity < 0 and price >= symbol_data['high']):
            order.mark_unfilled()
            event = OrderUnfilledEvent(order, 'Limit order unfilled: %0.8f \
                                       (price) vs %0.8f (high)' % (
                                           price, symbol_data['high']))
        elif comm > comm_asset_balance:
            order.mark_rejected()
            event = OrderRejectedEvent(order, 'Insufficient Brokerage: %0.8f \
                                       (comm) vs %0.8f (asset bal)' % (
                                           comm, comm_asset_balance))
        else:
            order.mark_closed()
            event = OrderFilledEvent(order)
        self.events.put(event)
        return order

    def cancel_pending_orders(self, filter_fn=None):
        for order in self.portfolio.orders:
            if filter_fn and not filter_fn(order):
                continue
            if order['status'] == 'open':
                order = Order.__construct_from_data(order, self.portfolio)
                order.mark_cancelled()
                self.events.put(OrderUnfilledEvent(order))

    # TODO: instead allow passing a % which will be added to opening price
    # for limit orders
    def get_order_price(self, request):
        symbol = self.datacenter.assets_to_symbol(request.asset, request.base)
        price = request.limit_price
        if price is None:
            if symbol not in self._future_tick_data.index:
                return
            price = fast_xs(self._future_tick_data, symbol)['open']
        return price

    def get_slippage(self, request, direction):
        slippage = 0
        if request.order_type == 'MARKET':
            slippage = self.context.slippage()/100
            slippage *= -1 if direction < 0 else 1
        return slippage

    def get_commission_asset_rate(self, comm_symbol):
        return fast_xs(self._future_tick_data, comm_symbol)['close']

    def calculate_order_shares_and_cost(self, request):
        price = self.get_order_price(request)
        if not price or request.quantity + request.position == 0:
            return (0, 0, 0)

        quantity = 0
        if request.quantity_type == 'SHARE':
            quantity = request.quantity
            price = price * (1 + self.get_slippage(request, quantity))
        elif request.quantity_type == 'PERCENT':
            required = self.account.equity * request.quantity/100.
            required = required - request.position * price
            price = price * (1 + self.get_slippage(request, required))
            quantity = required/price

        # ensure that we do not hold more than the specified amount of equity
        # for this asset
        if self.max_position_held:
            equity = (request.position + quantity) * price
            if equity > self.max_position_held:
                quantity = self.max_position_held/price - request.position

        cost = quantity * price

        # ensure that the cost does not exceed max permittable order size
        if self.max_order_size and abs(cost) > self.max_order_size:
            cost = self.max_order_size * (-1 if cost < 0 else 1)
            quantity = cost/price

        # if the request has a max order size defined with it, limit to that
        if request.max_order_size and abs(cost) > request.max_order_size:
            cost = request.max_order_size * (-1 if cost < 0 else 1)
            quantity = cost/price

        if cost > 0 and cost > self.account.cash:
            cost = self.account.cash * 0.98
            quantity = cost/price

        # now that we have the cost
        return (round(quantity, 8), round(cost, 8), round(price, 8))

    def calculate_commission(self, request, asset_quantity, order_cost):
        comm_rate = 1
        comm_asset = self.context.commission_asset
        if comm_asset and comm_asset != request.base:
            comm_sym = self.datacenter.assets_to_symbol(
                comm_asset, request.base)
            comm_rate = self.get_commission_asset_rate(comm_sym)
        else:
            comm_asset = request.base
        comm = self.context.commission/100. * abs(order_cost)
        comm /= comm_rate
        return (round(comm, 8), comm_asset, comm_rate)


class CcxtExchangePaperBroker(SimulatedBroker):
    def __init__(self, exchange, params={}):
        super().__init__()
        self.exchange = getattr(ccxt, exchange)(params)

    def get_slippage(self, request, direction):
        """ 1/3rd slippage than backtests to account for the fact that we are
        placing an order in real time. """
        return super().get_slippage(request, direction)/3

    def get_order_price(self, request):
        symbol = self.datacenter.assets_to_symbol(request.asset, request.base)
        price = request.limit_price
        if price is None:
            book = self.exchange.fetch_order_book(symbol)
            if request.quantity > 0:
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
    def __init__(self, *args):
        super().__init__(*args)

    def get_slippage(self, _request, _direction):
        return 0

    def cancel_pending_orders(self, symbol=None):
        """
        API calls made to obtain pending orders without a symbol is
        throttled in general.

        Polling frequency should be set to above 2-3 mins for avoiding bans.
        For example, on binance this call is rate-limited to 1 per 154sec.
        """
        self.exchange.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
        orders = self.exchange.fetch_open_orders()
        for order in orders:
            try:
                self.exchange.cancel_order(order['id'], order['symbol'])
                self.context.notify(
                    "Cancelled open %4s order: %s for %s at %.8f" % (
                     order['side'], order['id'], order['symbol'],
                     order['price']), formatted=True)
            except ccxt.base.errors.OrderNotFound:
                pass
            except Exception as e:
                self.context.notify_error(e)

    def reject_order(self, order, reason):
        order.mark_rejected()
        event = OrderRejectedEvent(order, '[BROKER]: %s' % reason)
        self.events.put(event)
        return order

    def create_order_after_placement(self, order_requested_event,
                                     exchange=None):
        request = order_requested_event.item
        if not exchange:
            exchange = self.datacenter.exchange.name

        quantity, cost, price = self.calculate_order_shares_and_cost(request)
        order = Order(request, exchange, quantity, cost, price,
                      0, self.context.commission_asset)
        self.context.events.put(OrderCreatedEvent(order))

    def execute_order_after_creation(self, order_created_event, exchange=None):
        order = order_created_event.item
        price = order.fill_price
        quantity = order.quantity
        cost = order.order_cost
        symbol = self.datacenter.assets_to_symbol(order.asset, order.base)

        market_data = self.datacenter.load_markets()
        if symbol not in market_data:
            return
        limits = market_data[symbol]['limits']

        if not quantity:
            return self.reject_order(order, 'Quantity zero?')
        if abs(quantity) < limits['amount']['min']:
            return self.reject_order(order, 'Quantity < min. Exchange Value')
        if abs(cost) < limits['cost']['min']:
            return self.reject_order(order, 'Cost < min. Exchange Value')
        if price < limits['price']['min']:
            return self.reject_order(order, 'Price < min. Exchange Value')
        if abs(cost) < self.min_order_size:
            return self.reject_order(order, 'Cost < min. Order Size')

        # create order on exchange
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
        orders = self.portfolio.orders

        for idx, order in enumerate(orders):
            order = Order._construct_from_data(order, self.portfolio)
            if not order.id or isnan(order.id) or not order.is_open:
                continue

            resp = {}
            symbol = self.datacenter.assets_to_symbol(order.asset, order.base)

            try:
                resp = self.exchange.fetch_order(order.id, symbol)
            except Exception as e:
                resp = {'status': 'unknown', 'remaining': 0,
                        'datetime': datetime.utcnow()}

            order.status = resp['status']
            order.remaining = resp['remaining']
            order.updated_at = resp['datetime']
            self.portfolio.orders[idx] = order.data

            if order.is_closed:
                self.context.events.put(OrderFilledEvent(order))
            elif order.is_cancelled:
                data = order.data.copy()
                data['quantity'] = data['quantity'] - data['remaining']
                if data['quantity']:
                    self.context.events.put(OrderFilledEvent(Order(**data)))

                data = order.data.copy()
                data['quantity'] = data['remaining']
                self.context.events.put(OrderUnfilledEvent(Order(**data)))
