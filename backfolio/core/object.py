from .utils import detect, generate_id
from .event import (
    OrderFilledEvent,
    OrderRejectedEvent,
    OrderUnfilledEvent,
    OrderCreatedEvent,
    OrderPendingEvent
)


class Tick:
    def __init__(self, time, history):
        self.time = time
        self.history = history

    @property
    def data(self):
        return {"time": self.time, "history": self.history}

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.time)


class Advice:
    _ids = []

    def __init__(self, strategy, asset, base, exchange, last_price,
                 quantity, quantity_type, order_type, limit_price=None,
                 max_cost=0, side=None, position=None, time=None, id=0):
        self.asset = asset
        self.base = base
        self.strategy = strategy
        self.exchange = exchange
        self.last_price = last_price
        self.quantity = quantity
        self.quantity_type = quantity_type
        self.order_type = order_type
        self.limit_price = limit_price
        self.id = id if id else generate_id('advice', self)
        self.time = time
        self.max_cost = max_cost
        self.side = side
        self.position = position
        self._symbol = None

    @classmethod
    def _construct_from_data(cls, data, _portfolio):
        advice = cls(data['strategy'], data['asset'], data['base'],
                     data['exchange'], data['last_price'], data['quantity'],
                     data['quantity_type'], data['order_type'],
                     data['limit_price'], data['max_cost'], data['side'],
                     data['position'], data['time'], data['id'])
        return advice

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        self._id = val
        self.__class__._ids.append(val)

    @property
    def max_order_size(self):
        return self.max_cost

    @property
    def is_buy(self):
        return self.side == "BUY"

    @property
    def is_sell(self):
        return self.side == "SELL"

    @property
    def is_limit(self):
        return self.order_type == 'LIMIT'

    @property
    def is_market(self):
        return self.order_type == 'MARKET'

    def symbol(self, context):
        if not self._symbol:
            self._symbol = context.datacenter.assets_to_symbol(
                self.asset, self.base)
        return self._symbol

    @property
    def data(self):
        return {"id": self.id, "time": self.time, "position": self.position,
                "exchange": self.exchange, "strategy": self.strategy,
                "last_price": self.last_price, "quantity": self.quantity,
                "quantity_type": self.quantity_type, "max_cost": self.max_cost,
                "order_type": self.order_type, "limit_price": self.limit_price,
                "side": self.side, "asset": self.asset, "base": self.base}

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.data)


class Order:
    _ids = []

    def __init__(self, advice, exchange, quantity, order_cost,
                 fill_price, commission=0, commission_asset=None,
                 commission_cost=0, local_id=0):
        self._id = 0
        self._local_id = local_id if local_id else generate_id('order', self)
        self.advice = advice
        if exchange:
            self.advice.exchange = exchange
        self._status = 'open'
        self.quantity = quantity
        self.remaining = quantity
        self.order_cost = order_cost
        self.fill_price = fill_price
        self.commission = commission
        self._commission_asset = commission_asset
        self.commission_cost = commission_cost
        self._updated_at = None
        self._time = None

    @classmethod
    def _construct_from_data(cls, data, portfolio):
        advice = detect(portfolio.advice_history,
                        lambda req: req.id == data['advice_id'])
        order = cls(advice, data['exchange'], data['quantity'],
                    data['order_cost'], data['fill_price'], data['commission'],
                    data['commission_asset'], data['commission_cost'],
                    data['local_id'])
        for field in ['id', 'time', 'status', 'updated_at', 'remaining']:
            if field in data and data[field]:
                setattr(order, field, data[field])
        return order

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def local_id(self):
        return self._local_id

    @local_id.setter
    def local_id(self, val):
        self._local_id = val
        self.__class__._ids.append(val)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value

    @property
    def time(self):
        return self._time if self._time else self.advice.time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def side(self):
        if self.advice.side and self.advice.side in ["BUY", "SELL"]:
            return self.advice.side
        else:
            return 'BUY' if self.quantity > 0 else 'SELL'

    @side.setter
    def side(self, val):
        self.advice.side = val

    def symbol(self, context):
        return self.advice.symbol(context)

    @property
    def exchange(self):
        return self.advice.exchange

    @property
    def advice_id(self):
        return self.advice.id

    @property
    def asset(self):
        return self.advice.asset

    @property
    def base(self):
        return self.advice.base

    @property
    def order_type(self):
        return self.advice.order_type

    @property
    def commission_asset(self):
        return self._commission_asset if self._commission_asset else self.base

    @commission_asset.setter
    def commission_asset(self, val):
        self._commission_asset = val

    @property
    def updated_at(self):
        return self._updated_at if self._updated_at else self.time

    @updated_at.setter
    def updated_at(self, val):
        self._updated_at = val

    @property
    def is_limit(self):
        return self.order_type == 'LIMIT'

    @property
    def is_market(self):
        return self.order_type == 'MARKET'

    @property
    def is_buy(self):
        return self.quantity > 0

    @property
    def is_sell(self):
        return self.quantity < 0

    @property
    def is_open(self):
        return self.status == 'open'

    @property
    def is_closed(self):
        return self.status == 'closed' and not self.remaining

    @property
    def is_cancelled(self):
        return self.status == 'cancelled' or self.status == 'rejected'

    @property
    def is_partially_filled(self):
        return self.remaining and self.remaining != self.quantity

    def mark_filled(self, context):
        self.remaining = 0
        self.status = 'closed'
        self.updated_at = context.current_time
        context.events.put(OrderFilledEvent(self))
        return self

    def mark_rejected(self, context, message=None, track=True):
        self.status = 'rejected'
        self.updated_at = context.current_time
        if track:
            context.events.put(OrderRejectedEvent(self, message))
        return self

    def mark_unfilled(self, context):
        self.status = 'cancelled'
        self.cost = 0
        self.updated_at = context.current_time
        context.events.put(OrderUnfilledEvent(self))
        return self

    def mark_created(self, context):
        self.updated_at = context.current_time
        context.events.put(OrderCreatedEvent(self))

    def mark_pending(self, context):
        self.updated_at = context.current_time
        context.events.put(OrderPendingEvent(self))

    def mark_cancelled(self, *args, **kwargs):
        return self.mark_unfilled(*args, **kwargs)

    def mark_closed(self, *args, **kwargs):
        return self.mark_filled(*args, **kwargs)

    @property
    def data(self):
        return {"id": self.id, "local_id": self.local_id, "time": self.time,
                "asset": self.asset, "base": self.base,
                "advice_id": self.advice_id,
                "exchange": self.exchange, "side": self.side,
                "quantity": self.quantity, "remaining": self.remaining,
                "order_cost": self.order_cost, "fill_price": self.fill_price,
                "order_type": self.order_type, "status": self.status,
                "commission": self.commission, "updated_at": self.updated_at,
                "commission_asset": self.commission_asset,
                "commission_cost": self.commission_cost}

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.data)


class OrderGroup:
    def __init__(self, order, local_id=0):
        self.asset = order.asset
        self._local_id = local_id if local_id else generate_id('ogroup', self)
        self.orders = [order]
        self.status = 'OPEN'
        self.buy_quantity = 0
        self.buy_cost = 0
        self.sell_quantity = 0
        self.sell_cost = 0
        self.started_at = order.time
        self.ended_at = order.time
        self.commission = order.commission_cost
        self.local_id = order.local_id
        if order.is_buy:
            self.buy_quantity = order.quantity
            self.buy_cost = order.order_cost
        elif order.is_sell:
            self.sell_quantity = order.quantity
            self.sell_cost = order.order_cost

    @classmethod
    def add_order_to_groups(cls, order, groups):
        matched = [og for og in groups if og.asset == order.asset]
        if not matched:
            groups.append(cls(order))
        else:
            last_group = matched[-1]
            if last_group.closed:
                groups.append(cls(order))
            else:
                last_group.add_order(order)
        return groups

    @classmethod
    def load_all(cls, orders):
        groups = []
        for order in orders:
            cls.add_order_to_groups(order, groups)
        return groups

    @property
    def buy_price(self):
        return self.buy_cost/self.buy_quantity if self.buy_quantity else 0

    @property
    def sell_price(self):
        return self.sell_cost/self.sell_quantity if self.sell_quantity else 0

    @property
    def remaining_quantity(self):
        return self.buy_quantity - self.sell_quantity

    @property
    def total_profits(self):
        return self.sell_cost - self.buy_cost - self.commission

    @property
    def closed(self):
        return self.status == 'CLOSED'

    @property
    def open(self):
        return self.status == 'OPEN'

    def add_order(self, order):
        self.orders.append(order)
        if order.asset != self.asset:
            raise ValueError("Invalid order being added to group: %s" % order)
        if order.is_buy:
            self.buy_quantity += order.quantity
            self.buy_cost += order.order_cost
        elif order.is_sell:
            self.sell_quantity += -order.quantity
            self.sell_cost += -order.order_cost
        self.ended_at = order.time
        self.commission += order.commission_cost
        if self.remaining_quantity < 1e-8:
            self.status = 'CLOSED'
        return self

    def mark_closed(self):
        self.status = "CLOSED"

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.data)

    @property
    def data(self):
        return {"local_id": self.local_id, "started_at": self.started_at,
                "ended_at": self.ended_at, "asset": self.asset,
                "buy_quantity": self.buy_quantity, "buy_cost": self.buy_cost,
                "buy_price": self.buy_price, "sell_price": self.sell_price,
                "sell_quantity": self.sell_quantity,
                "sell_cost": self.sell_cost,
                "remaining_quantity": self.remaining_quantity,
                "total_profits": self.total_profits, "status": self.status,
                "commission": self.commission, "num_orders": len(self.orders)}
