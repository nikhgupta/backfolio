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
        advice = cls(data['strategy'], data['symbol'],
                     data['exchange'], data['last_price'], data['quantity'],
                     data['quantity_type'], data['order_type'],
                     data['limit_price'], data['time'], data['id'])
        for field in ['time', 'max_cost', 'side', 'position']:
            if field in data and data[field]:
                setattr(advice, field, data[field])
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
                 fill_price, commission=0, commission_asset=None, local_id=0):
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
        self._updated_at = None
        self._time = None

    @classmethod
    def _construct_from_data(cls, data, portfolio):
        advice = detect(portfolio.advice_history,
                        lambda req: req['id'] == data['advice_id'])
        advice = Advice._construct_from_data(advice, portfolio)
        order = cls(advice, data['exchange'], data['quantity'],
                    data['order_cost'], data['fill_price'], data['commission'],
                    data['commission_asset'], data['local_id'])
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
        if self.advice.side:
            return self.advice.side
        else:
            return 'buy' if self.quantity > 0 else 'sell'

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
        return self.status == 'cancelled'

    @property
    def is_partially_filled(self):
        return self.remaining and self.remaining != self.quantity

    def mark_filled(self, context):
        if self.status != 'open':
            raise ValueError("Cannot mark order as closed: %s" % self)
        self.remaining = 0
        self.status = 'closed'
        self.updated_at = context.current_time
        context.events.put(OrderFilledEvent(self))
        return self

    def mark_rejected(self, context, message=None, track=True):
        if self.status != 'open':
            raise ValueError("Cannot mark order as rejected: %s" % self)
        self.status = 'rejected'
        self.updated_at = context.current_time
        if track:
            context.events.put(OrderRejectedEvent(self, message))
        return self

    def mark_unfilled(self, context):
        if self.status != 'open':
            raise ValueError("Cannot mark order as unfilled: %s" % self)
        self.status = 'cancelled'
        self.cost = 0
        self.updated_at = context.current_time
        context.events.put(OrderUnfilledEvent(self))
        return self

    def mark_created(self, context):
        if self.status != 'open':
            raise ValueError("Cannot mark order as created: %s" % self)
        self.updated_at = context.current_time
        context.events.put(OrderCreatedEvent(self))

    def mark_pending(self, context):
        if self.status != 'open':
            raise ValueError("Cannot mark order as pending: %s" % self)
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
                "commission_asset": self.commission_asset}

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.data)
