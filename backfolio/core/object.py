from .utils import detect, generate_id


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

    def __init__(self, strategy, symbol, exchange, last_price,
                 quantity, quantity_type, order_type, limit_price=None,
                 max_cost=0, time=None, id=0):
        self.symbol = symbol
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

    @classmethod
    def _construct_from_data(cls, data, _portfolio):
        advice = cls(data['strategy'], data['symbol'],
                     data['exchange'], data['last_price'], data['quantity'],
                     data['quantity_type'], data['order_type'],
                     data['limit_price'], data['time'], data['id'])
        if 'max_cost' in data:
            advice.max_cost = data['max_cost']
        for field in ['time']:
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
    def data(self):
        return {"id": self.id, "time": self.time, "symbol": self.symbol,
                "exchange": self.exchange, "strategy": self.strategy,
                "last_price": self.last_price, "quantity": self.quantity,
                "quantity_type": self.quantity_type, "max_cost": self.max_cost,
                "order_type": self.order_type, "limit_price": self.limit_price}

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.data)


class OrderRequest:
    _ids = []

    """
    Quantity of an asset should be in number of shares, and can be positive or
    negative depending on whether we want to LONG or SHORT a position.

    To exit a position, order quantity can be specified as 0.
    """
    def __init__(self, advice, asset, base, position=0, id=0):
        self.advice = advice
        self.base = base
        self.asset = asset
        self.position = position
        self.id = id if id else generate_id('request', self)

    @classmethod
    def _construct_from_data(cls, data, portfolio):
        advice = detect(portfolio.advice_history,
                        lambda adv: adv['id'] == data['advice_id'])
        advice = Advice._construct_from_data(advice, portfolio)
        request = cls(advice, data['asset'], data['base'], data['position'],
                      data['id'])
        return request

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, val):
        self._id = val
        self.__class__._ids.append(val)

    @property
    def advice_id(self):
        return self.advice.id

    @property
    def time(self):
        return self.advice.time

    @property
    def exchange(self):
        return self.advice.exchange

    @property
    def quantity(self):
        return self.advice.quantity

    @property
    def quantity_type(self):
        return self.advice.quantity_type

    @property
    def order_type(self):
        return self.advice.order_type

    @property
    def limit_price(self):
        return self.advice.limit_price

    @property
    def max_order_size(self):
        return self.advice.max_cost

    @property
    def data(self):
        return {"id": self.id, "time": self.time, "advice_id": self.advice_id,
                "asset": self.asset, "base": self.base,
                "quantity": self.quantity, "quantity_type": self.quantity_type,
                "order_type": self.order_type, "position": self.position,
                "limit_price": self.limit_price,
                "max_order_size": self.max_order_size}

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.data)


class Order:
    _ids = []

    def __init__(self, order_request, exchange, quantity, order_cost,
                 fill_price, commission=0, commission_asset=None, local_id=0):
        self._id = 0
        self._local_id = local_id if local_id else generate_id('order', self)
        self.order_request = order_request
        if exchange:
            self.order_request.advice.exchange = exchange
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
        request = detect(portfolio.order_requests,
                         lambda req: req['id'] == data['order_request_id'])
        request = OrderRequest._construct_from_data(request, portfolio)
        order = cls(request, data['exchange'], data['quantity'],
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
        return self._time if self._time else self.order_request.time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def exchange(self):
        return self.order_request.exchange

    @property
    def order_request_id(self):
        return self.order_request.id

    @property
    def asset(self):
        return self.order_request.asset

    @property
    def base(self):
        return self.order_request.base

    @property
    def order_type(self):
        return self.order_request.order_type

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

    def mark_closed(self):
        self.remaining = 0
        self.status = 'closed'

    def mark_rejected(self):
        self.status = 'rejected'

    def mark_unfilled(self):
        self.status = 'cancelled'
        self.cost = 0

    @property
    def data(self):
        return {"id": self.id, "local_id": self.local_id, "time": self.time,
                "asset": self.asset, "base": self.base,
                "order_request_id": self.order_request_id,
                "exchange": self.exchange,
                "quantity": self.quantity, "remaining": self.remaining,
                "order_cost": self.order_cost, "fill_price": self.fill_price,
                "order_type": self.order_type, "status": self.status,
                "commission": self.commission, "updated_at": self.updated_at,
                "commission_asset": self.commission_asset}

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.data)
