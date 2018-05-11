""" Various Event class for CryptoFolio """


class BaseEvent(object):
    """
    BaseEvent is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    """
    def __init__(self, item, priority=0):
        self._item = item
        self._priority = priority

    @property
    def priority(self):
        return self._priority

    @property
    def item(self):
        return self._item

    @property
    def data(self):
        return self.item.data

    def __eq__(self, other):
        return (self.data == other.data and self.__class__ == other.__class__)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.item.__repr__())


class TickUpdateEvent(BaseEvent):
    """
    Event emitted when a new market data is obtained from the datacenter.
    """
    def __init__(self, tick):
        super().__init__(tick, priority=1)


class StrategyAdviceEvent(BaseEvent):
    """
    Event emitted when a new advice is given by a strategy.
    """
    def __init__(self, advice):
        super().__init__(advice, priority=1)


class OrderRequestedEvent(BaseEvent):
    """
    Event emitted when a new order is placed by portfolio for an asset.
    """
    def __init__(self, order_request):
        super().__init__(order_request, priority=1)


class OrderCreatedEvent(BaseEvent):
    """
    Event emitted when a new order is created by the broker for an asset.
    """

    def __init__(self, order):
        super().__init__(order, priority=4)


class OrderFilledEvent(OrderCreatedEvent):
    pass


class OrderRejectedEvent(OrderCreatedEvent):
    """
    Event emitted when a new order is rejected by the broker for an asset.
    """

    def __init__(self, order, reason=None):
        super().__init__(order)
        self.reason = reason

    @property
    def data(self):
        return {**super().data, **{"reason": self.reason}}

    def __repr__(self):
        return "%s(%s, Reason: %s)" % (
            self.__class__.__name__, self.item.__repr__(), self.reason)


class OrderUnfilledEvent(OrderRejectedEvent):
    pass
