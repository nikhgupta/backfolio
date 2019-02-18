class NotifierMixin(object):
    def after_order_created_done(self, event):
        """
        Once an order has been created, if the order has an ID (usually,
        in live mode), notify all connected services about the order.
        """
        order = event.item
        if order.id and self.debug:
            self.notify(
                "  Created %4s %s order with ID %s for %0.8f %s at %.8f %s" % (
                   order.side, order.order_type, order.id, abs(order.quantity),
                   order.asset, order.fill_price, order.base),
                formatted=True, now=event.item.time, publish=False)

    def after_order_rejected(self, event):
        """
        Once an order has been rejected,
        notify all connected services about the order.
        This notifies in backtest mode too.
        """
        order = event.item
        if self.debug:
            self.notify(
                " Rejected %4s %s order for %0.8f %s at %.8f %s (Reason: %s)"
                % (order.side, order.order_type, abs(order.quantity),
                   order.asset, order.fill_price, order.base, event.reason),
                formatted=True, now=event.item.time, publish=False)
